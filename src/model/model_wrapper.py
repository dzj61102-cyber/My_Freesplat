from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import moviepy.editor as mpy
import torch
import wandb
from einops import pack, rearrange, repeat
from jaxtyping import Float
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor, nn, optim

from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from ..global_cfg import get_cfg
from ..loss import Loss
from ..misc.benchmarker import Benchmarker
from ..misc.image_io import prep_image, save_image
from ..misc.LocalLogger import LOG_PATH, LocalLogger
from ..misc.step_tracker import StepTracker
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from ..visualization.color_map import apply_color_map_to_image
from ..visualization.layout import add_border, hcat, vcat
from ..visualization.validation_in_3d import render_cameras, render_projections
from .decoder.decoder import Decoder, DepthRenderingMode, DecoderOutput
from .encoder import Encoder
from .encoder.encoder_freesplat import UseDepthMode
from .types import Gaussians
from .encoder.visualization.encoder_visualizer import EncoderVisualizer


from PIL import Image, ImageFont, ImageDraw
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import mmcv
import os
import json


def convert_array_to_pil(depth_map, no_text=False):
    # Input: depth_map -> HxW numpy array with depth values 
    # Output: colormapped_im -> HxW numpy array with colorcoded depth values
    depth_map = np.asarray(depth_map)
    if depth_map.ndim == 3 and depth_map.shape[0] == 1:
        depth_map = depth_map[0]
    elif depth_map.ndim == 3 and depth_map.shape[-1] == 1:
        depth_map = depth_map[..., 0]
    elif depth_map.ndim != 2:
        raise ValueError(f"Expected depth map with 2 dims, got shape {depth_map.shape}")

    mask = np.isfinite(depth_map) & (depth_map > 0)
    if not np.any(mask):
        colormapped_im = np.full((*depth_map.shape, 3), 255, dtype=np.uint8)
        return colormapped_im

    disp_map = np.zeros_like(depth_map, dtype=np.float32)
    disp_map[mask] = 1.0 / depth_map[mask]
    vmax = np.percentile(disp_map[mask], 95)
    vmin = np.percentile(disp_map[mask], 5)
    normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    mask_ = np.repeat(np.expand_dims(mask,-1), 3, -1)
    colormapped_im = (mapper.to_rgba(disp_map)[:, :, :3] * 255).astype(np.uint8)
    colormapped_im[~mask_] = 255
    min_depth, max_depth = depth_map[mask].min(), depth_map[mask].max()
    image = Image.fromarray(colormapped_im)
    if not no_text:
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 40)
        draw.text((20,20), '[%.2f, %.2f]'%(min_depth, max_depth), (255,255,255), font=font)
        colormapped_im = np.asarray(image)

    return colormapped_im

# Compute RGB metrics at novel views.
def compute_metrics(rgb_gt, rgb):
    rgb = rgb.clip(min=0, max=1)
    psnr = compute_psnr(rgb_gt, rgb)
    if rgb_gt.shape[0] < 100:
        lpips = compute_lpips(rgb_gt, rgb)
    else:
        lpips = torch.tensor(0.0, device=rgb.device)
    ssim = compute_ssim(rgb_gt, rgb)
    num = len(psnr)
    psnr = psnr.mean().item()
    ssim = ssim.mean().item()
    lpips = lpips.mean().item()
    print('psnr:', psnr, 'ssim:', ssim, 'lpips:', lpips, 'num:', num)
    return psnr, lpips, ssim, num

# 可视化代码，剪枝前后直方图和3d图
def save_gaussian_metric_visualizations(
    gaussians,
    output_root: Path,
    scene: str,
    pruning_mode: int,
    importance_scores=None,
    max_points_3d: int = 80000,
    suffix: str = "",
):
    # suffix 用于区分“剪枝前/剪枝后”输出文件名，例如 "_pruned"。
    title_suffix = " (Pruned)" if suffix == "_pruned" else ""
    if gaussians is None:
        return
    if isinstance(gaussians, list):
        if len(gaussians) == 0:
            return
        gaussians = gaussians[0]
    if isinstance(importance_scores, list):
        if len(importance_scores) == 0:
            importance_scores = None
        else:
            # 与 gaussians 一致：当前函数只绘制第 0 个样本。
            importance_scores = importance_scores[0]

    means = gaussians.means
    opacities = gaussians.opacities
    covariances = gaussians.covariances
    if means.ndim == 3:
        means = means[0]
    if opacities.ndim == 2:
        opacities = opacities[0]
    if covariances.ndim == 4:
        covariances = covariances[0]
    if importance_scores is not None and importance_scores.ndim == 2:
        # [B, G] -> [G]，与 means/opacities 对齐。
        importance_scores = importance_scores[0]

    means = means.detach().float().cpu()
    opacities = opacities.detach().float().cpu()
    covariances = covariances.detach().float().cpu()
    valid = torch.isfinite(means).all(dim=1) & torch.isfinite(opacities)
    if pruning_mode == 2:
        valid = valid & torch.isfinite(covariances).flatten(1).all(dim=1)
    if valid.sum().item() == 0:
        return
    means = means[valid]
    opacities = opacities[valid]
    covariances = covariances[valid]

    if pruning_mode in (0, 1):
        # mode 0/1 统一可视化 opacity（mode 1 也是按 opacity 剪枝）。
        metric_values = opacities
        metric_name = "Opacity"#图标题
        metric_prefix = "opacity"#文件标题
        color_min = 0.0
        color_max = 1.0
    elif pruning_mode == 2:
        # mode 2: 启发式重要性分数 = opacity * scale_proxy。
        # scale_proxy 由协方差特征值体积近似而来，反映“尺度贡献”。
        eigvals = torch.linalg.eigvalsh(covariances).clamp_min(0.0)
        scale_proxy = eigvals.prod(dim=-1).pow(1.0 / 6.0)
        metric_values = opacities * scale_proxy
        metric_name = "Opacity*Scale"
        metric_prefix = "opacity_scale"
        color_min = float(torch.quantile(metric_values, 0.01).item())
        color_max = float(torch.quantile(metric_values, 0.99).item())
        if color_max <= color_min:
            color_max = color_min + 1e-6
    elif pruning_mode == 3:
        # mode 3: 预算感知 learned score 可视化。
        # 若上游未提供 importance_scores，则回退到 opacity。
        if importance_scores is None:
            metric_values = opacities
        else:
            metric_values = importance_scores.detach().float().cpu()
            if metric_values.ndim == 2:
                metric_values = metric_values[0]
        metric_values = metric_values[valid]
        metric_name = "Learned Score"
        metric_prefix = "learned_score"
        color_min = float(torch.quantile(metric_values, 0.01).item())
        color_max = float(torch.quantile(metric_values, 0.99).item())
        if color_max <= color_min:
            color_max = color_min + 1e-6
    else:
        raise ValueError(f"Unsupported pruning_mode: {pruning_mode}")

    scene_dir = output_root / scene
    scene_dir.mkdir(parents=True, exist_ok=True)

    metric_np = metric_values.numpy()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(metric_np, bins=100, color="#3B82F6", edgecolor="black", linewidth=0.2)
    ax.set_title(f"Global Gaussian {metric_name} Histogram ({scene}){title_suffix}")
    ax.set_xlabel(metric_name)
    fig.tight_layout()
    fig.savefig(scene_dir / f"gaussian_{metric_prefix}_hist{suffix}.png", dpi=220)
    plt.close(fig)

    n = means.shape[0]
    if n > max_points_3d:
        idx = torch.randperm(n)[:max_points_3d]
        means = means[idx]
        metric_values = metric_values[idx]

    xyz = means.numpy()
    metric_np = metric_values.numpy()
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        c=metric_np,
        cmap="viridis",
        s=1.2,
        alpha=0.9,
        vmin=color_min,
        vmax=color_max,
        linewidths=0.0,
    )
    ax.set_title(f"Global Gaussian {metric_name} in 3D ({scene}){title_suffix}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    cbar = fig.colorbar(scatter, ax=ax, pad=0.08, shrink=0.75)
    cbar.set_label(metric_name)
    fig.tight_layout()
    fig.savefig(scene_dir / f"gaussian_{metric_prefix}_3d{suffix}.png", dpi=260)
    plt.close(fig)

# 获取剪枝分数，输入的importance_scores是学习所得分数
def compute_pruning_scores(
    gaussians: Gaussians,
    pruning_mode: int,
    importance_scores: Float[Tensor, "batch gaussian"] | None = None,
) -> Float[Tensor, "batch gaussian"]:
    # 统一剪枝打分入口：
    # mode=1 -> opacity；mode=2 -> opacity*scale；mode=3 -> learned score。
    if pruning_mode == 1:
        return gaussians.opacities
    if pruning_mode == 2:
        # eigvals shape: [batch, gaussian, 3]
        eigvals = torch.linalg.eigvalsh(gaussians.covariances)
        eigvals = eigvals.clamp_min(0.0)
        scale_proxy = eigvals.prod(dim=-1).pow(1.0 / 6.0)
        return gaussians.opacities * scale_proxy
    if pruning_mode == 3:
        # learned 剪枝模式必须显式提供 importance_scores。
        if importance_scores is None:
            raise ValueError("importance_scores is required when pruning_mode == 3")
        return importance_scores
    raise ValueError(f"Unsupported pruning_mode: {pruning_mode}")

# 核心剪枝函数，基于分数做 hard top-k 剪枝，保证所有高斯属性同步裁剪。
def prune_gaussians(
    gaussians: Gaussians,
    save_ratio: float,
    pruning_mode: int,
    importance_scores: Float[Tensor, "batch gaussian"] | None = None,
) -> Gaussians:
    if pruning_mode == 0 or save_ratio >= 1.0:
        # mode=0 或保留率=1 时，不做剪枝。
        return gaussians

    opacities = gaussians.opacities
    b, g = opacities.shape
    # 采用 floor(save_ratio * G)，与预算定义 K=floor(rho*M) 保持一致。
    # 同时保证至少保留 1 个高斯。
    keep = max(1, int(np.floor(g * save_ratio)))
    # 计算每个高斯的剪枝分数 [B, G]。
    scores = compute_pruning_scores(gaussians, pruning_mode, importance_scores)
    # 每个 batch 独立取 top-k 索引 [B, K]。
    topk = torch.topk(scores, k=keep, dim=1, largest=True).indices

    # 按相同 top-k 索引裁剪全部字段，维持几何/外观/opacity 一致性。
    means = torch.stack(
        [gaussians.means[i].index_select(0, topk[i]) for i in range(b)], dim=0
    )
    covariances = torch.stack(
        [gaussians.covariances[i].index_select(0, topk[i]) for i in range(b)], dim=0
    )
    harmonics = torch.stack(
        [gaussians.harmonics[i].index_select(0, topk[i]) for i in range(b)], dim=0
    )
    opacities = torch.stack(
        [gaussians.opacities[i].index_select(0, topk[i]) for i in range(b)], dim=0
    )
    return Gaussians(means, covariances, harmonics, opacities)


def prune_gaussians_container(
    gaussians,
    save_ratio: float,
    pruning_mode: int,
    importance_scores=None,
):
    # 兼容 Gaussians 与 list[Gaussians] 两种输入。
    if pruning_mode == 0 or save_ratio >= 1.0:
        return gaussians
    if isinstance(gaussians, list):
        # list 场景下，importance_scores 也按样本一一对齐。
        return [
            prune_gaussians(
                gs,
                save_ratio,
                pruning_mode,
                None if importance_scores is None else importance_scores[i],
            )
            for i, gs in enumerate(gaussians)
        ]
    return prune_gaussians(gaussians, save_ratio, pruning_mode, importance_scores)


def count_gaussians(gaussians) -> int:
    if isinstance(gaussians, list):
        return sum(int(gs.means.shape[1]) for gs in gaussians)
    return int(gaussians.means.shape[1])

# Compute Depth metrics at novel views.
def depth_render_metrics(prediction, batch) -> Float[Tensor, ""]:
    if not 'depth' in batch['target']:
        return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
    target = batch['target']['depth'].squeeze(2)
    gt_bN = rearrange(target.clone(), 'b v h w -> (b v) (h w)')
    pred_bN = rearrange(prediction.depth.clone(), 'b v h w -> (b v) (h w)')
    mask = gt_bN > 0.5
    gt_bN[~mask] = torch.nan
    pred_bN[~mask] = torch.nan
    abs_rel_b = torch.nanmean(torch.abs(gt_bN - pred_bN) / gt_bN, dim=1).mean()
    abs_diff_b = torch.nanmean(torch.abs(gt_bN - pred_bN), dim=1).mean()
    thresh_bN = torch.max(torch.stack([(gt_bN / pred_bN), (pred_bN / gt_bN)], 
                                                            dim=2), dim=2)[0]
    a25_val = (thresh_bN < (1.0+0.25)     ).float()
    a25_val[~mask] = torch.nan
    delta_25 = torch.nanmean(a25_val, dim=1).mean()

    a10_val = (thresh_bN < (1.0+0.1)     ).float()
    a10_val[~mask] = torch.nan
    delta_10 = torch.nanmean(a10_val, dim=1).mean()
    return abs_diff_b, abs_rel_b, delta_25, delta_10


@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    cosine_lr: bool


@dataclass
class TestCfg:
    output_path: Path
    # 测试阶段保留比例（0,1]。当 mode=3 且该值为 1.0 时，会在 test_step 中回退到 encoder 返回的 prune_keep_ratio。
    save_ratio: float = 1.0
    # 剪枝模式：
    # 0=不剪枝，1=opacity，2=opacity*scale_proxy，3=learned importance score。
    pruning_mode: int = 0


@dataclass
class TrainCfg:
    depth_mode: DepthRenderingMode | None
    load_depth: UseDepthMode | None
    extended_visualization: bool
    has_depth: bool = False
    train_only_importance_head: bool = False #仅训练 Importance Head 开关配置项


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass


class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    encoder: nn.Module
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
        losses: list[Loss],
        step_tracker: StepTracker | None,
        cfg_dict,
        run_dir,
        num_context_views: int = 2,
        dataset_name: str = 'scannet',
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker
        self.run_dir = run_dir
        self.num_context_views = num_context_views
        self.dataset_name = dataset_name

        # Set up the model.
        self.encoder = encoder
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        self.data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)
        self.trainable_param_count = 0

        # This is used for testing.
        self.benchmarker = Benchmarker()

        self.losses_log = {}
        # 先全冻结，再仅解冻 self.encoder.importance_head
        if self.train_cfg.train_only_importance_head:
            for param in self.parameters():
                param.requires_grad = False
            importance_head = getattr(self.encoder, "importance_head", None)
            if importance_head is None:
                raise ValueError(
                    "train.train_only_importance_head=true but encoder has no importance_head."
                )
            for param in importance_head.parameters():
                param.requires_grad = True
            # 打印可训练参数名预览   
            trainable_names = [
                name for name, param in self.named_parameters() if param.requires_grad
            ]
            preview_count = min(12, len(trainable_names))
            print(
                f"[FreezeCheck] train_only_importance_head=True, "
                f"trainable_tensors={len(trainable_names)}"
            )
            for name in trainable_names[:preview_count]:
                print(f"[FreezeCheck] trainable: {name}")
            if len(trainable_names) > preview_count:
                print(
                    f"[FreezeCheck] ... and {len(trainable_names) - preview_count} more trainable tensors"
                )
        self.loss_total = []
        self.metrics = {}
        for metric in ['psnr', 'lpips', 'ssim']:
            self.metrics[metric] = []
        self.num_evals = []
        self.test_scene_list = []
        self.test_fvs_list = []

        for k1 in cfg_dict:
            try:
                keys = cfg_dict[k1].keys()
                print(f'{k1}:')
                for k2 in keys:
                    print(f'    {k2}: {cfg_dict[k1][k2]}')
            except:
                print(f'{k1}: {cfg_dict[k1]}')

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print(f"Skip loading parameter: {k}, "
                                f"required shape: {model_state_dict[k].shape}, "
                                f"loaded shape: {state_dict[k].shape}")
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                print(f"Dropping parameter {k}")
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)

    def training_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        B, _, _, h, w = batch["target"]["image"].shape
        
        encoder_results = self.encoder(batch["context"], self.global_step, False, is_testing=False)
        
        gaussians = encoder_results['gaussians']

        if not isinstance(gaussians, list):
            output = self.decoder.forward(
                gaussians,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h, w),
                depth_mode=self.train_cfg.depth_mode,
            )
        else:
            output_list = []
            for i, gs in enumerate(gaussians):
                output_list.append(self.decoder.forward(
                    gs,
                    batch["target"]["extrinsics"][i:i+1],
                    batch["target"]["intrinsics"][i:i+1],
                    batch["target"]["near"][i:i+1],
                    batch["target"]["far"][i:i+1],
                    (h, w),
                    depth_mode=self.train_cfg.depth_mode,
                ))
            output = DecoderOutput(None, None)
            output.color = torch.cat([x.color for x in output_list], dim=0)
            try:
                output.depth = torch.cat([x.depth for x in output_list], dim=0)
            except:
                pass
        output_dr = None
        target_gt = batch["target"]["image"]

        psnr_probabilistic = compute_psnr(
            rearrange(target_gt, "b v c h w -> (b v) c h w"),
            rearrange(output.color, "b v c h w -> (b v) c h w"),
        )
        psnr = psnr_probabilistic.mean()
        self.log("train/psnr", psnr, on_step=True, on_epoch=True, sync_dist=True, logger=True)

        total_loss = 0
        for loss_fn in self.losses:
            # 主渲染损失（如 MSE / LPIPS），由配置中的 loss 列表决定。
            loss = loss_fn.forward(output, batch, gaussians, encoder_results, self.global_step, output_dr)
            self.log(f"loss/{loss_fn.name}", loss, on_step=True, on_epoch=True, sync_dist=True, logger=True)
            total_loss = total_loss + loss
            self.losses_log[loss_fn.name] = self.losses_log.get(loss_fn.name, [])
            self.losses_log[loss_fn.name].append(loss)
        self.log("loss/total", total_loss, on_step=True, on_epoch=True, sync_dist=True, logger=True)
        self.loss_total.append(total_loss)
        context_indices = batch['context']['index'].tolist()

        if batch_idx %10 == 0:
            to_print = f"train step {self.global_step}; "+\
                       f"scene = {batch['scene']}; " + \
                       f"context = {context_indices}; " +\
                       f"loss = {torch.mean(torch.tensor(self.loss_total)):.6f} "+\
                       f"psnr = {torch.mean(torch.tensor(psnr)):.2f}"
            for name in self.losses_log:
                to_print = to_print + f' loss_{name} = {torch.mean(torch.tensor(self.losses_log[name])):.6f}'
            if 'gs_ratio' in encoder_results:
                to_print = to_print + f' gs_ratio = {torch.mean(torch.tensor(encoder_results["gs_ratio"])):.6f}'
            print(to_print)
            # 训练循环中的梯度范数检查，特别关注 importance_head 和一个冻结参数的梯度。
            if self.train_cfg.train_only_importance_head:
                grad_head = None
                if hasattr(self.encoder, "importance_head"):
                    head_grads = [
                        p.grad.detach().norm()
                        for p in self.encoder.importance_head.parameters()
                        if p.grad is not None
                    ]
                    if len(head_grads) > 0:
                        grad_head = torch.stack(head_grads).mean().item()

                grad_frozen_example = None
                for name, param in self.named_parameters():
                    if not param.requires_grad:
                        if param.grad is None:
                            grad_frozen_example = 0.0
                        else:
                            grad_frozen_example = float(param.grad.detach().norm().item())
                        frozen_name = name
                        break
                if grad_frozen_example is None:
                    frozen_name = "None"
                    grad_frozen_example = -1.0
                print(
                    f"[FreezeCheck] step={self.global_step} "
                    f"grad_importance_head_mean={grad_head if grad_head is not None else -1.0:.6e} "
                    f"frozen_example={frozen_name} "
                    f"grad_norm={grad_frozen_example:.6e}"
                )
            self.losses_log = {}
            self.loss_total = []
            
            
        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        return total_loss

    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        # keep_ratio（save_ratio）取值要求：必须在 (0,1]。
        if not (0.0 < self.test_cfg.save_ratio <= 1.0):
            raise ValueError(
                f"test.save_ratio must be in (0, 1], got {self.test_cfg.save_ratio}"
            )
        # 推理阶段支持 4 种模式，mode=3 为 learned importance 硬剪枝。
        if self.test_cfg.pruning_mode not in (0, 1, 2, 3):
            raise ValueError(
                f"test.pruning_mode must be one of [0, 1, 2, 3], got {self.test_cfg.pruning_mode}"
            )

        b, v, _, h, w = batch["target"]["image"].shape
        assert b == 1
        if batch_idx % 100 == 0:
            print(f"Test step {batch_idx:0>6}.")
    
        # Render Gaussians.
        with self.benchmarker.time("encoder"):
            encoder_results = self.encoder(
                batch["context"],
                self.global_step,
                deterministic=False,
                is_testing=True,
                export_ply=self.encoder_visualizer.cfg.export_ply,
            )
            gaussians_full = encoder_results['gaussians']

            # 测试阶段输出每个场景的体素尺寸统计。
            if (
                "test_voxel_size" in encoder_results
                and "test_voxel_size_raw" in encoder_results
                and "test_bbox_diag" in encoder_results
            ):
                for i, scene in enumerate(batch["scene"]):
                    voxel_size = float(encoder_results["test_voxel_size"][i].detach().cpu().item())
                    v_raw = float(encoder_results["test_voxel_size_raw"][i].detach().cpu().item())
                    bbox_diag = float(encoder_results["test_bbox_diag"][i].detach().cpu().item())
                    v_min = 0.0001 * bbox_diag
                    v_max = 0.0003 * bbox_diag
                    print(
                        f"[VoxelSize] scene={scene} "
                        f"v={voxel_size:.8f} v_raw={v_raw:.8f} "
                        f"v_min={v_min:.8f} v_max={v_max:.8f}"
                    )

            # encoder 输出的 learned importance 分数（mode=3 使用）。
            importance_scores = encoder_results.get("gaussians_importance_scores", None)
            if self.test_cfg.pruning_mode == 3:
                # mode=3 时优先使用 test.save_ratio；
                # 若用户保持默认 1.0，则回退到训练预算 prune_keep_ratio。
                keep_ratio = self.test_cfg.save_ratio
                if keep_ratio >= 1.0 and "prune_keep_ratio" in encoder_results:
                    keep_ratio = float(encoder_results["prune_keep_ratio"])
                # 基于 learned score 执行 hard top-k。
                gaussians = prune_gaussians_container(
                    gaussians_full,
                    keep_ratio,
                    self.test_cfg.pruning_mode,
                    importance_scores=importance_scores,
                )
            else:
                # mode=1/2：启发式剪枝路径。
                gaussians = prune_gaussians_container(
                    gaussians_full, self.test_cfg.save_ratio, self.test_cfg.pruning_mode
                )
            rendered_num_gaussians = count_gaussians(gaussians)
            
        with self.benchmarker.time("decoder", num_calls=v):
            if not isinstance(gaussians, list):
                output = self.decoder.forward(
                    gaussians,
                    batch["target"]["extrinsics"],
                    batch["target"]["intrinsics"],
                    batch["target"]["near"],
                    batch["target"]["far"],
                    (h, w),
                    depth_mode='depth',
                )
            else:
                output_list = []
                n_targets = batch["target"]["extrinsics"].shape[1]
                for i, gs in enumerate(gaussians):
                    output = []
                    for j in range(np.ceil(n_targets/50).astype(int)):
                        output.append(self.decoder.forward(
                            gs,
                            batch["target"]["extrinsics"][i:i+1, j*50:(j+1)*50],
                            batch["target"]["intrinsics"][i:i+1, j*50:(j+1)*50],
                            batch["target"]["near"][i:i+1, j*50:(j+1)*50],
                            batch["target"]["far"][i:i+1, j*50:(j+1)*50],
                            (h, w),
                            depth_mode='depth',
                        ))
                    now = DecoderOutput(None, None)
                    now.color = torch.cat([x.color for x in output], dim=1)
                    now.depth = torch.cat([x.depth for x in output], dim=1)
                    output_list.append(now)
                output = DecoderOutput(None, None)
                output.color = torch.cat([x.color for x in output_list], dim=0)
                try:
                    output.depth = torch.cat([x.depth for x in output_list], dim=0)
                except:
                    pass

        # Save images.
        (scene,) = batch["scene"]
        print(f'processing {scene}')
        print(f"[RenderGaussians] scene={scene} num_gaussians={rendered_num_gaussians}")
        self.test_scene_list.append(scene)
        name = get_cfg()["wandb"]["name"]
        path = self.test_cfg.output_path / name
        save_gaussian_metric_visualizations(
            gaussians_full,
            path,
            scene,
            pruning_mode=self.test_cfg.pruning_mode,
            importance_scores=importance_scores,
            suffix="",
        )
        save_gaussian_metric_visualizations(
            gaussians,
            path,
            scene,
            pruning_mode=self.test_cfg.pruning_mode,
            suffix="_pruned",
        )
        abs_diff, rel_diff, delta_25, delta_10 = depth_render_metrics(output, batch)
        print(f'abs_diff: {abs_diff}, rel_diff: {rel_diff}, delta_25: {delta_25}, delta_10: {delta_10}')
        self.benchmarker.store('depth_abs_diff', float(abs_diff.detach().cpu().numpy()))
        self.benchmarker.store('depth_rel_diff', float(rel_diff.detach().cpu().numpy()))
        self.benchmarker.store('depth_delta_25', float(delta_25.detach().cpu().numpy()))
        self.benchmarker.store('depth_delta_10', float(delta_10.detach().cpu().numpy()))

        try:
            fvs_length = batch["target"]["test_fvs"]
            test_fvs = fvs_length > 0
        except:
            fvs_length = 0
            test_fvs = False
            
        count = 0
        for i, index, fig in zip(range(len(batch["context"]["index"][0])), batch["context"]["index"][0], batch["context"]["image"][0]):
            length = len(encoder_results[f"depth_num0_s-1"][0])
            save_image(fig, path / scene / f"contexts/{index:0>6}.png")
            save_image(torch.from_numpy(convert_array_to_pil(encoder_results[f"depth_num0_s-1"][0][i].cpu().numpy().reshape(h,w), no_text=True).transpose(2,0,1)\
                                        .astype(np.float32)/255).to(batch["context"]["image"][0].device),
                                        path / scene / f"depth_pred/{index:0>6}.png")
            

        
        for i, index in enumerate(batch["target"]["index"][0]):
            depth_render = output.depth[0][i]

            save_image(torch.from_numpy(convert_array_to_pil(depth_render.cpu().numpy(), no_text=True).transpose(2,0,1)\
                                            .astype(np.float32)/255).to(batch["context"]["image"][0].device),
                                            path / scene / f"depth_render/{index:0>6}.png")
            
            if 'depth' in batch["target"]:
                depth_gt = batch["target"]['depth'][0][i]
                
                save_image(torch.from_numpy(convert_array_to_pil(depth_gt.cpu().numpy(), no_text=True).transpose(2,0,1)\
                                                .astype(np.float32)/255).to(batch["context"]["image"][0].device),
                                                path / scene / f"depth_render_gt/{index:0>6}.png")

        for index, color, color_gt in zip(batch["target"]["index"][0], output.color[0], batch["target"]["image"][0]):
            if not test_fvs:
                save_image(color, path / scene / f"color/{index:0>6}.png")
                save_image(color_gt, path / scene / f"color_gt/{index:0>6}.png")
            else:
                if count < batch["target"]["index"][0].shape[0]-fvs_length:
                    save_image(color, path / scene / f"interpolation/{index:0>6}.png")
                    save_image(color_gt, path / scene / f"interapolation_gt/{index:0>6}.png")
                else:
                    save_image(color, path / scene / f"extrapolation/{index:0>6}.png")
                    save_image(color_gt, path / scene / f"extrapolation_gt/{index:0>6}.png")
                count += 1
        
        if not test_fvs:
            psnr, lpips, ssim, num = compute_metrics(batch["target"]["image"][0], output.color[0])
        
            self.benchmarker.store('psnr_inter', float(psnr))
            self.benchmarker.store('lpips_inter', float(lpips))
            self.benchmarker.store('ssim_inter', float(ssim))
            self.benchmarker.store('num_inter', float(num))
            self.benchmarker.store('num_gaussians', encoder_results['num_gaussians'])
            self.benchmarker.store('rendered_num_gaussians', rendered_num_gaussians)
            self.test_fvs_list.append(False)
        else:
            length = batch["target"]["index"][0].shape[0]
            psnr_inter, lpips_inter, ssim_inter, num_inter = compute_metrics(batch["target"]["image"][0][:length-fvs_length], 
                                                                  output.color[0][:length-fvs_length])
            psnr_extra, lpips_extra, ssim_extra, num_extra = compute_metrics(batch["target"]["image"][0][length-fvs_length:],
                                                                  output.color[0][length-fvs_length:])
            
            self.benchmarker.store('psnr_inter', float(psnr_inter))
            self.benchmarker.store('lpips_inter', float(lpips_inter))
            self.benchmarker.store('ssim_inter', float(ssim_inter))
            self.benchmarker.store('num_inter', float(num_inter))
            self.benchmarker.store('psnr_extra', float(psnr_extra))
            self.benchmarker.store('lpips_extra', float(lpips_extra))
            self.benchmarker.store('ssim_extra', float(ssim_extra))
            self.benchmarker.store('num_extra', float(num_extra))
            self.benchmarker.store('num_gaussians', encoder_results['num_gaussians'])
            self.benchmarker.store('rendered_num_gaussians', rendered_num_gaussians)
            self.test_fvs_list.append(True)
        if self.encoder_visualizer is not None:
            for k, image in self.encoder_visualizer.visualize(
                encoder_results, batch["context"], batch_idx, out_path=self.test_cfg.output_path / 'gaussians'
            ).items():
                self.logger.log_image(k, [prep_image(image)], step=self.global_step)

    def on_test_end(self) -> None:
        name = get_cfg()["wandb"]["name"]
        self.benchmarker.dump(self.test_cfg.output_path / name / "benchmark.json")
        self.benchmarker.dump_memory(
            self.test_cfg.output_path / name / "peak_memory.json"
        )
        self.benchmarker.dump_stats(
            self.test_cfg.output_path / name / "stats.json"
        )
        nan_scenes = []
        for i in range(min(len(self.test_scene_list), len(self.test_fvs_list))):
            if self.test_fvs_list[i]:
                print(self.test_scene_list[i], self.benchmarker.benchmarks['psnr_inter'][i], 
                                self.benchmarker.benchmarks['ssim_inter'][i],
                                self.benchmarker.benchmarks['lpips_inter'][i],
                                self.benchmarker.benchmarks['psnr_extra'][int(np.sum(self.test_fvs_list[:i]))], 
                                self.benchmarker.benchmarks['ssim_extra'][int(np.sum(self.test_fvs_list[:i]))],
                                self.benchmarker.benchmarks['lpips_extra'][int(np.sum(self.test_fvs_list[:i]))],
                                self.benchmarker.benchmarks['depth_abs_diff'][i],
                                self.benchmarker.benchmarks['depth_rel_diff'][i],
                                self.benchmarker.benchmarks['depth_delta_25'][i],
                                self.benchmarker.benchmarks['depth_delta_10'][i])
            else:
                print(self.test_scene_list[i], self.benchmarker.benchmarks['psnr_inter'][i], 
                                self.benchmarker.benchmarks['ssim_inter'][i],
                                self.benchmarker.benchmarks['lpips_inter'][i],
                                self.benchmarker.benchmarks['depth_abs_diff'][i],
                                self.benchmarker.benchmarks['depth_rel_diff'][i],
                                self.benchmarker.benchmarks['depth_delta_25'][i],
                                self.benchmarker.benchmarks['depth_delta_10'][i])
            if np.isnan(self.benchmarker.benchmarks['depth_abs_diff'][i]) or np.isnan(self.benchmarker.benchmarks['depth_rel_diff'][i]) or np.isnan(self.benchmarker.benchmarks['depth_delta_25'][i]) or np.isnan(self.benchmarker.benchmarks['depth_delta_10'][i]):
                nan_scenes.append(self.test_scene_list[i])
        inter_num = np.array(self.benchmarker.benchmarks['num_inter'])
        psnr_inter_avg = (np.array(self.benchmarker.benchmarks['psnr_inter']) * inter_num).sum() / inter_num.sum()
        ssim_inter_avg = (np.array(self.benchmarker.benchmarks['ssim_inter']) * inter_num).sum() / inter_num.sum()
        lpips_inter_avg = (np.array(self.benchmarker.benchmarks['lpips_inter']) * inter_num).sum() / inter_num.sum()
        depth_abs_diff_avg = torch.nanmean(torch.tensor(self.benchmarker.benchmarks['depth_abs_diff'])).cpu().numpy()
        depth_rel_diff_avg = torch.nanmean(torch.tensor(self.benchmarker.benchmarks['depth_rel_diff'])).cpu().numpy()
        depth_delta_25_avg = torch.nanmean(torch.tensor(self.benchmarker.benchmarks['depth_delta_25'])).cpu().numpy()
        depth_delta_10_avg = torch.nanmean(torch.tensor(self.benchmarker.benchmarks['depth_delta_10'])).cpu().numpy()
        mean_encoder_time = np.mean(self.benchmarker.execution_times["encoder"]) if len(self.benchmarker.execution_times["encoder"]) > 0 else float("nan")
        mean_decoder_time = np.mean(self.benchmarker.execution_times["decoder"]) if len(self.benchmarker.execution_times["decoder"]) > 0 else float("nan")
        fps = (1.0 / mean_decoder_time) if np.isfinite(mean_decoder_time) and mean_decoder_time > 0 else float("nan")
        peak_memory_bytes = torch.cuda.memory_stats()["allocated_bytes.all.peak"] if torch.cuda.is_available() else 0.0
        peak_memory_gb = peak_memory_bytes / (1024 ** 3)
        summary_metrics = {}
        def log_metric(key, value):
            value = float(value)
            print(f'{key}: {value:.3f}')
            summary_metrics[key] = f"{value:.3f}"

        log_metric('psnr_inter_avg', psnr_inter_avg)
        log_metric('ssim_inter_avg', ssim_inter_avg)
        log_metric('lpips_inter_avg', lpips_inter_avg)
        log_metric('depth_abs_diff_avg', depth_abs_diff_avg)
        log_metric('depth_rel_diff_avg', depth_rel_diff_avg)
        log_metric('depth_delta_25_avg', depth_delta_25_avg)
        log_metric('depth_delta_10_avg', depth_delta_10_avg)
        try:
            extra_num = np.array(self.benchmarker.benchmarks['num_extra'])
            psnr_extra_avg = (np.array(self.benchmarker.benchmarks['psnr_extra']) * extra_num).sum() / extra_num.sum()
            ssim_extra_avg = (np.array(self.benchmarker.benchmarks['ssim_extra']) * extra_num).sum() / extra_num.sum()
            lpips_extra_avg = (np.array(self.benchmarker.benchmarks['lpips_extra']) * extra_num).sum() / extra_num.sum()
            log_metric('psnr_extra_avg', psnr_extra_avg)
            log_metric('ssim_extra_avg', ssim_extra_avg)
            log_metric('lpips_extra_avg', lpips_extra_avg)
        except:
            pass
        rendered_num_gaussians_avg = self.benchmarker.benchmarks.get(
            "rendered_num_gaussians_avg",
            self.benchmarker.benchmarks["num_gaussians_avg"] * self.test_cfg.save_ratio,
        )
        log_metric('num_gaussians_avg', self.benchmarker.benchmarks["num_gaussians_avg"])
        log_metric('rendered_num_gaussians_avg', rendered_num_gaussians_avg)
        log_metric('peak_memory_gb', peak_memory_gb)
        log_metric('mean_encoder_time', mean_encoder_time)
        log_metric('fps', fps)
        summary_metrics_path = self.test_cfg.output_path / "terminal_metrics.json"
        summary_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_metrics_path.open("w") as f:
            json.dump(summary_metrics, f, indent=2)
        if len(nan_scenes) > 0:
            print('nan_depth_scenes:', nan_scenes)

    @rank_zero_only
    def validation_step(self, batch, batch_idx):# 训练中的验证
        batch: BatchedExample = self.data_shim(batch)
        context_indices = batch['context']['index'].tolist()

        (scene,) = batch["scene"]
        print(
            f"validation step {self.global_step}; "
            f"scene = {batch['scene']}; "
            f"context = {context_indices}"
        )

        # Render Gaussians.
        b, _, _, h, w = batch["target"]["image"].shape
        assert b == 1
        encoder_probabilistic_results = self.encoder(
            batch["context"],
            self.global_step,
            deterministic=False,
            is_testing=True,
        )
        gaussians_probabilistic_full = encoder_probabilistic_results['gaussians']
        # 验证阶段强制使用与测试 mode=3 一致的 hard 剪枝：
        # 1) 分数来源：encoder 输出的 learned importance_scores
        # 2) 保留比例：训练配置 keep_ratio（由 encoder 返回 prune_keep_ratio）
        importance_scores = encoder_probabilistic_results.get("gaussians_importance_scores", None)
        keep_ratio = float(
            encoder_probabilistic_results.get(
                "prune_keep_ratio",
                getattr(getattr(self.encoder, "cfg", None), "budget_pruning", None).keep_ratio
                if getattr(getattr(self.encoder, "cfg", None), "budget_pruning", None) is not None
                else 0.8,
            )
        )
        gaussians_probabilistic = prune_gaussians_container(
            gaussians_probabilistic_full,
            keep_ratio,
            pruning_mode=3,
            importance_scores=importance_scores,
        )
        if not isinstance(gaussians_probabilistic, list):
            output_probabilistic = self.decoder.forward(
                gaussians_probabilistic,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h, w),
                depth_mode='depth',
            )
        else:
            output_probabilistic_list = []
            for i, gs in enumerate(gaussians_probabilistic):
                output_probabilistic_list.append(self.decoder.forward(
                    gs,
                    batch["target"]["extrinsics"][i:i+1],
                    batch["target"]["intrinsics"][i:i+1],
                    batch["target"]["near"][i:i+1],
                    batch["target"]["far"][i:i+1],
                    (h, w),
                    depth_mode='depth',
                ))
            output_probabilistic = DecoderOutput(None, None)
            output_probabilistic.color = torch.cat([x.color for x in output_probabilistic_list], dim=0)
            try:
                output_probabilistic.depth = torch.cat([x.depth for x in output_probabilistic_list], dim=0)
            except:
                pass
        output_dr = None
        rgb_probabilistic = output_probabilistic.color[0]

        # Compute validation metrics.
        rgb_gt = batch["target"]["image"][0]
        tag, rgb = "probabilistic", rgb_probabilistic
        psnr, lpips, ssim, num = compute_metrics(rgb_gt, rgb)
        self.log(f"val/psnr_{tag}", psnr)
        if tag == "probabilistic":
            self.log("val_psnr_probabilistic", psnr, prog_bar=True, logger=True)
        self.log(f"val/lpips_{tag}", lpips)
        self.log(f"val/ssim_{tag}", ssim)
        abs_diff, rel_diff, delta_25, delta_10 = depth_render_metrics(output_probabilistic, batch)
        self.log(f"val/depth_abs_diff_{tag}", abs_diff)
        self.log(f"val/depth_rel_diff_{tag}", rel_diff)
        self.log(f"val/depth_delta_25_{tag}", delta_25)
        self.log(f"val/depth_delta_10_{tag}", delta_10)
        for metric in ['psnr', 'lpips', 'ssim']:
            self.metrics[metric].append(eval(metric))
        self.num_evals.append(num)

        # Construct comparison image.
        if not self.train_cfg.has_depth:
            context_figs = []
            for fig in batch["context"]["image"][0]:
                context_figs.append(fig)
            if 'depth' in batch["context"]:
                for fig in batch["context"]["depth"][0]:
                    context_figs.append(torch.from_numpy(convert_array_to_pil(fig.cpu().numpy()[0]).transpose(2,0,1)\
                                                        .astype(np.float32)/255).to(batch["context"]["image"][0].device))
            comparison = hcat(
                add_label(vcat(*context_figs), "Context"),
                add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
                add_label(vcat(*rgb_probabilistic), "Target (Probabilistic)"),
            )
        else:
            context_figs = []
            
            context_depth_figs = []
            pred_depth_figs = []
            for fig in batch["context"]["image"][0]:
                context_figs.append(fig)
                length = len(encoder_probabilistic_results[f"depth_num0_s-1"][0])
                for i in range(length):
                    try:
                        context_depth_figs.append(torch.from_numpy(convert_array_to_pil(mmcv.imresize(batch["context"][f"depth_s-1"][0][i][0].cpu().numpy(), (w,h),interpolation='nearest')).transpose(2,0,1)\
                                                                .astype(np.float32)/255).to(batch["context"]["image"][0].device))
                    except:
                        pass
                    try:
                        pred_depth_figs.append(torch.from_numpy(convert_array_to_pil(encoder_probabilistic_results[f"depth_num0_s-1"][0][i].cpu().numpy().reshape(h,w)).transpose(2,0,1)\
                                                            .astype(np.float32)/255).to(batch["context"]["image"][0].device))
                    except:
                        pred_depth_figs.append(torch.from_numpy(convert_array_to_pil(mmcv.imresize(encoder_probabilistic_results[f"depth_num0_s-1"][0][i].cpu().numpy().reshape(h//(2**(s+1)), w//(2**(s+1))), (w,h),interpolation='nearest'))\
                                                                        .transpose(2,0,1).astype(np.float32)/255).to(batch["context"]["image"][0].device))

            try:
                comparison = hcat(
                add_label(vcat(*context_figs), "Context"),
                add_label(vcat(*context_depth_figs), "Context GT Depths"),
                add_label(vcat(*pred_depth_figs), "Depths Predictions"),
                add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
                add_label(vcat(*rgb_probabilistic), "Target (Predictions)"),
            )
            except:
                comparison = hcat(
                    add_label(vcat(*context_figs), "Context"),
                    add_label(vcat(*pred_depth_figs), "Depths Predictions"),
                    add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
                    add_label(vcat(*rgb_probabilistic), "Target (Predictions)"),
                )
        self.logger.log_image(
            "comparison",
            [prep_image(add_border(comparison))],
            step=self.global_step,
            caption=batch["scene"],
        )
  

        if self.encoder_visualizer is not None:
            for k, image in self.encoder_visualizer.visualize(
                encoder_probabilistic_results, batch["context"], self.global_step, out_path=self.test_cfg.output_path
            ).items():
                self.logger.log_image(k, [prep_image(image)], step=self.global_step)


    def on_validation_end(self) -> None:
        with open(self.run_dir + "/val_metrics.txt", "a") as f:
            line = '' 
            for metric in ['psnr', 'lpips', 'ssim']:
                try:
                    line = line + f'{metric}=' + str((np.array(self.metrics[metric])*np.array(self.num_evals)).sum() / np.array(self.num_evals).sum()) + ' '
                except:
                    pass
            f.write(line + '\n')
            print(line)
        for metric in ['psnr', 'lpips', 'ssim']:
            self.metrics[metric] = []
            self.num_evals = []

    @rank_zero_only
    def render_video_wobble(self, batch: BatchedExample) -> None:
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            extrinsics = generate_wobble(
                batch["context"]["extrinsics"][:, 0],
                delta * 0.25,
                t,
            )
            intrinsics = repeat(
                batch["context"]["intrinsics"][:, 0],
                "b i j -> b v i j",
                v=t.shape[0],
            )
            return extrinsics, intrinsics

        return self.render_video_generic(batch, trajectory_fn, "wobble", num_frames=60)

    @rank_zero_only
    def render_video_interpolation(self, batch: BatchedExample) -> None:
        _, v, _, _ = batch["context"]["extrinsics"].shape

        def trajectory_fn(t):
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                batch["context"]["extrinsics"][0, 1]
                if v == 2
                else batch["target"]["extrinsics"][0, 0],
                t,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                batch["context"]["intrinsics"][0, 1]
                if v == 2
                else batch["target"]["intrinsics"][0, 0],
                t,
            )
            return extrinsics[None], intrinsics[None]

        return self.render_video_generic(batch, trajectory_fn, "rgb")

    @rank_zero_only
    def render_video_interpolation_exaggerated(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            tf = generate_wobble_transformation(
                delta * 0.5,
                t,
                5,
                scale_radius_with_t=False,
            )
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                batch["context"]["extrinsics"][0, 1]
                if v == 2
                else batch["target"]["extrinsics"][0, 0],
                t * 5 - 2,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                batch["context"]["intrinsics"][0, 1]
                if v == 2
                else batch["target"]["intrinsics"][0, 0],
                t * 5 - 2,
            )
            return extrinsics @ tf, intrinsics[None]

        return self.render_video_generic(
            batch,
            trajectory_fn,
            "interpolation_exagerrated",
            num_frames=300,
            smooth=False,
            loop_reverse=False,
        )

    @rank_zero_only
    def render_video_generic(
        self,
        batch: BatchedExample,
        trajectory_fn: TrajectoryFn,
        name: str,
        num_frames: int = 30,
        smooth: bool = True,
        loop_reverse: bool = True,
    ) -> None:
        # Render probabilistic estimate of scene.
        gaussians_prob = self.encoder(batch["context"], self.global_step, False, is_testing=False)['gaussians']
        gaussians_det = self.encoder(batch["context"], self.global_step, True, is_testing=False)['gaussians']

        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

        extrinsics, intrinsics = trajectory_fn(t)

        _, _, _, h, w = batch["context"]["image"].shape

        # Color-map the result.
        def depth_map(result):
            near = result[result > 0][:16_000_000].quantile(0.01).log()
            far = result.view(-1)[:16_000_000].quantile(0.99).log()
            result = result.log()
            result = 1 - (result - near) / (far - near)
            return apply_color_map_to_image(result, "turbo")

        near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
        far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)
        output_prob = self.decoder.forward(
            gaussians_prob, extrinsics, intrinsics, near, far, (h, w), "depth"
        )
        images_prob = [
            vcat(rgb, depth)
            for rgb, depth in zip(output_prob.color[0], depth_map(output_prob.depth[0]))
        ]
        output_det = self.decoder.forward(
            gaussians_det, extrinsics, intrinsics, near, far, (h, w), "depth"
        )
        images_det = [
            vcat(rgb, depth)
            for rgb, depth in zip(output_det.color[0], depth_map(output_det.depth[0]))
        ]
        images = [
            add_border(
                hcat(
                    add_label(image_prob, "Probabilistic"),
                    add_label(image_det, "Deterministic"),
                )
            )
            for image_prob, image_det in zip(images_prob, images_det)
        ]

        video = torch.stack(images)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* c h w")[0]
        visualizations = {
            f"video/{name}": wandb.Video(video[None], fps=30, format="mp4")
        }

        # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.
        try:
            wandb.log(visualizations)
        except Exception:
            assert isinstance(self.logger, LocalLogger)
            for key, value in visualizations.items():
                tensor = value._prepare_video(value.data)
                clip = mpy.ImageSequenceClip(list(tensor), fps=value._fps)
                dir = LOG_PATH / key
                dir.mkdir(exist_ok=True, parents=True)
                clip.write_videofile(
                    str(dir / f"{self.global_step:0>6}.mp4"), logger=None
                )
    # 仅收集 requires_grad=True 参数，并打印可训练参数量
    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            raise ValueError("No trainable parameters found for optimizer.")
        self.trainable_param_count = sum(p.numel() for p in trainable_params)
        print(f"Trainable parameters: {self.trainable_param_count}")
        optimizer = optim.Adam(trainable_params, lr=self.optimizer_cfg.lr)
        warm_up_steps = self.optimizer_cfg.warm_up_steps
        if self.optimizer_cfg.cosine_lr:
            warm_up = torch.optim.lr_scheduler.OneCycleLR(
                            optimizer, self.optimizer_cfg.lr,
                            self.trainer.max_steps + 1,
                            pct_start=0.001,
                            cycle_momentum=False,
                            anneal_strategy='cos',
                        )
        else:
            warm_up = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                1 / warm_up_steps,
                1,
                total_iters=warm_up_steps,
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": warm_up,
                "interval": "step",
                "frequency": 1,
            },
        }
