# 标准库：数据类与类型标注
from dataclasses import dataclass
from typing import Literal, Optional, List

# 第三方：张量运算与形状重排
import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

# 项目内：数据结构、几何工具、编码器组件
from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from .backbone import Backbone, BackboneCfg
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
from .encoder import Encoder
from .visualization.encoder_visualizer_epipolar_cfg import EncoderVisualizerEpipolarCfg

from .modules.networks import CVEncoder, DepthDecoder
from .modules.cost_volume import AVGFeatureVolumeManager

import timm
from .modules.layers import TensorFormatter

from .modules.networks import GRU

from einops import *

"""FreeSplat 编码器实现。

整体流程：
1) 使用 2D backbone 提取多尺度图像特征；
2) 基于多视角几何构建 cost volume，并解码得到深度/密度/特征；
3) 将像素级预测转换为高斯表示；
4) 在多视角间融合高斯，输出渲染/训练所需结构。
"""


@dataclass
class OpacityMappingCfg:
    # initial/final: 训练早期与后期使用的指数参数（log2 空间）
    # warm_up: 在多少步内从 initial 线性过渡到 final
    initial: float
    final: float
    warm_up: int


UseDepthMode = Literal[
    "depth"
]

def rotation_distance(rotations):
    """计算视角间旋转差异（弧度）。

    Args:
        rotations: [B, V, 3, 3] 的旋转矩阵。
    Returns:
        [V, V] 的两两旋转角距离矩阵（当前实现 squeeze 了 batch 维）。
    """
    # 为两两组合准备维度：R1 对应行视角，R2 对应列视角
    R1 = rotations.unsqueeze(2) 
    R2 = rotations.unsqueeze(1) 
    # 相对旋转 R_rel = R1^T * R2
    R_rel = torch.matmul(R1.transpose(-2, -1), R2) 

    # 由 trace(R)=1+2cos(theta) 恢复夹角，先做 clamp 提升数值稳定性
    trace = torch.diagonal(R_rel, dim1=-2, dim2=-1).sum(-1) 
    trace = torch.clamp(trace, -1, 3)
    angle = torch.acos((trace - 1) / 2)
    return angle.squeeze(0) 

def calculate_distance_matrix(poses):
    """构建视角间综合距离矩阵（平移距离 + 旋转距离）。"""
    # 提取位姿中的平移与旋转部分
    translations = poses[:, :, :3, 3]
    rotations = poses[:, :, :3, :3]
    
    # 欧氏平移距离矩阵 [V, V]
    translation_dist = torch.cdist(translations, translations).squeeze(0)
    
    # 旋转角距离矩阵 [V, V]
    rotation_dist = rotation_distance(rotations)
    
    # 简单相加作为邻近视角选择的代价
    combined_dist = translation_dist + rotation_dist

    return combined_dist

def positional_encoding(positions, freqs, ori=False):
    """对输入做正余弦位置编码（NeRF 常用形式）。

    Args:
        positions: 形状 (..., D) 的输入向量。
        freqs: 频带数量 F，频率为 2^0 ... 2^(F-1)。
        ori: True 时在输出中保留原始 positions。
    Returns:
        编码结果，默认维度为 (..., 2*D*F)；ori=True 时额外拼接原值。
    """
    # 生成频率带 [F]
    freq_bands = (2**torch.arange(freqs).float()).to(positions.device)  # (F,)
    ori_c = positions.shape[-1]
    # 扩展到每个频带：(..., D) -> (..., D*F)
    pts = (positions[..., None] * freq_bands).reshape(positions.shape[:-1] +
                                                      (freqs * positions.shape[-1], ))  # (..., DF)
    if ori:
        # 输出 [x, sin(wx), cos(wx)]
        pts = torch.cat([positions, torch.sin(pts), torch.cos(pts)], dim=-1).reshape(pts.shape[:-1]+(pts.shape[-1]*2+ori_c,))
    else:
        # 输出 [sin(wx), cos(wx)]
        pts = torch.stack([torch.sin(pts), torch.cos(pts)], dim=-1).reshape(pts.shape[:-1]+(pts.shape[-1]*2,))
    return pts


def set_bn_eval(m):
    # 只针对 BN 层调整模式；其余层保持当前模式
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        # 这里显式调用 train()：保留 BN 的训练态统计/行为
        m.train()

@dataclass
class EncoderFreeSplatCfg:
    # name: 注册名；d_feature/num_surfaces: 特征维与每像素高斯面数
    name: Literal["freesplat"]
    d_feature: int
    num_surfaces: int
    # backbone/visualizer/gaussian_adapter: 子模块配置
    backbone: BackboneCfg
    visualizer: EncoderVisualizerEpipolarCfg
    gaussian_adapter: GaussianAdapterCfg
    opacity_mapping: OpacityMappingCfg
    
    # 深度离散采样数量（cost volume 深度 bin）
    num_depth_candidates: int = 64
    # 每个目标视角参与匹配的上下文视角数
    num_views: int = 2
    # 训练/推理默认输入分辨率
    image_H: int = 384
    image_W: int = 512

    # 深度采样是否在 log 空间均匀
    log_planes: bool = True

# 核心改动开始
class EncoderFreeSplat(Encoder[EncoderFreeSplatCfg]):
    # backbone: 主干特征提取器
    backbone: Backbone
    # backbone_projection: 预留的 backbone 投影头（当前文件中未显式使用）
    backbone_projection: nn.Sequential
    # to_gaussians: 特征到高斯参数的映射头
    to_gaussians: nn.Sequential
    # gaussian_adapter: 几何解码器，把参数转为 3D 高斯对象。。。。。。。
    gaussian_adapter: GaussianAdapter
    # high_resolution_skip: 从 RGB 提取高分辨率细节的跳连分支
    high_resolution_skip: nn.Sequential

    def __init__(self, cfg: EncoderFreeSplatCfg, depth_range=[0.5, 15.0]) -> None:
        """初始化网络结构与几何相关模块。

        Args:
            cfg: 编码器配置。
            depth_range: 深度范围 [near, far]，用于深度解码器离散采样。
        """
        # 初始化父类并注册配置
        super().__init__(cfg)
        # 全局激活函数，供多处子模块复用
        activation_func = nn.ReLU()

        # 保存深度范围供深度解码/可视化等后续步骤使用
        self.depth_range = depth_range

        # 将网络预测参数（特征、深度）转换为 3D 高斯属性（均值/协方差/谐波/不透明度等）
        self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        # 主干特征提取器：EfficientNetV2，仅返回多尺度特征图
        self.backbone = timm.create_model(
                                        "tf_efficientnetv2_s_in21ft1k", 
                                        pretrained=True, 
                                        features_only=True,
                                    )

        # 记录每个尺度输出通道数，供后续编码器/解码器构建网络宽度
        self.backbone.num_ch_enc = self.backbone.feature_info.channels()
        
        # 高频细节跳连分支：把 RGB 投影到 64 通道并逐级下采样
        self.high_resolution_skip = nn.ModuleList(
                                        [nn.Sequential(
                                            nn.Conv2d(3, 64, 7, 1, 3),
                                            activation_func,
                                        ),
                                        nn.Sequential(
                                            nn.Conv2d(3, 64, 6, 2, 2),
                                            activation_func,
                                        ),
                                        nn.Sequential(
                                            nn.Conv2d(3, 64, 8, 4, 2),
                                            activation_func,
                                        ),
                                        nn.Sequential(
                                            nn.Conv2d(3, 64, 16, 8, 4),
                                            activation_func,
                                        ),
                                        nn.Sequential(
                                            nn.Conv2d(3, 64, 32, 16, 8),
                                            activation_func,
                                        )]
                                    )

        # 将融合后的每点64维特征映射成高斯参数：
        # 每个 surface 预测 2 + d_in 维（通常含密度/其他几何外观参数）
        self.to_gaussians = nn.Sequential(
            activation_func,
            nn.Linear(
                64,
                cfg.num_surfaces * (2 + self.gaussian_adapter.d_in),
            ),
        )

        # 缓存高斯参数总通道数
        self.gausisans_ch = cfg.num_surfaces * (2 + self.gaussian_adapter.d_in)
        
        # cost volume 构建器：在 1/4 分辨率下做多视角特征匹配
        self.cost_volume = AVGFeatureVolumeManager(matching_height=self.cfg.image_H//4, 
                                                    matching_width=self.cfg.image_W//4,
                                                    num_depth_bins=self.cfg.num_depth_candidates,
                                                    matching_dim_size=48,)
        # 对 cost volume + backbone 特征做多尺度编码
        self.cv_encoder = CVEncoder(num_ch_cv=self.cfg.num_depth_candidates,
                                    num_ch_enc=self.backbone.num_ch_enc[1:],
                                    num_ch_outs=[64, 128, 256, 384])
        # depth decoder 输入通道 = 最浅层 backbone + CV encoder 多尺度输出
        dec_num_input_ch = (self.backbone.num_ch_enc[:1] 
                                        + self.cv_encoder.num_ch_enc)

        # 深度解码器同时输出深度与用于高斯生成的像素特征
        self.depth_decoder = DepthDecoder(dec_num_input_ch, 
                                            num_output_channels=1+64,
                                            near=depth_range[0],
                                            far=depth_range[1],
                                            num_samples=self.cfg.num_depth_candidates,
                                            log_planes=self.cfg.log_planes,)
        # 多尺度深度层级数量（s0~s3）
        self.max_depth = 4
        # 张量格式化工具（当前 forward 中未直接调用）
        self.tensor_formatter = TensorFormatter()

        # 融合时的权重嵌入（把 2 维权重映射到更高语义空间）
        self.weight_embedding = nn.Sequential(nn.Linear(2, 12), 
                                    activation_func,
                                    nn.Linear(12, 12),)
        # 用于多视角高斯特征融合的循环单元
        self.gru = GRU()

    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # 根据训练步数在 [initial, final] 间平滑过渡指数，控制透明度映射曲线陡峭程度
        cfg = self.cfg.opacity_mapping
        # 线性 warm-up：从 initial 平滑过渡到 final
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        # 在 2 的指数空间里控制曲线形状
        exponent = 2**x

        # 将概率密度映射为不透明度，兼顾低密度与高密度区域梯度
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))
    
    def forward(
        self,
        context,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        is_testing: bool = False,
        export_ply: bool = False,
        dataset_name: str = 'scannet',
    ) -> dict:
        """前向：多视角输入 -> 深度/高斯表示。

        关键输入字段（context）：
        - image: [B, V, 3, H, W]，归一化图像
        - intrinsics: [B, V, 3, 3]，归一化内参（相对宽高）
        - extrinsics: [B, V, 4, 4]，相机外参
        - near/far: 深度范围
        """
        # 从输入张量读取设备信息
        device = context["image"].device
        # batch_size、输入视角数、图像高宽
        b, n_views, _, h, w = context["image"].shape
        # 结果字典：收集中间可视化与训练监督项
        results = {}
        # 控制每个当前视角最多选多少邻居视角参与匹配
        num_context_views = self.cfg.num_views

        gaussians = []
        coords = []
        results = {}

        # 仅对 backbone 中 BN 层设置训练态（其余层按模型当前模式）
        self.backbone.apply(set_bn_eval)
        # 记录原始图像尺寸，后续融合时会把归一化内参还原到像素坐标系
        context['image_shape'] = (h, w)
        # 拷贝一份内参并缩放到 1/4 特征分辨率，用于 cost volume 几何投影
        context_intrinsics = context['intrinsics'].clone()
        context_intrinsics[:,:,0] *= (w // 4)
        context_intrinsics[:,:,1] *= (h // 4)

        # 当前视角索引 0 ~ n_veiws-1
        cur_indices = torch.arange(n_views, device=context['image'].device)
            
        # 根据 cur_indices 收集当前视角的内参、外参、图像
        # 内参：[batch_size, n_veiws, 3, 3]
        # 外参[batch_size, n_veiws, 4, 4]
        # 图像[batch_size*n_veiws,3,h,w]
        cur_intrinsics = context_intrinsics.gather(dim=1, index=cur_indices.view(1,-1,1,1).repeat(b,1,3,3))
        cur_extrinsics = context['extrinsics'].gather(dim=1, index=cur_indices.view(1,-1,1,1).repeat(b,1,4,4))
        cur_image = context['image'].gather(dim=1, index=cur_indices.view(1,-1,1,1,1).repeat(b,1,3,h,w)).view(-1,3,h,w)

        # 1、获取2D特征，逐 batch 送入 backbone
        cur_feats = []
        for bb in range(b):
            # 提取当前 batch 中全部视角的多尺度特征
            cur_feats.append(self.backbone(cur_image[bb*n_views:(bb+1)*n_views]))
        
        # 将各 batch 的同层特征拼接，得到每层 [B*V, C, H, W]
        cur_feats = [torch.cat([x[l] for x in cur_feats], dim=0) for l in range(len(cur_feats[0]))]

        # full_indices[i,j] = j：用于后续构造每个目标视角对应的源视角集合，视角索引矩阵[v,v]
        full_indices = torch.arange(n_views, device=context['image'].device)[None].repeat(n_views,1)
        
        # 若输入视角数 <= 设定上下文数，则全连接互看；否则按几何距离选局部邻居
        use_local = (n_views > num_context_views)
        if not use_local:
            # 去掉自身后的所有视角都作为 source veiws，shape: [B, V, V-1]
            src_indices = full_indices[~(full_indices == cur_indices[:,None])].view(1,n_views,n_views-1).repeat(b,1,1)
        else:
            # slide_mask[i,j]=True 表示视角 i 选择视角 j 作为 source
            slide_mask = torch.zeros((n_views, n_views), dtype=torch.bool, device=full_indices.device)
            dist_matrix = calculate_distance_matrix(context["extrinsics"])

            # 每个视角保留几何距离最近的 num_context_views 个邻居（含自己，后面再去掉）
            _, indices = torch.topk(dist_matrix, min(num_context_views, n_views), largest=False, sorted=False, dim=1)
            # 把近邻索引写入布尔掩码矩阵
            slide_mask.scatter_(1, indices, True)
            # 显式去掉自己到自己的连接
            slide_mask[torch.arange(n_views), torch.arange(n_views)] = False

            # 最终 source 索引，shape: [B, V, min(V, num_context_views)-1]
            src_indices = full_indices[(~(full_indices == cur_indices[:,None]))*slide_mask].view(1,n_views,min(n_views, num_context_views)-1).repeat(b,1,1)

        
        # 按 source 索引重排每个 target 视角对应的 source 外参/内参/图像
        src_extrinsics = context['extrinsics'][:,None].repeat(1,n_views,1,1,1).gather(dim=2, index=src_indices[...,None,None].repeat(1,1,1,4,4))
        src_intrinsics = context_intrinsics[:,None].repeat(1,n_views,1,1,1).gather(dim=2, index=src_indices[...,None,None].repeat(1,1,1,3,3))
        src_image = context['image'][:,None].repeat(1,n_views,1,1,1,1).gather(dim=2, index=src_indices[...,None,None,None].repeat(1,1,1,3,h,w))\
                                .view(-1,n_views-1 if not use_local else min(n_views, num_context_views)-1,3,h,w)
        # 坐标变换：source<->current 相机坐标系之间的相对位姿
        src_cam_t_world = src_extrinsics.inverse()
        cur_cam_t_world = cur_extrinsics.inverse()
        # 当前相机坐标 -> 源相机坐标
        src_cam_T_cur_cam = src_cam_t_world @ cur_extrinsics.unsqueeze(2)
        # 源相机坐标 -> 当前相机坐标
        cur_cam_T_src_cam = cur_cam_t_world.unsqueeze(2) @ src_extrinsics

        # 匹配特征使用 backbone 的 1/4 分辨率层（cur_feats[1]）,[B*V,C,H/4,W/4]
        matching_cur_feats = cur_feats[1]
        dim = matching_cur_feats.shape[-3] # C
        # 为每个 target 视角收集对应的 source 特征,[B*V,V-1,C,H/4,W/4]
        matching_src_feats = rearrange(cur_feats[1], "(b v) c h w -> b v c h w", b=b, v=n_views)[:,None].repeat(1,n_views,1,1,1,1).\
                            gather(dim=2, index=src_indices[...,None,None,None].repeat(1,1,1,dim,h//4,w//4))\
                            .view(-1,n_views-1 if not use_local else min(n_views, num_context_views)-1,dim,h//4,w//4)

        # 把 [B,V,...] 整理成 [B*V,...]，对齐 cost volume 接口输入格式
        src_cam_T_cur_cam_ = rearrange(src_cam_T_cur_cam, 'b v n x y -> (b v) n x y')
        cur_cam_T_src_cam_ = rearrange(cur_cam_T_src_cam, 'b v n x y -> (b v) n x y')
        src_intrinsics_ = rearrange(src_intrinsics, 'b v n x y -> (b v) n x y')
        cur_intrinsics_ = rearrange(cur_intrinsics, 'b v x y -> (b v) x y')
        # 组装 4x4 K 矩阵（只填左上 3x3），便于齐次坐标投影
        src_K = torch.eye(4, device=context['image'].device)[None,None].repeat(src_intrinsics_.shape[0], src_intrinsics_.shape[1],1,1)
        # 写入每个 source 的 3x3 内参
        src_K[:,:,:3,:3] = src_intrinsics_
        cur_inverse = torch.eye(4, device=context['image'].device)[None].repeat(cur_intrinsics_.shape[0],1,1)
        # 写入当前视角 K^{-1}
        cur_inverse[:,:3,:3] = cur_intrinsics_.inverse()
        
        # 使用 batch 中第一个样本的 near/far 作为当前匹配深度范围
        near = context["near"][:1,0].type_as(src_K).view(1, 1, 1, 1)
        far = context["far"][:1,0].type_as(src_K).view(1, 1, 1, 1)
        
        # 2、构建 cost volume：在每个深度 bin 下做跨视角重投影匹配并聚合
        cost_volume = self.cost_volume(cur_feats=matching_cur_feats,
                                        src_feats=matching_src_feats,
                                        src_extrinsics=src_cam_T_cur_cam_,
                                        src_poses=cur_cam_T_src_cam_,
                                        src_Ks=src_K,
                                        cur_invK=cur_inverse,
                                        min_depth=near,
                                        max_depth=far,
                                    )
        
        # 进一步编码 cost volume，并与主干特征融合后送入深度解码器
        # list [B*V,64,H/4,W/4] [B*V,128,H/8,W/8] [B*V,256,H/16,W/16] [B*V,384,H/32,W/32]
        cost_volume_features = self.cv_encoder(
                                cost_volume, 
                                cur_feats[1:],
                            )
        # 用 CV 编码结果替换原 backbone 对应层作为解码输入
        # list [B*V,24,H/2,W/2] [B*V,64,H/4,W/4] [B*V,128,H/8,W/8] [B*V,256,H/16,W/16] [B*V,384,H/32,W/32]
        cur_feats = cur_feats[:1] + cost_volume_features
    
        # 3、获得深度，depth_outputs是一个dict，同时包含多尺度深度、log-depth、深度权重、像素特征等
        depth_outputs = self.depth_decoder(cur_feats)
        
        # 从 RGB 构造高分辨率 skip 特征，补充纹理细节到高斯特征中
        to_skip = context['image']
        # 把 [B,V,C,H,W] 合并成 [B*V,C,H,W] 便于 2D 卷积处理
        to_skip = rearrange(to_skip, "b v c h w -> (b v) c h w")

        # 取第一层跳连分支（不降采样）增强局部细节
        skip = self.high_resolution_skip[0](to_skip)
        # 每个像素的归一化光线方向/平面坐标
        xy_ray, _ = sample_image_grid((h, w), device)
        # 取 depth decoder 的 64 维特征（除第一个通道）并还原为 [B,V,H,W,C]
        gaussians_feats = rearrange(depth_outputs[f'output_pred_s-1_b1hw'][:,1:], '(b v) c h w -> b v h w c', b=b, v=n_views)
        # 融合 skip 特征提升局部外观信息，把高分辨率外观信息加到每像素的特征里
        gaussians_feats = gaussians_feats + rearrange(skip, "(b v) c h w -> b v h w c", b=b, v=n_views)
        # 第一个通道经过Sigmoid映射到 [0,1]作为密度/置信度density
        densities = nn.Sigmoid()(rearrange(depth_outputs[f'output_pred_s-1_b1hw'][:,:1], '(b v) c h w -> b v (c h w) () ()', b=b, v=n_views))
        # 深度与深度权重展平到像素维，便于后续融合[B,V,H*W,1,1]
        depths = rearrange(depth_outputs[f'depth_pred_s-1_b1hw'], "(b v) c h w -> b v (c h w) () ()", b=b)
        weights = rearrange(depth_outputs[f'depth_weights'], "(b v) c h w -> b v (c h w) () ()", b=b)
        
        # 特征展平为每像素一个 token
        gaussians_feats = rearrange(gaussians_feats, "b v h w c -> b v (h w) c") # [B,V,H*W,C]
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy") # [H*W,1,2]
        # 预留每个 surface 的 xy 偏移（当前初始化为 0）
        offset_xy = torch.zeros_like(rearrange(gaussians_feats[..., :2], "... (srf c) -> ... srf c", srf=self.cfg.num_surfaces),
                                        device=gaussians_feats.device) # 0,[B,V,H*W,1,2]
        xy_ray = xy_ray + offset_xy # [B,V,H*W,1,2]

        # 4、获取高斯，单视角高斯解码：像素射线 + 深度 + 密度 + 特征 -> 3D 高斯坐标/属性
        # 
        coords.append(self.gaussian_adapter.forward(
                    rearrange(context["extrinsics"], "b v i j -> b v () () () i j"),
                    rearrange(context["intrinsics"], "b v i j -> b v () () () i j"),
                    rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                    depths,
                    densities,
                    gaussians_feats,
                    (h, w),
                    fusion=True,
                ))
        # 暂存每像素高斯特征，后续做跨视角融合
        gaussians.append(gaussians_feats)

        # 记录 finest scale 深度预测（供损失与可视化）
        results[f'depth_num0_s-1'] = depths
        try:
            # 若 GT 深度存在，保存原始深度与有效掩码
            depths_raw = rearrange(context[f'depth_s-1'], "b v c h w -> b v (h w) c 1")
            results[f'depth_num0_s-1_raw'] = depths_raw
            mask = (depths_raw > 1e-3) * (depths_raw < 10)
            results[f'depth_num0_s-1_mask'] = mask
        except:
            pass
        results[f'depth_num0_s-1_b1hw'] = depth_outputs[f'depth_pred_s-1_b1hw']

        # 保存多尺度深度结果（s0..s3）
        for s in range(self.max_depth):
            # 保存展平后的第 s 层深度
            results[f'depth_num0_s{s}'] = rearrange(depth_outputs[f'depth_pred_s{s}_b1hw'], "(b v) c h w -> b v (c h w) () ()", b=b)

            # 读取 log-depth（当前仅保留变量，便于后续扩展）
            log_depths = depth_outputs[f'log_depth_pred_s{s}_b1hw']
            # 读取原始 depth map（B*V,1,H,W）
            depths = depth_outputs[f'depth_pred_s{s}_b1hw']
            try:
                # 若有监督深度，则拉平成同尺度 map
                depths_raw = rearrange(context[f'depth_s{s}'], "b v c h w -> (b v) c h w")
                results[f'depth_num0_s{s}_raw_b1hw'] = depths_raw
                # 有效深度区间掩码（过滤 0 与远距噪声）
                mask = (depths_raw > 1e-3) * (depths_raw < 10)
                results[f'depth_num0_s{s}_mask_b1hw'] = mask
            except:
                pass
            
            # 保存该尺度可视化形式的深度
            results[f'depth_num0_s{s}_b1hw'] = depths
                        
        # 5、高斯融合，对 batch 中每个样本独立做多视角高斯融合
        our_gaussians = []
        num_raw_gaussians = gaussians[0].shape[2] * gaussians[0].shape[1]
        B = gaussians[0].shape[0]
        for b in range(B):
            # 取单个样本的高斯特征序列
            cur_gs = [x[b:b+1] for x in gaussians]
            # 取单个样本的高斯坐标序列
            cur_coords = [x[b:b+1] for x in coords]
            # 取单个样本的密度与深度权重
            cur_densities = densities[b:b+1]
            cur_weights = weights[b:b+1]
            # 取单个样本最细尺度深度图
            cur_depth = rearrange(depth_outputs[f'depth_pred_s-1_b1hw'], "(b v) c h w -> b v c h w", b=B)[b]
            
            # 按几何一致性融合：返回融合后特征/坐标/外参/深度
            cur_gaussians, cur_coords, cur_extrinsics, cur_depths = self.fuse_gaussians(cur_gs, cur_coords, 
                                            cur_densities, cur_weights, 
                                            cur_depth, 
                                            context["extrinsics"][b:b+1], \
                                            context["intrinsics"][b:b+1], context['image_shape'])

            # 将融合特征映射成高斯参数，并再次走 gaussian_adapter 得到标准 Gaussians 结构
            cur_gaussians_now = rearrange(
                            self.to_gaussians(cur_gaussians),
                            "... (srf c) -> ... srf c",
                            srf=self.cfg.num_surfaces,
                        )
            cur_gaussians = self.gaussian_adapter.forward(
                # 每个融合点都有对应的融合外参
                rearrange(cur_extrinsics, "b r i j -> b () r () () i j"),
                # 内参复用第 0 视角，扩展到所有融合点
                repeat(context["intrinsics"][b:b+1,0], "b i j -> b () N () () i j", N=cur_gaussians_now.shape[1]),
                # 使用像素光线网格恢复高斯中心方向
                rearrange(xy_ray[b:b+1], "b v r srf xy -> b v r srf () xy"),
                # 融合后深度作为射线采样距离
                rearrange(cur_depths, "b r -> b () r () ()"),
                # 第 0~1 通道用于不透明度/密度参数
                nn.Sigmoid()(rearrange(cur_gaussians_now[..., :1], "b r srf c -> b () r srf c")),
                # 剩余通道作为几何与外观参数
                rearrange(cur_gaussians_now[..., 2:], "b r srf c -> b () r srf () c"),
                (h, w),
                fusion=False,
                # 传入融合后坐标，避免重复反投影
                coords=rearrange(cur_coords, "b r c -> b () r () () c"),
            )
            # 收集该样本最终高斯对象
            our_gaussians.append(cur_gaussians)
        # 融合前后高斯数量比例，可用于监控重复点剔除效果
        num_gaussians = our_gaussians[0].means.shape[2]
        results['gs_ratio'] = num_gaussians / num_raw_gaussians
        gaussians = cur_gaussians
 
        results['num_gaussians'] = num_gaussians
        visualization_dump = {}
        try:
            # 展平用于可视化的 scale / rotation
            visualization_dump["scales"] = rearrange(
                gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
            )
            visualization_dump["rotations"] = rearrange(
            gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
            )
        except:
            # 若维度已是扁平形式，则直接输出
            visualization_dump["scales"] = gaussians.scales
            visualization_dump["rotations"] = gaussians.rotations
        
        results['visualizations'] = visualization_dump

        # 每个样本输出一个 Gaussians 对象（展开 v/r/srf/spp 维）
        final_gs = []
        for i in range(len(our_gaussians)):
            # 将每个样本的结构化高斯字段统一展平
            final_gs.append(Gaussians(
                rearrange(
                    our_gaussians[i].means,
                    "b v r srf spp xyz -> b (v r srf spp) xyz",
                ),
                rearrange(
                    our_gaussians[i].covariances,
                    "b v r srf spp i j -> b (v r srf spp) i j",
                ),
                rearrange(
                    our_gaussians[i].harmonics,
                    "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
                ),
                rearrange(
                    our_gaussians[i].opacities,
                    "b v r srf spp -> b (v r srf spp)",
                ),
            ))
        results['gaussians'] = final_gs

        return results

    def fuse_gaussians(self, gaussians, coords, densities, weight_emb, depths, 
                       extrinsics, intrinsics, image_shape, depth_thres=0.1):
        """融合多视角高斯点，减少重复并整合属性。

        主要思想：
        1) 把“全局已融合高斯”重投影到当前视角；
        2) 通过深度一致性筛选可对应点；
        3) 用 GRU 融合特征，并按密度加权融合坐标/外参/深度；
        4) 未匹配到的当前视角点直接追加到全局集合。

        Args:
            depth_thres: 最小深度容差（米），实际阈值为 max(0.05*depth, depth_thres)。
        """
        # 视角长度 V（此处 gaussians[0] 形状近似 [1, V, HW, C]）
        length = gaussians[0].shape[1]
        # 以第 0 个视角初始化全局高斯池
        global_gaussians = gaussians[0][:,0]
        global_densities = densities[:, 0]
        global_weight_emb = weight_emb[:, 0]
        global_coords = coords[0][:,0,:,0,0]
        # 每个高斯关联一个“来源外参”，后续融合时也做加权更新
        global_extrinsics = extrinsics[:,0][:,None].repeat(1,global_gaussians.shape[1],1,1)
        depths = rearrange(depths, "v c h w -> v (c h w)")
        # 初始化全局深度缓存
        global_depths = depths[None, 0]

        # 读取像素级高宽，用于投影落点合法性判断
        h, w = image_shape
        for i in range(1, length):
            # 当前待融合视角的相机参数
            extrinsic = extrinsics[0,i]
            intrinsic = intrinsics[0,i].clone()
            # 归一化内参恢复到像素尺度
            intrinsic[:1,:] *= w
            intrinsic[1:2,:] *= h
            focal_length = (intrinsic[0, 0], intrinsic[1, 1])
            principal_point = (intrinsic[0, 2], intrinsic[1, 2])
            principal_point_mat = torch.tensor([principal_point[0], principal_point[1]]).to(intrinsic.device)
            principal_point_mat = principal_point_mat.reshape(1, 2)
            focal_length_mat = torch.tensor([focal_length[0], focal_length[1]]).to(intrinsic.device)
            focal_length_mat = focal_length_mat.reshape(1, 2)
            # 全局 3D 点 -> 当前相机坐标系 -> 像素坐标
            means1 = torch.cat([global_coords[0], torch.ones_like(global_coords[..., :1][0])], dim=-1).permute(1,0) # [4, 196608]
            post_xy_coords = torch.matmul(extrinsic.inverse(), means1)[:3]
            curr_depths = post_xy_coords[2:3, :]
            post_xy_coords = (post_xy_coords / curr_depths)[:2].permute(1,0)
            curr_depths = curr_depths.squeeze()
            post_xy_coords = post_xy_coords * focal_length_mat.reshape(1,2) + principal_point_mat # [196608, 2]
            pixel_coords = post_xy_coords.round().long()[:,[1,0]]
            # 仅保留落在图像内且深度为正的投影点
            valid = (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < h) & (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < w) & (curr_depths > 0)
            proj_map = - torch.ones((h*w), device=coords[0].device, dtype=curr_depths.dtype)
            # 初始化深度图为大值，后续通过 scatter_reduce 取最小深度
            depth_map = torch.ones((h*w), device=coords[0].device, dtype=curr_depths.dtype) * 10000

            # 对同一像素多个投影，保留最近深度（z-buffer）
            pixel_indices = (pixel_coords[:, 1] + pixel_coords[:, 0]*w)[valid]
            depth_map.scatter_reduce_(0, pixel_indices, curr_depths[valid], reduce='amin')

            # 深度一致性掩码：|z_proj - z_cur| < max(0.05*z_cur, depth_thres)
            fusion_mask = torch.abs(depth_map - depths[i]) < torch.clamp_min(depths[i] * 0.05, depth_thres)

            # proj_map: 仅记录“最近点”对应索引；fusion_indices: 满足深度一致性索引
            proj_map = torch.where(depth_map[pixel_indices] == curr_depths[valid])[0]
            fusion_indices = torch.where(fusion_mask[pixel_indices])[0]
            # 二者取交集，得到真正可融合的一一对应
            fusion_indices_ = fusion_indices[torch.isin(fusion_indices, proj_map)]
            corr_indices = proj_map[torch.isin(proj_map, fusion_indices)]
            valid_indices = torch.zeros(valid.sum(), device=valid.device, dtype=torch.bool)
            valid_indices.scatter_(0, corr_indices, True)
            mask = torch.zeros_like(valid, device=valid.device, dtype=torch.bool)
            mask[valid] = valid_indices

            # 保留原实现中的重复掩码构建逻辑（不改动计算路径）
            valid_indices = torch.zeros(valid.sum(), device=valid.device, dtype=torch.bool)
            valid_indices.scatter_(0, corr_indices, True)
            mask = torch.zeros_like(valid, device=valid.device, dtype=torch.bool)
            mask[valid] = valid_indices

            if mask.sum() > 0:
                # 用两路密度/权重构造 PE 嵌入，作为 GRU 门控的条件
                input_weights_emb = positional_encoding(torch.cat([global_densities[:, mask], weight_emb[:, i, pixel_indices][:,fusion_indices_]], dim=-1), 6)
                hidden_weights_emb = positional_encoding(torch.cat([densities[:, i, pixel_indices][:,fusion_indices_], global_weight_emb[:, mask]], dim=-1), 6)
                # GRU 融合“当前视角特征”和“全局特征”
                fusion_feat = self.gru(gaussians[0][:, i, pixel_indices][:,fusion_indices_].unsqueeze(2),
                                       global_gaussians[:, mask].unsqueeze(2),
                                       input_weights_emb,
                                       hidden_weights_emb).squeeze(2)

                # 保留未参与融合的旧点 + 融合后的新点
                global_gaussians = torch.cat([global_gaussians[:, ~mask], fusion_feat], dim=1)
                # 使用密度作为融合权重（重复到兼容目标张量形状）
                weights_0 = global_densities[:, mask].repeat(1, 1, 1, 2)
                weights_1 = densities[:, i, pixel_indices][:,fusion_indices_].repeat(1, 1, 1, 2)

                # 位置、密度、权重嵌入、外参、深度做加权融合
                global_coords = torch.cat([global_coords[:, ~mask], (global_coords[:, mask]*weights_0[...,1] +
                                                coords[0][:, i, pixel_indices][:,fusion_indices_,0,0]*weights_1[...,1]) / (weights_0[...,1]+weights_1[...,1])], dim=1)
                global_densities = torch.cat([global_densities[:, ~mask], (global_densities[:, mask] +
                                                        densities[:, i, pixel_indices][:,fusion_indices_])], dim=1)
                global_weight_emb = torch.cat([global_weight_emb[:, ~mask], (global_weight_emb[:, mask] +
                                                        weight_emb[:, i, pixel_indices][:,fusion_indices_])], dim=1)
                
                global_extrinsics = torch.cat([global_extrinsics[:, ~mask], (global_extrinsics[:, mask]*weights_0[...,:1] +
                                                extrinsics[:, i, None]*weights_1[...,:1]) / (weights_0[...,:1]+weights_1[...,:1])], dim=1)
                global_depths = torch.cat([global_depths[:, ~mask], (global_depths[:, mask]*weights_0[...,0,0] +
                                                        depths[None, i, pixel_indices][:,fusion_indices_]*weights_1[...,0,0])/(weights_0[...,0,0]+weights_1[...,0,0])], dim=1)
            
            # 把当前视角中未通过融合匹配的点直接追加到全局池
            global_gaussians = torch.cat([global_gaussians, gaussians[0][:,i]\
                                                [:,~fusion_mask]], dim=1)
            global_coords = torch.cat([global_coords, coords[0][:,i]\
                                                [:,~fusion_mask,0,0]], dim=1)
            
            global_densities = torch.cat([global_densities, densities[:,i]\
                                                [:,~fusion_mask]], dim=1)
            global_weight_emb = torch.cat([global_weight_emb, weight_emb[:,i]\
                                                [:,~fusion_mask]], dim=1)
            global_extrinsics = torch.cat([global_extrinsics, extrinsics[:,i,None].repeat(1,(~fusion_mask).sum(),1,1)], dim=1)
            global_depths = torch.cat([global_depths, depths[None,i]\
                                                [:,~fusion_mask]], dim=1)

        # 返回融合后的特征、坐标、外参、深度
        return global_gaussians, global_coords, global_extrinsics, global_depths
