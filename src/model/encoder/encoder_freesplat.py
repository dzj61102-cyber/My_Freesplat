# 标准库：数据类与类型标注
# vscode_save_probe_20260325
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


@dataclass
class BudgetPruningCfg:
    enabled: bool = True
    keep_ratio: float = 0.8
    lambda_budget: float = 0.2
    lambda_bin: float = 0.02
    tau_start: float = 0.5
    tau_end: float = 0.2
    total_steps: int = 300000


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
    budget_pruning: BudgetPruningCfg
    
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
                                                    num_depth_bins=self.cfg.num_depth_candidates,#深度离散采样的数量
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

        # 预算感知剪枝的轻量打分头：66 -> 128 -> 64 -> 1
        self.importance_head = nn.Sequential(
            nn.Linear(66, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    # 温度退火，温度呈指数规律从 tau_start(0.5) 衰减到 tau_end(0.2)
    def _pruning_temperature(self, global_step: int) -> float:
        cfg = self.cfg.budget_pruning
        if cfg.total_steps <= 0:
            return cfg.tau_end
        progress = min(max(global_step, 0) / cfg.total_steps, 1.0)
        tau_start = max(cfg.tau_start, 1e-6)
        tau_end = max(cfg.tau_end, 1e-6)
        return tau_start * ((tau_end / tau_start) ** progress)

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
    ) -> dict:#返回值类型注解，返回一个字典
        """前向：多视角输入 -> 深度/高斯表示。

        关键输入字段（context，输入数据的dict）：
        - image: [B, V, 3, H, W]，归一化图像
        - intrinsics: [B, V, 3, 3]，归一化相机内参，每个 batch、每个视角都有一个 3×3 内参矩阵
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

        # 对 backbone 里的每一层都执行一次 set_bn_eval(module)
        # 仅对 backbone 中 BN 层设置训练态（其余层按模型当前模式）
        self.backbone.apply(set_bn_eval)
        # 记录原始图像尺寸，后续融合时会把归一化内参还原到像素坐标系
        context['image_shape'] = (h, w)
        # 拷贝归一化内参
        context_intrinsics = context['intrinsics'].clone()
        # 通常相机内参矩阵长这样：
        # [[fx,  0, cx],
        # [ 0, fy, cy],
        # [ 0,  0,  1]]
        # 把 归一化内参 转换成 1/4分辨率特征图上的像素内参
        context_intrinsics[:,:,0] *= (w // 4)
        context_intrinsics[:,:,1] *= (h // 4)

        # 一维张量，当前视角索引 [0,1,...,n_veiws-1]，并放到 image 同样的设备上
        cur_indices = torch.arange(n_views, device=context['image'].device)
            
        # 根据 cur_indices 收集当前视角的内参、外参、图像
        # 内参cur_intrinsics [b, v, 3, 3]
        # 外参cur_extrinsics [b, v, 4, 4]
        # 图像cur_image [b*v,3,h,w],把 batch 维和视角维合并，便于统一送进 2D CNN

        # 在视角维上，根据 index 提供的索引取值，index的shape需要和输出一致
        # cur_indices的shape是[v]，通过view调整为[1,v,1,1]，再在batch维上重复b次，在3x3和4x4维上重复以匹配内外参的形状
        cur_intrinsics = context_intrinsics.gather(dim=1, index=cur_indices.view(1,-1,1,1).repeat(b,1,3,3))
        cur_extrinsics = context['extrinsics'].gather(dim=1, index=cur_indices.view(1,-1,1,1).repeat(b,1,4,4))
        cur_image = context['image'].gather(dim=1, index=cur_indices.view(1,-1,1,1,1).repeat(b,1,3,h,w)).view(-1,3,h,w)

        # 1、获取2D特征，
        cur_feats = []
        # self.backbone 输出的是一个 5 层特征 list（1/2, 1/4, 1/8, 1/16, 1/32 分辨率）
        for bb in range(b):
            cur_feats.append(self.backbone(cur_image[bb*n_views:(bb+1)*n_views]))# 逐batch送入backbone提取特征，一个batch=一个多视角采样
        
        # 将各 batch 的同层特征拼接，得到cur_feats，list，长度5
        # cur_feats每个元素是一个张量，形状如下  C：24-48-64-160-256
        #   - cur_feats[0]: [B*V, C0, H/2,  W/2 ]
        #   - cur_feats[1]: [B*V, C1, H/4,  W/4 ]（专门用于匹配）
        #   - cur_feats[2]: [B*V, C2, H/8,  W/8 ]
        #   - cur_feats[3]: [B*V, C3, H/16, W/16]
        #   - cur_feats[4]: [B*V, C4, H/32, W/32]
        cur_feats = [torch.cat([x[l] for x in cur_feats], dim=0) for l in range(len(cur_feats[0]))]

        # 视角索引矩阵，形状[v,v]
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

            # shape: [B, V, num_context_views-1]
            src_indices = full_indices[(~(full_indices == cur_indices[:,None]))*slide_mask].view(1,n_views,min(n_views, num_context_views)-1).repeat(b,1,1)

        # 最终 source 索引的shape: [B, V, K],K=min(V, num_context_views)-1
        # 对每个 target 视角 i，按 src_indices[b,i,:] 取出它的K个邻居视角信息：外参/内参/图像
        # 外参[B, V, K, 4, 4]
        src_extrinsics = context['extrinsics'][:,None].repeat(1,n_views,1,1,1).gather(dim=2, index=src_indices[...,None,None].repeat(1,1,1,4,4))
        # 内参[B, V, K, 3, 3]
        src_intrinsics = context_intrinsics[:,None].repeat(1,n_views,1,1,1).gather(dim=2, index=src_indices[...,None,None].repeat(1,1,1,3,3))
        # 图像[B*V, K, 3, H, W]
        src_image = context['image'][:,None].repeat(1,n_views,1,1,1,1).gather(dim=2, index=src_indices[...,None,None,None].repeat(1,1,1,3,h,w))\
                                .view(-1,n_views-1 if not use_local else min(n_views, num_context_views)-1,3,h,w)

        # 坐标变换：source<->current 相机坐标系之间的相对位姿
        src_cam_t_world = src_extrinsics.inverse()#[B, V, K, 4, 4]（world->src）
        cur_cam_t_world = cur_extrinsics.inverse()#[B, V, 4, 4]（world->cur）
        src_cam_T_cur_cam = src_cam_t_world @ cur_extrinsics.unsqueeze(2)#[B, V, K, 4, 4]（cur->src）
        cur_cam_T_src_cam = cur_cam_t_world.unsqueeze(2) @ src_extrinsics#[B, V, K, 4, 4]（src->cur）

        # 匹配特征使用 backbone 的 1/4 分辨率层（cur_feats[1]）,[B*V,C1,H/4,W/4]
        matching_cur_feats = cur_feats[1]
        dim = matching_cur_feats.shape[-3] # C1=48
        # 为每个 target 视角收集对应的 source 特征,[B*V,K,C1,H/4,W/4]
        matching_src_feats = rearrange(cur_feats[1], "(b v) c h w -> b v c h w", b=b, v=n_views)[:,None].repeat(1,n_views,1,1,1,1).\
                            gather(dim=2, index=src_indices[...,None,None,None].repeat(1,1,1,dim,h//4,w//4))\
                            .view(-1,n_views-1 if not use_local else min(n_views, num_context_views)-1,dim,h//4,w//4)

        # 把 [B,V,...] 整理成 [B*V,...]，对齐 cost volume 接口输入格式
        src_cam_T_cur_cam_ = rearrange(src_cam_T_cur_cam, 'b v n x y -> (b v) n x y')
        cur_cam_T_src_cam_ = rearrange(cur_cam_T_src_cam, 'b v n x y -> (b v) n x y')
        src_intrinsics_ = rearrange(src_intrinsics, 'b v n x y -> (b v) n x y')
        cur_intrinsics_ = rearrange(cur_intrinsics, 'b v x y -> (b v) x y')

        # 把 3x3 内参扩成 4x4 齐次形式，便于齐次坐标投影
        # src视角内参[B*V, K, 4, 4]
        src_K = torch.eye(4, device=context['image'].device)[None,None].repeat(src_intrinsics_.shape[0], src_intrinsics_.shape[1],1,1)
        # 写入每个 source 的 3x3 内参
        src_K[:,:,:3,:3] = src_intrinsics_
        # cur视角内参[B*V, 4, 4]
        cur_inverse = torch.eye(4, device=context['image'].device)[None].repeat(cur_intrinsics_.shape[0],1,1)
        # 把 K^{-1} 写到 4x4 左上角，得到 current 相机的齐次逆内参
        # K 把相机坐标系 3D 点投到像素平面，K^{-1} 反过来把像素坐标 (u,v,1) 变成相机坐标系下的“射线方向
        cur_inverse[:,:3,:3] = cur_intrinsics_.inverse()
        
        # 使用 batch 中第一个样本的第一个视角 near/far 作为当前匹配深度范围，[1, 1, 1, 1]
        near = context["near"][:1,0].type_as(src_K).view(1, 1, 1, 1)
        far = context["far"][:1,0].type_as(src_K).view(1, 1, 1, 1)
        
        # 2、构建 cost volume：在每个深度 bin 下做跨视角重投影匹配并聚合
        # cost_volume: [B*V, D, H/4, W/4]，D=128 候选深度数
        cost_volume = self.cost_volume(cur_feats=matching_cur_feats,#[B*V, C1, H/4, W/4]，当前视角特征
                                        src_feats=matching_src_feats,#[B*V, K, C1, H/4, W/4]，每个cur对应的 src 特征
                                        src_extrinsics=src_cam_T_cur_cam_,#当前相机坐标 -> source 相机坐标
                                        src_poses=cur_cam_T_src_cam_,#source -> 当前相机坐标
                                        src_Ks=src_K,# src视角内参
                                        cur_invK=cur_inverse,#current 相机的齐次逆内参
                                        min_depth=near,
                                        max_depth=far,
                                    )
        """
        cost_volume构建细节：
        当前像素 (u,v) 在每个候选深度 i 下，投到每个 source 去采样特征，和当前特征做匹配，跨 source 聚合后写回该像素的第 i 个 depth 通道
        1. 构建深度平面depth_planes
            shape 为 [B*V, D, H/4, W/4]，其中 D 是深度假设数量
        2. 当前视角反投影到 3D
            对第 i 个深度平面 depth_plane[:, i] -> [B*V,1,H/4,W/4]，
            用当前视角逆内参 cur_invK + 假设深度，将像素反投影到 3D（齐次坐标）：
            [B*V, 4, (H/4)*(W/4)]（4 是齐次坐标维度）。
        3. 投影到每个 source 视角
            用 src内参 和 cur->src 外参，将这些 3D 点投到每个 source，
            shape [B*V*K, 3, H/4, W/4]（K 是 source 视角数，3表示(u,v,z)信息）。
            取 z>0 作为有效性 mask：点在 source 相机前方才有效。
        4. 从 source 特征图采样（warp）
            用投影得到的 (u,v) 通过 grid_sample 在 source 特征图采样，得到对齐到当前视角的warped source特征：
            src_feat_warped，shape [B*V, K, C, H/4, W/4]。
        5. 匹配强度分支（1 通道）
            src_feat_warped 与 cur_feats 做通道点积（并乘有效 mask），得到
            [B*V, K, H/4, W/4]；
            再对 source 维做有效均值，得到
            [B*V, 1, H/4, W/4]，表示该深度假设下的匹配强度。
        6. 外观分支（C 通道）
            对 src_feat_warped 在 source 维做有效均值，得到
            [B*V, C, H/4, W/4]，表示该深度假设下跨视角聚合后的外观信息。
        7. 融合并得到该深度平面的响应
            拼接成 [B*V, C+1, H/4, W/4]，输入 MLP，输出
            [B*V, 1, H/4, W/4]，表示各视角各像素在第 i 个深度假设下的 learned score（可理解为可信度/代价值特征）。
        8. 拼接所有深度平面
            对 i=1..D 的输出在深度维拼接，得到最终 cost volume：
            [B*V, D, H/4, W/4]，表示每个像素都有一条长度为 D 的深度假设打分曲线。
        """

        # 把 cost_volume 和 backbone 的多尺度图像特征融合，编码成新的多尺度深度特征
        # list [B*V,64,H/4,W/4] [B*V,128,H/8,W/8] [B*V,256,H/16,W/16] [B*V,384,H/32,W/32]
        cost_volume_features = self.cv_encoder(
                                cost_volume, #[B*V, D（128）, H/4, W/4]
                                cur_feats[1:],#list [B*V,48,H/4,W/4] [B*V,64,H/8,W/8] [B*V,160,H/16,W/16] [B*V,256,H/32,W/32]
                            )
        # 保留 backbone 最浅层特征（高分辨率纹理层），把其余层替换为 CV 编码后的特征
        # list [B*V,24,H/2,W/2] [B*V,64,H/4,W/4] [B*V,128,H/8,W/8] [B*V,256,H/16,W/16] [B*V,384,H/32,W/32]
        cur_feats = cur_feats[:1] + cost_volume_features
    
        # 3、多尺度深度解码，输出深度和中间特征
        # dict[str, torch.Tensor]，各 key 含义和 shape 如下：
        #   - output_pred_s{i}_b1hw (i=0,1,2,3)
        #       - 含义：第 i 个尺度的解码特征头输出（不是最终深度）
        #       - shape：[B*V, 65, H/2^(i+1), W/2^(i+1)]， 65 = 1 + 64（1 个深度相关通道 + 64 个像素特征通道）
        #   - depth_pred_s{i}_b1hw (i=0,1,2,3)
        #       - 含义：第 i 个尺度的深度图
        #       - shape：[B*V, 1, H/2^(i+1), W/2^(i+1)]
        #   - log_depth_pred_s{i}_b1hw (i=0,1,2,3)
        #       - 含义：第 i 个尺度的对数深度/视差域值（与上面的 depth 一一对应）
        #       - shape：[B*V, 1, H/2^(i+1), W/2^(i+1)]
        #   - depth_pred_s-1_b1hw
        #       - 含义：全分辨率的最终深度图
        #       - shape：[B*V, 1, H, W]
        #   - output_pred_s-1_b1hw
        #       - 含义：全分辨率后的解码特征头输出（第0通道表示密度/不透明度先验/权重,后64通道是高斯特征）
        #       - shape：[B*V, 65, H, W]
        #   - depth_weights
        #       - 含义：全分辨率的深度置信度（取深度候选 softmax 后的最大概率并上采样）
        #       - shape：[B*V, 1, H, W]
        depth_outputs = self.depth_decoder(cur_feats)
        
        # 把 depth_outputs 拆成几何量（depth/weight/xy）+ 密度 + 外观特征，并全部变成逐像素 token 形式

        to_skip = context['image']
        # 把原始 RGB 从 [B,V,C,H,W] 合并成 [B*V,C,H,W] 便于 2D 卷积处理
        to_skip = rearrange(to_skip, "b v c h w -> (b v) c h w")
        # 提取高分辨率纹理特征 [B*V,64,H,W]
        skip = self.high_resolution_skip[0](to_skip)

        # 每个像素的归一化 2D 坐标网格（0~1），shape：[H,W,2]
        xy_ray, _ = sample_image_grid((h, w), device)
        # 高斯外观特征，取 depth decoder 的后 64 通道并还原为 [B,V,H,W,64]
        gaussians_feats = rearrange(depth_outputs[f'output_pred_s-1_b1hw'][:,1:], '(b v) c h w -> b v h w c', b=b, v=n_views)
        # 高斯特征 = 高分辨率纹理特征 + 外观特征，增强细节表达，结果仍是 [B,V,H,W,64]
        gaussians_feats = gaussians_feats + rearrange(skip, "(b v) c h w -> b v h w c", b=b, v=n_views)

        # 密度（不透明度先验），第1个通道经Sigmoid映射到 [0,1]，[B,V,H*W,1,1]
        densities = nn.Sigmoid()(rearrange(depth_outputs[f'output_pred_s-1_b1hw'][:,:1], '(b v) c h w -> b v (c h w) () ()', b=b, v=n_views))
        # 深度与深度置信展平成像素序列，便于后续融合[B,V,H*W,1,1]
        depths = rearrange(depth_outputs[f'depth_pred_s-1_b1hw'], "(b v) c h w -> b v (c h w) () ()", b=b)
        weights = rearrange(depth_outputs[f'depth_weights'], "(b v) c h w -> b v (c h w) () ()", b=b)
        
        # 高斯特征展平，[B,V,H*W,64]，每个像素一个 token，token 维是高斯特征维
        gaussians_feats = rearrange(gaussians_feats, "b v h w c -> b v (h w) c") 

        # 像素网格展平，[H*W,1,2]（后续可广播到 [B,V,H*W,1,2]）
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy") 
        # 预留每个 surface 的 xy 偏移（当前初始化为 0）,[B,V,H*W,num_surfaces,2]
        offset_xy = torch.zeros_like(rearrange(gaussians_feats[..., :2], "... (srf c) -> ... srf c", srf=self.cfg.num_surfaces),
                                        device=gaussians_feats.device) 
        # 最终 xy_ray 形状，[B,V,H*W,num_surfaces,2]，支持“一个像素多个 surface 的射线偏移
        xy_ray = xy_ray + offset_xy 

        # 4、获取高斯，单视角高斯解码：像素射线 + 深度 + 密度 + 特征 -> 3D 高斯坐标/属性

        # 只返回 means（3D坐标）,shape[B, V, H*W, num_surfaces, 3]
        # coords[0] 的语义是：每个 batch、每个视角、每个像素（及每个 surface）的初始 3D 点
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
        # 暂存每像素高斯特征，shape: [B, V,  H*W, 64],（来自深度解码器+skip特征）后续做跨视角融合
        gaussians.append(gaussians_feats)

        # 把全分辨率和多尺度深度图，以及可能存在的真实深度图和有效区域掩码，统一整理到 results 字典里，方便后续算 loss、做可视化、做评估
        # 保存全分辨率深度图，展平格式，[B,V,H*W,1,1]
        results[f'depth_num0_s-1'] = depths 
        try:
            # 若真实深度存在，保存原始深度与有效掩码，展平格式
            depths_raw = rearrange(context[f'depth_s-1'], "b v c h w -> b v (h w) c 1")
            results[f'depth_num0_s-1_raw'] = depths_raw
            mask = (depths_raw > 1e-3) * (depths_raw < 10)
            results[f'depth_num0_s-1_mask'] = mask
        except:
            pass
        # 保存全分辨率深度图，图像格式，[B*V, 1, H, W]
        results[f'depth_num0_s-1_b1hw'] = depth_outputs[f'depth_pred_s-1_b1hw'] 

        # 保存多尺度深度结果（s0..s3）
        for s in range(self.max_depth):
            # 保存多尺度深度，展平格式，[B*V, 1, H/2^(s+1), W/2^(s+1),1,1]
            results[f'depth_num0_s{s}'] = rearrange(depth_outputs[f'depth_pred_s{s}_b1hw'], "(b v) c h w -> b v (c h w) () ()", b=b)
            # 读取 log-depth（当前仅保留变量，便于后续扩展）
            log_depths = depth_outputs[f'log_depth_pred_s{s}_b1hw']
            # 读取原始第 s 层深度，[B*V,1,H/2^(s+1), W/2^(s+1)]
            depths = depth_outputs[f'depth_pred_s{s}_b1hw']
            try:
                # 若多尺度真实深度存在，保存多尺度原始深度与有效掩码，展平格式
                depths_raw = rearrange(context[f'depth_s{s}'], "b v c h w -> (b v) c h w")
                results[f'depth_num0_s{s}_raw_b1hw'] = depths_raw
                mask = (depths_raw > 1e-3) * (depths_raw < 10)
                results[f'depth_num0_s{s}_mask_b1hw'] = mask
            except:
                pass
            # 保存多尺度深度，图像格式，[B*V,1,H/2^(s+1), W/2^(s+1)]
            results[f'depth_num0_s{s}_b1hw'] = depths
                        

        # 5、高斯融合，逐样本
        # 把多视角、多个尺度产生的大量高斯点，按每个 batch 样本分别做“去重+融合”，再把融合后的结果整理成标准的 Gaussians 表示，最后存进 results 返回
        our_gaussians = []#每个 batch 样本融合后的高斯结果
        fused_features = []#每个样本的全局融合特征（与最终高斯一一对应）
        fused_weights = []#每个样本的全局融合权重（与最终高斯一一对应）
        num_raw_gaussians = gaussians[0].shape[2] * gaussians[0].shape[1]#融合前原始高斯数量V*H*W
        B = gaussians[0].shape[0]#B
        for b in range(B):
            # 取当前样本需要融合的数据
            # 高斯特征[1, V,  H*W, 64]
            cur_gs = [x[b:b+1] for x in gaussians] #gaussians[0]的shape:[B, V,  H*W, 64]
            # 3D坐标[1, V, H*W, num_surfaces, 3]
            cur_coords = [x[b:b+1] for x in coords] #coords[0]的shape:[B, V, H*W, num_surfaces, 3]
            # 密度[1,V,H*W,1,1]
            cur_densities = densities[b:b+1] #densities的shape:[B,V,H*W,1,1]
            # 深度权重[1,V,H*W,1,1]
            cur_weights = weights[b:b+1] #weights的shape:[B,V,H*W,1,1]
            # 全分辨率深度图[V,1,H,W]
            cur_depth = rearrange(depth_outputs[f'depth_pred_s-1_b1hw'], "(b v) c h w -> b v c h w", b=B)[b]#[B*V,1,H,W]->[B,V,1,H,W]
            
            # 按几何一致性融合：返回融合后特征/坐标/外参/深度
            (
                cur_gaussians,
                cur_coords,
                cur_extrinsics,
                cur_depths,
                cur_fused_densities,
            ) = self.fuse_gaussians(
                cur_gs,
                cur_coords,
                cur_densities,
                cur_weights,
                cur_depth,
                context["extrinsics"][b:b+1],
                context["intrinsics"][b:b+1],
                context['image_shape'],
            )
            # 记录融合后的全局特征（在映射到高斯参数之前）
            cur_fused_features = cur_gaussians

            # 将融合特征映射成高斯参数
            cur_gaussians_now = rearrange(
                            self.to_gaussians(cur_gaussians),
                            "... (srf c) -> ... srf c",
                            srf=self.cfg.num_surfaces,
                        )

            # 通过高斯参数生成标准 Gaussians 对象            
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
            # 保存当前样本的全局高斯对象，our_gaussians = [样本0的高斯, 样本1的高斯, ...]
            our_gaussians.append(cur_gaussians)
            # 保留融合中间量，供后续按同索引访问每个全局高斯
            fused_features.append(cur_fused_features)
            fused_weights.append(cur_fused_densities)
            
        # 统计融合效果
        # 融合前后高斯数量和压缩率
        num_gaussians = our_gaussians[0].means.shape[2]
        results['gs_ratio'] = num_gaussians / num_raw_gaussians
        gaussians = cur_gaussians #把 gaussians 赋成了最后一个样本的 cur_gaussians
        results['num_gaussians'] = num_gaussians


        # 准备可视化数据
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

        # 把每个样本的高斯对象展平，得到一个样本里所有高斯点的总列表
        final_gs = []
        for i in range(len(our_gaussians)):
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
        fused_weights_flat = [rearrange(x, "b g () () -> b g") for x in fused_weights]
        results['gaussians'] = final_gs
        # 与 results['gaussians'] 同列表顺序、同高斯索引的一一对应附加属性
        results['gaussians_fused_features'] = fused_features
        results['gaussians_fused_weights'] = fused_weights_flat
        # - 对于每个样本 i，高斯索引 g 一一对应：
        # - results['gaussians'][i].means[:, g, ...] / covariances / harmonics / opacities
        # - results['gaussians_fused_features'][i][:, g, ...]
        # - results['gaussians_fused_weights'][i][:, g]
        # 例：对于样本scene0000_01_0：
        # results['gs_ratio']: 0.5598941379123263
        # results['num_gaussians']: 330239（融合后高斯数量）
        # results['gaussians'][0].means shape: torch.Size([1, 330239, 3])
        # results['gaussians'][0].covariances shape: torch.Size([1, 330239, 3, 3])
        # results['gaussians'][0].harmonics shape: torch.Size([1, 330239, 3, 9])
        # results['gaussians'][0].opacities shape: torch.Size([1, 330239])
        # results['gaussians_fused_features'][0] shape: torch.Size([1, 330239, 64])
        # results['gaussians_fused_weights'][0] shape: torch.Size([1, 330239])
   
        if self.cfg.budget_pruning.enabled:
            cfg_prune = self.cfg.budget_pruning
            tau = None if is_testing else self._pruning_temperature(global_step)
            scores_per_sample = []
            gates_per_sample = []
            pruned_gaussians = []
            budget_terms = []
            bin_terms = []
            # gs原始标准高斯对象, feat高斯融合特征, weight高斯融合权重
            for gs, feat, weight in zip(final_gs, fused_features, fused_weights_flat):
                z = torch.cat(
                    [feat, torch.log1p(weight).unsqueeze(-1), gs.opacities.unsqueeze(-1)],
                    dim=-1,
                )
                scores = self.importance_head(z).squeeze(-1)
                scores_per_sample.append(scores)#重要性分数
                if not is_testing:#训练软门控
                    gates = torch.sigmoid(scores / tau)
                    gates_per_sample.append(gates)
                    pruned_gaussians.append(
                        Gaussians(#剪枝后的高斯对象，opacities乘以 gates 软门控
                            gs.means,
                            gs.covariances,
                            gs.harmonics,
                            gs.opacities * gates,
                        )
                    )
                    budget_terms.append((gates.mean() - cfg_prune.keep_ratio) ** 2)#预算损失，鼓励平均保留率接近目标 keep_ratio
                    bin_terms.append((gates * (1 - gates)).mean())#二值化正则项，鼓励 gate 接近 0 或 1
                else:#测试直接硬剪枝,不改变高斯属性
                    pruned_gaussians.append(gs)

            results['gaussians'] = pruned_gaussians#原有高斯对象被覆盖
            results['gaussians_importance_scores'] = scores_per_sample
            results['prune_keep_ratio'] = cfg_prune.keep_ratio
            if not is_testing:
                results['gaussians_soft_gates'] = gates_per_sample
                results['prune_budget_loss'] = cfg_prune.lambda_budget * torch.stack(budget_terms).mean()
                results['prune_bin_loss'] = cfg_prune.lambda_bin * torch.stack(bin_terms).mean()
                # 软门控的平均激活率，可理解为当前实际保留比例（soft）
                results['prune_gate_ratio'] = torch.stack([x.mean() for x in gates_per_sample]).mean()
                results['prune_tau'] = torch.tensor(
                    tau, device=device, dtype=final_gs[0].opacities.dtype
                )

        # 打印
        # print(f"[EncoderFreeSplat] results['gs_ratio']: {results['gs_ratio']}")
        # print(f"[EncoderFreeSplat] results['num_gaussians']: {results['num_gaussians']}")

        # for idx, gs in enumerate(results['gaussians']):
        #     print(f"[EncoderFreeSplat] results['gaussians'][{idx}].means shape: {gs.means.shape}")
        #     print(f"[EncoderFreeSplat] results['gaussians'][{idx}].covariances shape: {gs.covariances.shape}")
        #     print(f"[EncoderFreeSplat] results['gaussians'][{idx}].harmonics shape: {gs.harmonics.shape}")
        #     print(f"[EncoderFreeSplat] results['gaussians'][{idx}].opacities shape: {gs.opacities.shape}")
    
        # for idx, x in enumerate(results['gaussians_fused_features']):
        #     print(f"[EncoderFreeSplat] results['gaussians_fused_features'][{idx}] shape: {x.shape}")
      
        # for idx, x in enumerate(results['gaussians_fused_weights']):
        #     print(f"[EncoderFreeSplat] results['gaussians_fused_weights'][{idx}] shape: {x.shape}")

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
        # 视角长度 V（ gaussians[0] 形状 [1, V, HW, 64]）
        length = gaussians[0].shape[1]
        # 以第 0 个视角初始化全局高斯池
        global_gaussians = gaussians[0][:,0]#[1,HW,64]
        global_densities = densities[:, 0]
        global_weight_emb = weight_emb[:, 0]
        global_coords = coords[0][:,0,:,0,0]
        # 每个高斯关联一个“来源外参”，后续融合时做加权更新，用第 0 个视角的外参初始化
        global_extrinsics = extrinsics[:,0][:,None].repeat(1,global_gaussians.shape[1],1,1)#[1, HW, 4, 4]
        depths = rearrange(depths, "v c h w -> v (c h w)")#[V 1 H W] ->[V HW]
        # 初始化全局深度缓存
        global_depths = depths[None, 0]#[1, HW]

        # 从第 1 个视角开始，逐个和全局池融合
        h, w = image_shape
        for i in range(1, length):
            # extrinsic：描述当前相机在世界中的位置和姿态
            # intrinsic：描述相机的成像参数
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
            #  全局 3D 点投影所得像素坐标[N, 2]，N表示全局高斯点数量，2表示y,x
            pixel_coords = post_xy_coords.round().long()[:,[1,0]]
            # 落在图像内且深度为正，为有效投影点
            valid = (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < h) & (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < w) & (curr_depths > 0)
            proj_map = - torch.ones((h*w), device=coords[0].device, dtype=curr_depths.dtype)
            # 初始化深度图为大值，后续通过 scatter_reduce 取最小深度
            depth_map = torch.ones((h*w), device=coords[0].device, dtype=curr_depths.dtype) * 10000

            # 有效投影点在图像展平后的像素编号，shape：[N_valid]，pixel_indices[k]表示第 k 个合法投影点，落在当前图像展平后的第几个像素位置
            pixel_indices = (pixel_coords[:, 1] + pixel_coords[:, 0]*w)[valid]
            depth_map.scatter_reduce_(0, pixel_indices, curr_depths[valid], reduce='amin')

            # 深度一致性掩码：|全局池投影到当前视角后的深度-当前视角自己的深度| < max(0.05*z_cur, depth_thres)，最小误差阈值不能低于 depth_thres
            fusion_mask = torch.abs(depth_map - depths[i]) < torch.clamp_min(depths[i] * 0.05, depth_thres)

            # 可融合对应点判定，同时满足：是投影到该像素的最近点+和当前视角深度一致
            # 合法投影点中，最近点索引 
            proj_map = torch.where(depth_map[pixel_indices] == curr_depths[valid])[0]
            # 合法投影点中，满足深度一致性的点索引
            fusion_indices = torch.where(fusion_mask[pixel_indices])[0]
            # 合法投影点中，同时满足最近 + 深度一致性的点索引
            fusion_indices_ = fusion_indices[torch.isin(fusion_indices, proj_map)]
            # 全局投影点中，同时满足最近 + 深度一致性的点索引，即最终真正和当前视角建立一一对应关系的点索引
            corr_indices = proj_map[torch.isin(proj_map, fusion_indices)]
            # 全局投影点中，有效投影点的索引
            valid_indices = torch.zeros(valid.sum(), device=valid.device, dtype=torch.bool)
            valid_indices.scatter_(0, corr_indices, True)
            mask = torch.zeros_like(valid, device=valid.device, dtype=torch.bool)
            mask[valid] = valid_indices

            # 保留原实现中的重复掩码构建逻辑（不改动计算路径）
            valid_indices = torch.zeros(valid.sum(), device=valid.device, dtype=torch.bool)
            valid_indices.scatter_(0, corr_indices, True)
            mask = torch.zeros_like(valid, device=valid.device, dtype=torch.bool)
            mask[valid] = valid_indices

            # 如果存在可融合点，就开始融合
            # 显式几何加权平均：主要看 densities
            # 特征融合/GRU 更新：weights 作为辅助输入一起参与

            if mask.sum() > 0:
                # 把密度和权重做位置编码后送给 GRU，当成门控条件
                # 当前视角的新点对应的权重/密度信息编码
                input_weights_emb = positional_encoding(torch.cat([global_densities[:, mask], weight_emb[:, i, pixel_indices][:,fusion_indices_]], dim=-1), 6)
                # 当前全局池中旧点对应的权重/密度信息编码
                hidden_weights_emb = positional_encoding(torch.cat([densities[:, i, pixel_indices][:,fusion_indices_], global_weight_emb[:, mask]], dim=-1), 6)
                # GRU 融合“当前视角特征”和“全局特征”
                # 为什么用 GRU
                # GRU 本来是序列模型，但这里作者把它当成一种“带门控的特征更新器”：
                # 保留旧信息多少
                # 接纳新信息多少
                # 如何平衡两者
                fusion_feat = self.gru(gaussians[0][:, i, pixel_indices][:,fusion_indices_].unsqueeze(2),
                                       global_gaussians[:, mask].unsqueeze(2),
                                       input_weights_emb,
                                       hidden_weights_emb).squeeze(2)

                # 保留未参与融合的旧点 + 融合后的新点
                global_gaussians = torch.cat([global_gaussians[:, ~mask], fusion_feat], dim=1)
                # 使用密度作为融合权重（重复到兼容目标张量形状）
                weights_0 = global_densities[:, mask].repeat(1, 1, 1, 2)
                weights_1 = densities[:, i, pixel_indices][:,fusion_indices_].repeat(1, 1, 1, 2)

                # 位置、密度（权重）、深度权重（置信度）、外参、深度做加权融合，(旧值 * 旧权重 + 新值 * 新权重) / (旧权重 + 新权重)
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
            # 特征、位置、密度、深度权重、外参、深度
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

        # 返回融合后的特征、坐标、外参、深度和融合权重
        return global_gaussians, global_coords, global_extrinsics, global_depths, global_densities
