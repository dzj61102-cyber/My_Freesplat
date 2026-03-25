# FreeSplat 逐文件到函数级阅读顺序

这份文档给你两套路线：
- 30 分钟：快速建立“主流程心智图”
- 2 小时：把核心模块读到可改代码

---

## 0. 先看总入口（两条路线都通用）

1. `README.md`
   - 只看运行命令和配置入口（`+experiment=...`）
2. `src/main.py`
   - 重点函数：`train`
   - 看清：配置加载 -> 模型组装 -> `trainer.fit/test`
3. `config/experiment/scannet/2views.yaml`
   - 先用一个最典型实验配置建立映射关系

---

## A. 30 分钟速读路线（先通主链路）

目标：搞清楚“一次训练 step 到底发生了什么”。

### 1) 训练总控（5 分钟）

文件顺序：
1. `src/main.py`
   - `train`
   - 关注 `get_encoder` / `get_decoder` / `ModelWrapper(...)`
2. `src/model/model_wrapper.py`
   - `ModelWrapper.__init__`
   - `training_step`

读完应回答：
- 数据从哪里来？
- encoder/decoder 在哪被调用？
- loss 在哪累加？

### 2) 模型核心（15 分钟）

文件顺序：
1. `src/model/encoder/__init__.py`
   - `ENCODERS`、`get_encoder`
2. `src/model/encoder/encoder_freesplat.py`
   - `EncoderFreeSplat.__init__`
   - `EncoderFreeSplat.forward`
   - `EncoderFreeSplat.fuse_gaussians`
3. `src/model/decoder/decoder_splatting_cuda.py`
   - `DecoderSplattingCUDA.forward`
4. `src/model/decoder/cuda_splatting.py`
   - `render_cuda`

读完应回答：
- feature -> depth -> gaussians 的转换在哪？
- 多视角 gaussian 融合在哪？
- 最终渲染调用的是哪个 CUDA 接口？

### 3) 数据和损失（10 分钟）

文件顺序：
1. `src/dataset/data_module.py`
   - `train_dataloader` / `val_dataloader` / `test_dataloader`
2. `src/dataset/dataset_scannet.py`
   - `__getitem__`
3. `src/dataset/view_sampler/view_sampler_bounded.py`
   - `sample`
4. `src/loss/loss_mse.py`
   - `LossMse.forward`
5. `src/loss/loss_lpips.py`
   - `LossLpips.forward`

读完应回答：
- context/target 视角是怎么采样的？
- 训练时监督的是哪些目标？

---

## B. 2 小时深读路线（可开始改模型）

目标：不仅能跑，还能定位改动点、评估影响范围。

## Phase 1: 启动与配置映射（15 分钟）

按顺序：
1. `config/main.yaml`
2. `config/experiment/scannet/2views.yaml`
3. `config/model/encoder/freesplat.yaml`
4. `config/model/decoder/splatting_cuda.yaml`
5. `src/config.py`
   - `RootCfg`
   - `load_typed_root_config`

产出：
- 画一张“配置键 -> 代码字段”映射表（如 `model.encoder.num_views` -> `EncoderFreeSplatCfg.num_views`）。

## Phase 2: 训练主流程（20 分钟）

按顺序：
1. `src/main.py`
   - `train`
2. `src/model/model_wrapper.py`
   - `__init__`
   - `training_step`
   - `validation_step`
   - `test_step`

重点追踪变量：
- `batch`
- `encoder_results['gaussians']`
- `output.color` / `output.depth`
- `total_loss`

## Phase 3: 编码器主干（35 分钟）

按顺序：
1. `src/model/encoder/encoder_freesplat.py`
   - `__init__`（看模块组装）
   - `forward`（主路径）
   - `fuse_gaussians`（融合细节）
2. `src/model/encoder/modules/cost_volume.py`
   - `build_cost_volume`
   - `warp_features`
3. `src/model/encoder/modules/networks.py`
   - `DepthDecoder.forward`
   - `CVEncoder.forward`
   - `GRU.forward`
4. `src/model/encoder/common/gaussian_adapter.py`
   - `GaussianAdapter.forward`
   - `get_scale_multiplier`

读法建议：
- 在 `forward` 中按注释分块读：2D 特征 -> cost volume -> depth decoder -> gaussian adapter -> fuse。

## Phase 4: 解码与渲染（20 分钟）

按顺序：
1. `src/model/decoder/decoder.py`
   - `DecoderOutput`（先看输入输出约定）
2. `src/model/decoder/decoder_splatting_cuda.py`
   - `forward`
3. `src/model/decoder/cuda_splatting.py`
   - `_make_rasterization_settings`
   - `_unpack_render_output`
   - `render_cuda`
   - `render_depth_cuda`
4. `docs/cuda_splatting_changes_summary.md`
   - 看你仓库里已有的兼容性修改背景

## Phase 5: 数据与采样（20 分钟）

按顺序：
1. `src/dataset/__init__.py`
   - `DATASETS`、`get_dataset`
2. `src/dataset/data_module.py`
3. `src/dataset/dataset_scannet.py`
   - `__getitem__`
   - `index`
4. `src/dataset/dataset_replica.py`（对比 test_fvs 分支）
5. `src/dataset/dataset_re10k.py`（IterableDataset 路径）
6. `src/dataset/view_sampler/view_sampler_bounded.py`
7. `src/dataset/view_sampler/view_sampler_evaluation.py`

产出：
- 弄清三件事：如何取 context、如何取 target、评估 index 如何覆盖默认采样。

## Phase 6: 损失/指标/几何工具（10 分钟）

按顺序：
1. `src/loss/__init__.py`
2. `src/loss/loss_mse.py`
3. `src/loss/loss_lpips.py`
4. `src/evaluation/metrics.py`
5. `src/geometry/projection.py`
   - `get_world_rays`
   - `project`

---

## C. 一眼定位“我要改哪里”

- 改多视角融合策略：`src/model/encoder/encoder_freesplat.py` 的 `fuse_gaussians`
- 改 depth/cost volume：`src/model/encoder/modules/cost_volume.py` + `src/model/encoder/modules/networks.py`
- 改高斯参数生成（scale/rot/SH）：`src/model/encoder/common/gaussian_adapter.py`
- 改渲染行为：`src/model/decoder/cuda_splatting.py`
- 改采样策略：`src/dataset/view_sampler/view_sampler_bounded.py` 或 `view_sampler_evaluation.py`
- 改训练目标：`src/loss/loss_mse.py`、`src/loss/loss_lpips.py`

---

## D. 推荐阅读节奏

- 第一次：严格按 30 分钟路线走，不抠细节。
- 第二次：走 2 小时路线，并在每个 Phase 记下“输入张量 shape”。
- 第三次（开始改代码前）：只重读你准备修改的那 2-3 个文件。

