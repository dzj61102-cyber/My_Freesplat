# `src/model/decoder/cuda_splatting.py` 改动总结

## 背景

这次修改的目标是让 `src/model/decoder/cuda_splatting.py` 兼容当前项目实际使用的
`diff-gaussian-rasterization-w-depth` 后端，解决测试阶段在 CUDA splatting 调用处出现的接口不匹配问题。

核心问题有两个：

1. `GaussianRasterizationSettings` 在当前后端里不接受 `debug` 参数。
2. `GaussianRasterizer(...)` 的返回值个数和深度张量形状，与原代码的假设不一致。

---

## 具体改动

### 1. 新增 `_make_rasterization_settings`

位置：`src/model/decoder/cuda_splatting.py:17`

作用：

- 统一构造 `GaussianRasterizationSettings`
- 仅当当前后端的 `_fields` 里存在 `debug` 字段时，才附加 `debug=False`

这样做的原因是：

- 当前环境中的 `diff_gaussian_rasterization` 不支持 `debug`
- 原先直接传 `debug=False` 会触发：

```text
TypeError: GaussianRasterizationSettings.__new__() got an unexpected keyword argument 'debug'
```

---

### 2. 新增 `_unpack_render_output`

位置：`src/model/decoder/cuda_splatting.py:23`

作用：

- 兼容 `rasterizer(...)` 返回 3 个值或 4 个值的情况
- 统一解包为 `image, radii, depth`

兼容逻辑：

- 返回 4 个值时，按 `image, radii, depth, _` 处理
- 返回 3 个值时，按 `image, radii, depth` 处理
- 其他情况直接抛出明确错误

---

### 3. 新增 `_unpack_image_output`

位置：`src/model/decoder/cuda_splatting.py:35`

作用：

- 兼容正交渲染路径 `render_cuda_orthographic(...)` 下不同返回值长度
- 统一解包为 `image, radii`

兼容逻辑：

- 支持 4 值、3 值、2 值返回
- 其他情况直接报错

---

### 4. 修改 `render_cuda(...)` 中的 rasterization settings 构造方式

位置：`src/model/decoder/cuda_splatting.py:132`

改动前：

- 直接调用 `GaussianRasterizationSettings(...)`
- 显式传入 `debug=False`

改动后：

- 改为调用 `_make_rasterization_settings(...)`
- 不再硬编码传 `debug=False`

收益：

- 兼容当前后端接口
- 保留对其他可能支持 `debug` 的后端版本的兼容性

---

### 5. 修改 `render_cuda(...)` 的 rasterizer 输出解包

位置：`src/model/decoder/cuda_splatting.py:151`

改动前：

- 代码固定按 4 个返回值解包

改动后：

- 通过 `_unpack_render_output(...)` 做兼容解包

这样可以避免后端返回值数量变化时直接崩溃。

---

### 6. 修复 `render_cuda(...)` 中的深度张量维度处理

位置：`src/model/decoder/cuda_splatting.py:161`

问题：

- 当前后端返回的 `depth` 可能已经是 `3` 维
- 原代码会无条件 `unsqueeze(0)`，导致输出维度多出一层
- 下游 `decoder_splatting_cuda.py` 在 `rearrange(...)` 时因此报维度错误

改动后：

- 若 `depth.ndim == 2`，补一个通道维
- 若 `depth.ndim == 3`，直接使用
- 若维度既不是 2 也不是 3，直接抛出明确错误

对应解决的问题是类似：

```text
EinopsError: Wrong shape: expected 4 dims. Received 5-dim tensor.
```

---

### 7. 修改 `render_cuda_orthographic(...)` 的 settings 构造和输出解包

位置：

- `src/model/decoder/cuda_splatting.py:237`
- `src/model/decoder/cuda_splatting.py:254`

改动内容：

- 同样改为使用 `_make_rasterization_settings(...)`
- 同样改为通过 `_unpack_image_output(...)` 解包输出

作用：

- 保证正交投影可视化路径与主渲染路径使用相同的兼容策略

---

## 这次改动解决了什么

这次对 `src/model/decoder/cuda_splatting.py` 的修改，主要解决了以下三类问题：

1. 当前 `diff-gaussian-rasterization-w-depth` 后端不支持 `debug` 参数导致的初始化报错
2. 不同版本 rasterizer 返回值个数不同导致的解包报错
3. 深度张量 shape 与下游 `einops.rearrange(...)` 预期不一致导致的维度报错

---

## 改动后的兼容思路

总体策略不是把代码硬编码到某一个返回格式，而是让该文件对以下差异做适配：

- `GaussianRasterizationSettings` 是否包含 `debug`
- `GaussianRasterizer(...)` 返回 2/3/4 个值
- `depth` 是二维还是三维

这样可以让 `src/model/decoder/cuda_splatting.py` 在当前项目依赖版本下稳定工作，也更容易适配后续可能切换的 rasterizer 实现。

---

## 相关文件

- `src/model/decoder/cuda_splatting.py`

如果后续还要继续整理本次调试全过程，可以再补一份包含 `model_wrapper.py` 可视化修复的完整排障记录。
