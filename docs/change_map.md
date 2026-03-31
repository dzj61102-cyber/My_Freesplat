对每个输入视角  $i$ ：

### 1\. 光度变化图

直接对输入图像  $I_i$  求梯度幅值：

$$
g_{\text{photo},i} = \sqrt{\|\nabla_x I_i\|_2^2+\|\nabla_y I_i\|_2^2}
$$

实现上就是对 RGB 图像做 x/y 梯度，再按通道求 L2 范数。

* * *

### 2\. 几何变化图

先从当前视角的**像素对齐高斯中心**恢复深度图。  
你这里已经有每个像素对应的局部高斯 3D 坐标 `coords[0]`，以及全分辨率深度 `depth_outputs['depth_pred_s-1_b1hw']`。第一版直接用这个深度图即可。

Pasted code

由深度图求法线图  $n_i$ ，再对法线图求梯度幅值：

$$
g_{\text{geo},i} = \sqrt{\|\nabla_x n_i\|_2^2+\|\nabla_y n_i\|_2^2}
$$

* * *

### 3\. 最终变化图

$$
g_i=\frac{g_{\text{photo},i}+g_{\text{geo},i}}{2}
$$

然后做归一化到  $[0,1]$ 。

* * *

接到 FreeSplat 代码流里的 5 步
----------------------

### 第 1 步

在 `forward` 中，拿到：

*   原图 `context['image']`
*   全分辨率深度 `depth_outputs['depth_pred_s-1_b1hw']`

计算每个视角的  $g_i$ ，shape 做成：

$$
[B, V, H\!\times\!W, 1]
$$

这样和局部高斯 token 对齐。

Pasted code

* * *

### 第 2 步

把  $g_i$  当作每个局部高斯的局部变化值：

$$
m^{local}_{i,p}=g_i(p)
$$

* * *

### 第 3 步

在 `fuse_gaussians` 里，像融合密度那样维护一个 `global_change`：

*   匹配成功的高斯：按密度加权平均
    
$$
m^{global}_{new} = \frac{\alpha^{old}m^{old}+\alpha^{cur}m^{cur}} {\alpha^{old}+\alpha^{cur}+\epsilon}
$$
*   未匹配的高斯：直接追加当前局部变化值

最终每个全局高斯有一个连续变化值  $m^{global}_k$ 。

Pasted code

* * *

### 第 4 步

保留你现有 importance head：

$$
s_k = h([f_k,\log(1+w_k),\alpha_k^{base}])
$$
 
$$
g_k^{gate}=\sigma(s_k/\tau)
$$

不要用变化值直接替代 importance head，  
而是把  $m^{global}_k$  当作 **teacher**。

第一版推荐先做 soft target：

$$
y_k = \text{Normalize}(m^{global}_k)
$$

* * *

### 第 5 步

训练时仍用外乘门控：

$$
\alpha_k^{new}=\alpha_k^{base}\cdot g_k^{gate}
$$

总损失：

$$
\mathcal L = \mathcal L_{render} + \lambda_b \mathcal L_{budget} + \lambda_{bin}\mathcal L_{bin} + \lambda_{bce}\mathcal L_{bce}
$$

其中：

$$
\mathcal L_{budget} = \left(\frac{1}{M}\sum_k g_k^{gate}-\rho\right)^2
$$
 
$$
\mathcal L_{bin} = \frac{1}{M}\sum_k g_k^{gate}(1-g_k^{gate})
$$
 
$$
\mathcal L_{bce} = -\frac{1}{M}\sum_k \left[ y_k\log g_k^{gate}+(1-y_k)\log(1-g_k^{gate}) \right]
$$

* * *

第一版推荐参数
-------

$$
\rho=0.8,\quad \tau: 0.5\rightarrow 0.3,\quad \lambda_b=0.2,\quad \lambda_{bin}=0.005,\quad \lambda_{bce}=0.05
$$

* * *

最后一句
----
你应当用：

*   图像梯度幅值做 photometric variation
*   深度恢复法线后再做法线梯度幅值做 geometric variation
*   二者平均得到变化图

