1\. 方法目标
针对当前项目的 FreeSplat模型做固定预算下的可学习高斯剪枝，首先需要引入一个轻量 Importance Head，用于对 PTF 融合后的全局高斯进行重要性评分。对于每个高斯  $i$ ，构造特征  $z_i=[f_i,\log(1+\omega_i),\alpha_i]$ ，并通过 MLP 输出重要性分数  $r_i$ 。训练阶段使用温度化 sigmoid 得到 soft gate  $m_i=\sigma(r_i/\tau(t))$ ，并以  $\tilde{\alpha}_i=m_i\alpha_i$  替换原始 opacity 参与渲染。总损失由渲染损失、预算一致性损失和二值化正则组成：$$\mathcal{L} = \mathcal{L}_{render} + \lambda_b \left( \frac{1}{M}\sum_i m_i-\rho \right)^2 + \lambda_{bin} \frac{1}{M}\sum_i m_i(1-m_i)$$，其中  $\rho$  为固定保留比例，表示最终用于渲染的高斯和PTF融合后高斯的数量比。推理阶段根据  $r_i$  进行 hard top-K 选择，仅保留前  $K=\lfloor \rho M\rfloor$  个高斯用于渲染。


2\. 模块设计
设 FreeSplat 原始前向在 PTF 和 Gaussian decoder 之后，得到全局高斯集合：
$$\mathcal{P}=\{(\mu_i,\Sigma_i,\alpha_i,s_i,f_i,\omega_i)\}_{i=1}^{M}$$
其中：
*    $\mu_i$ ：高斯中心
*    $\Sigma_i$ ：协方差
*    $\alpha_i$ ：不透明度
*    $s_i$ ：颜色/SH 参数
*    $f_i$ ：PTF 融合后的全局特征
*    $\omega_i$ ：PTF 累积融合权重

2.1 Importance Head
目标：学习一个轻量评分函数，对每个全局高斯输出重要性分数。这个分数不直接参与几何建模，而只负责排序与筛选。
输入：$$z_i=[f_i,\log(1+\omega_i),\alpha_i]$$
输出：
定义 importance score：
$$
r_i = F_{\text{imp}}(z_i)
$$
其中  $r_i\in\mathbb{R}$ 。

网络结构：采用最简单的 3 层 MLP：
$$
h_i^{(1)}=\text{GELU}(W_1 z_i+b_1)
$$
 
$$
h_i^{(2)}=\text{GELU}(W_2 h_i^{(1)}+b_2)
$$
 
$$
r_i=W_3 h_i^{(2)}+b_3
$$

推荐超参数：

*   输入维度： 66
*   hidden dim 1：128
*   hidden dim 2：64
*   输出维度：1
*   激活函数：GELU


2.2 Soft Gating
---------------

目标
--

训练阶段不做不可导的硬裁剪，而是通过一个连续 gate 去调节高斯对渲染的贡献。

公式
--

采用温度化 sigmoid：

$$
m_i=\sigma\!\left(\frac{r_i}{\tau(t)}\right)
$$

其中：

*    $m_i\in(0,1)$ 
*    $\tau(t)$  是温度参数， $\tau(t)$  越小，sigmoid 越陡，gate 越接近二值。因此用指数退火：
$$
\tau(t)=0.5\left(\frac{0.2}{0.5}\right)^{t/N}
$$
其中  $t$  是当前 step， $N$  是总训练步数。  
前期 gate 更软，便于学习排序；后期 gate 更尖，便于逼近真实剪枝。

然后只对 opacity 做软门控：

$$
\tilde{\alpha}_i = m_i \alpha_i
$$

最终用于渲染的高斯集合为：

$$
\tilde{\mathcal{P}}=\{(\mu_i,\Sigma_i,\tilde{\alpha}_i,s_i)\}_{i=1}^{M}
$$


* * *

2.3 损失函数
----------

目标
--

在训练中控制平均激活比例接近预设预算  $\rho$ ，同时避免所有 gate 都塌缩到  $\rho$  附近。

损失函数公式为：

$$
\mathcal{L} = \mathcal{L}_{render} + \lambda_b \left( \frac{1}{M}\sum_i m_i-\rho \right)^2 + \lambda_{bin} \frac{1}{M}\sum_i m_i(1-m_i)
$$

这三项分别是：渲染损失、预算损失和二值化正则
其中，
*   保留比例： $\rho=0.8$ ，表示最终渲染所用高斯与融合后高斯的比例
*    $\lambda_b=0.2$ 
*    $\lambda_{bin}=0.02$ 

* * *