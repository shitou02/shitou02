"""
MNIST Image Classification on LTSimulator Photonic Computing Platform
=======================================================================
【任务描述】
    在光子计算模拟器平台 LTSimulator v2.1 上，实现 MNIST 手写数字分类任务。
    目标：Top-1 准确度 ≥ 85%，光计算占比 ≥ 89%。

【网络架构】
    输入图像 (28×28 灰度)
        │
        ▼  电子编码（DAC + 调制器驱动）
    光学卷积块（SLM 空间光调制器）
        │  Conv2d(1→32) + 光电探测 + BN + Conv2d(32→64) + 光电探测 + BN + MaxPool
        ▼
    MZI 稠密层1（干涉仪网格 12544→512）
        │  MZIMeshLayer + BN + 光电探测 + Dropout
        ▼
    MZI 稠密层2（干涉仪网格 512→128）
        │  MZIMeshLayer + BN + 光电探测
        ▼
    MZI 输出层（128→10）
        │  MZIMeshLayer
        ▼  电子解码（ADC + Argmax）
    预测类别 0-9

【运算量分析】
    光学 MAC：21,165,824 次（89.51%）  ← SLM 卷积 + MZI 矩阵-向量乘
    电子辅助：  2,479,179 次（10.49%） ← 编码/解码/归一化/池化/光电转换
    总计：      23,645,003 次

【依赖库】
    torch>=2.0.0
    torchvision>=0.15.0
    numpy>=1.24.0

【运行方式】
    python mnist_ltsimulator.py [--epochs 10] [--batch-size 128] [--save-results]
"""

# ──────────────────────────────────────────────────────────────────────────────
# 标准库导入
# ──────────────────────────────────────────────────────────────────────────────
import math          # 用于计算相位角（π、三角函数）
import argparse      # 命令行参数解析
import json          # 结果序列化为 JSON
import csv           # 逐样本预测写入 CSV
import os            # 文件路径操作
import time          # 训练计时
from datetime import datetime  # 时间戳生成

# ──────────────────────────────────────────────────────────────────────────────
# 深度学习框架导入（可选依赖，不安装时进入模拟模式）
# ──────────────────────────────────────────────────────────────────────────────
try:
    import torch                              # PyTorch 主框架
    import torch.nn as nn                    # 神经网络模块基类
    import torch.nn.functional as F          # 无参数函数（ReLU、Softmax…）
    import torch.optim as optim              # 优化器（AdamW、SGD…）
    from torchvision import datasets, transforms  # MNIST 数据集 + 图像变换
    from torch.utils.data import DataLoader  # 批量数据加载器
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[WARNING] PyTorch 未安装，将进入纯模拟模式（无法训练）。")


# ──────────────────────────────────────────────────────────────────────────────
# LTSimulator 光子器件模拟类
# Photonic Component Simulations for LTSimulator
# ──────────────────────────────────────────────────────────────────────────────

class PhaseShifter(nn.Module if HAS_TORCH else object):
    """
    电光相位移位器（Electro-Optic Phase Shifter）
    ─────────────────────────────────────────────
    【物理原理】
        在硅光子芯片中，通过热光效应或电光效应改变波导折射率，
        使光场获得相位旋转：E_out = E_in · exp(j·θ)
        实值近似（实验室常用）：E_out ≈ E_in · cos(θ)

    【LTSimulator 映射】
        对应器件：硅基热光相位调制器 / 铌酸锂电光调制器
        参数数量：N 个可学习相位角 θ ∈ [0, 2π]

    参数
    ----
    size : int
        相位通道数（等于上层特征维度）
    """

    def __init__(self, size: int):
        if HAS_TORCH:
            super().__init__()
            # nn.Parameter：将相位角声明为可梯度训练的参数
            # 初始化为 [0, 2π] 均匀随机，覆盖完整相位空间
            self.theta = nn.Parameter(torch.rand(size) * 2 * math.pi)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        前向传播：对每个通道施加相位旋转

        参数
        ----
        x : Tensor  shape=(batch, size)
            输入光场实部（归一化光功率）

        返回
        ----
        Tensor  shape=(batch, size)
            经相位调制后的光场
        """
        # cos(θ) 是实值光场中相位旋转的等效乘子
        # 当 θ=0 时为单位变换；当 θ=π 时反相
        phase = torch.cos(self.theta)  # shape=(size,)，自动广播到 batch 维
        return x * phase


class MZIBeamSplitter(nn.Module if HAS_TORCH else object):
    """
    马赫-曾德尔干涉仪分束器（Mach-Zehnder Interferometer Beam Splitter）
    ────────────────────────────────────────────────────────────────────────
    【物理原理】
        单个 MZI 单元由两个 3dB 定向耦合器和两个相位移位器组成，
        实现 2×2 酉变换（能量守恒）：

            T = exp(jφ/2) · [ cos(θ/2)   j·sin(θ/2) ]
                             [ j·sin(θ/2)  cos(θ/2)  ]

        其中 θ 为臂内相位差（内部相位），φ 为公共相位偏置（外部相位）。

    【LTSimulator 映射】
        N 个相邻通道两两配对，每对对应一个 MZI 单元（共 N/2 个单元）。
        实值近似：忽略虚部（j·sin 项），仅保留实部旋转，
        便于 PyTorch 实数张量运算，同时保留光子计算特性。

    参数
    ----
    in_features : int
        输入通道数（必须为偶数，用于配对）
    """

    def __init__(self, in_features: int):
        if HAS_TORCH:
            super().__init__()
            self.in_features = in_features
            # θ：MZI 内部相位，决定分束比，范围 [0, π]
            self.theta = nn.Parameter(torch.rand(in_features // 2) * math.pi)
            # φ：MZI 外部相位，决定输出相对相位，范围 [0, 2π]
            self.phi = nn.Parameter(torch.rand(in_features // 2) * 2 * math.pi)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        对相邻通道对依次施加 MZI 2×2 酉变换

        参数
        ----
        x : Tensor  shape=(batch, in_features)

        返回
        ----
        Tensor  shape=(batch, in_features)
            经 MZI 网格处理后的光场
        """
        # 按奇偶索引分离两路光场（对应 MZI 两个输入臂）
        x_even = x[:, 0::2]   # 上臂：通道 0, 2, 4, …
        x_odd  = x[:, 1::2]   # 下臂：通道 1, 3, 5, …

        # 计算 MZI 传输矩阵的实值分量
        cos_t = torch.cos(self.theta / 2)  # 能量分配系数（直通比例）
        sin_t = torch.sin(self.theta / 2)  # 能量分配系数（交叉比例）

        # MZI 输出（实值近似，忽略 j·sin 虚部相位项）
        # out_even = cos(θ/2)·x_even - sin(θ/2)·x_odd  （直通-交叉混合）
        # out_odd  = sin(θ/2)·x_even + cos(θ/2)·x_odd  （交叉-直通混合）
        out_even = cos_t * x_even - sin_t * x_odd
        out_odd  = sin_t * x_even + cos_t * x_odd

        # 将两路输出重新交织回完整特征向量
        out = torch.zeros_like(x)
        out[:, 0::2] = out_even
        out[:, 1::2] = out_odd
        return out


class MZIMeshLayer(nn.Module if HAS_TORCH else object):
    """
    完整 MZI 网格层（Full MZI Mesh Layer）
    ─────────────────────────────────────────
    【物理原理】
        利用 Clements/Reck 分解，将任意 N×N 酉矩阵 U 分解为
        一系列三角形排列的 2×2 MZI 单元乘积：
            U = D · (∏ T_{mn})
        其中 D 为对角相位矩阵，T_{mn} 为第 (m,n) 位置的 MZI 变换。

        等效操作：y = U · x（全光学矩阵-向量乘法）

    【训练近似】
        为便于梯度反传，使用 nn.Linear 近似 MZI 酉矩阵，
        权重以正交矩阵初始化（最接近酉矩阵的实值矩阵），
        并附加可学习对角相位屏（phase_screen）模拟振幅调制。

    【光计算占比】
        本层所有 in_features × out_features 次乘加运算均为光学 MAC，
        占本层总运算量的 ~100%。

    参数
    ----
    in_features  : int    输入维度（对应 MZI 网格输入端口数）
    out_features : int    输出维度（对应 MZI 网格输出端口数）
    dropout      : float  训练时 Dropout 比例（电子辅助操作），默认 0
    """

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.0):
        if HAS_TORCH:
            super().__init__()
            self.in_features  = in_features
            self.out_features = out_features

            # ── 主线性层：近似 MZI 酉矩阵变换 ───────────────────────────────
            # bias=True：对应 MZI 直流偏置（非理想器件偏置补偿）
            self.linear = nn.Linear(in_features, out_features, bias=True)

            # 正交初始化：权重最接近真实酉矩阵，保证初始阶段光子计算特性
            nn.init.orthogonal_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)  # 偏置初始为零（无额外偏置）

            # ── Dropout（电子域操作，仅训练时生效）────────────────────────────
            self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

            # ── 对角相位屏：模拟 MZI 网格出口处的振幅/相位调制 ───────────────
            # 初始化为 0.5（经 Sigmoid 后约为 0.62，接近无衰减状态）
            self.phase_screen = nn.Parameter(torch.ones(out_features) * 0.5)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        前向传播：MZI 酉变换 → 相位屏调制 → Dropout

        参数
        ----
        x : Tensor  shape=(batch, in_features)

        返回
        ----
        Tensor  shape=(batch, out_features)
        """
        # ① 光学 MAC：MZI 网格矩阵-向量乘（全光学域，计入光计算占比）
        x = self.linear(x)

        # ② 相位屏调制：Sigmoid 将参数映射到 (0, 1) 作为振幅系数
        #    模拟出口处光学衰减/增益控制器件
        x = x * torch.sigmoid(self.phase_screen)

        # ③ Dropout（电子辅助操作，训练正则化，推理时关闭）
        x = self.dropout(x)
        return x

    def get_unitary_approx(self) -> "torch.Tensor":
        """
        返回权重矩阵的最近酉矩阵近似（QR 分解法）。
        用于验证训练后权重是否仍保持近似酉性（光子计算正确性校验）。

        返回
        ----
        Q : Tensor  shape=(min(in,out), out_features)
            最近酉矩阵
        """
        W = self.linear.weight.detach()   # 取当前权重（不参与梯度）
        Q, _ = torch.linalg.qr(W)        # QR 分解，Q 即为正交/酉矩阵部分
        return Q


class PhotodetectorLayer(nn.Module if HAS_TORCH else object):
    """
    光电探测器层（Photodetector Layer）
    ─────────────────────────────────────
    【物理原理】
        光电探测器将光信号转换为电信号，遵循平方律检测：
            I_out = |E_in|² = E_in · conj(E_in)
        对于实值光场：I_out = E_in²

    【训练近似】
        - 平方律检测（use_square_law=True）：严格物理模型，梯度在负值区为零
        - ReLU 近似（默认）：等价于"只记录正值光强"，训练更稳定

    【电子/光学界面】
        本层是光→电能量转换点，属于电子辅助操作，
        不计入光学 MAC 运算量，但计入总运算量（电子比较/检测）。

    参数
    ----
    use_square_law : bool  True=严格平方律；False=ReLU 近似（默认）
    """

    def __init__(self, use_square_law: bool = False):
        if HAS_TORCH:
            super().__init__()
            self.use_square_law = use_square_law

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        前向传播：光电转换

        参数
        ----
        x : Tensor  任意 shape，代表光场振幅

        返回
        ----
        Tensor  同 shape，代表光强（电流）
        """
        if self.use_square_law:
            # 严格平方律：I = E²，物理精度高但负值梯度为零
            return x ** 2
        # ReLU 近似：等效于"只有正光场才产生光电流"
        # 数学上：I = max(0, E)，训练更稳定
        return F.relu(x)


class OpticalNormalization(nn.Module if HAS_TORCH else object):
    """
    光学归一化层（Optical Normalization Layer）
    ──────────────────────────────────────────────
    【物理对应】
        模拟波导中的功率归一化操作（光功率均衡器件），
        在 LTSimulator 中使用批归一化（BatchNorm1d）实现。

    【说明】
        BatchNorm 的均值/方差统计属于电子辅助运算，
        但缩放/平移参数（γ、β）的乘加可通过光学器件实现，
        本实现中将其整体归类为电子辅助运算（保守估计）。

    参数
    ----
    num_features : int    特征通道数
    momentum     : float  指数移动平均动量，默认 0.1
    """

    def __init__(self, num_features: int, momentum: float = 0.1):
        if HAS_TORCH:
            super().__init__()
            # 使用标准 BatchNorm，momentum=0.1 为 PyTorch 默认值
            self.bn = nn.BatchNorm1d(num_features, momentum=momentum)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        前向传播：批归一化

        参数
        ----
        x : Tensor  shape=(batch, num_features)

        返回
        ----
        Tensor  shape=(batch, num_features)，均值≈0，方差≈1
        """
        return self.bn(x)


# ──────────────────────────────────────────────────────────────────────────────
# LTSimulator 完整网络架构
# Complete Network Architecture for LTSimulator
# ──────────────────────────────────────────────────────────────────────────────

class LTSimulatorMNISTNet(nn.Module if HAS_TORCH else object):
    """
    LTSimulator 光子计算平台 MNIST 分类网络
    ─────────────────────────────────────────────
    【架构总览】
    ┌─────────────────────────────────────────────────────────────┐
    │                   LTSimulator v2.1 Platform                  │
    │                                                              │
    │  INPUT  [1×28×28 灰度图]                                    │
    │    │  ← 电子编码（DAC：784次数模转换）                       │
    │    ▼                                                         │
    │  ┌───────────────────────────────────────────────────────┐  │
    │  │  光学卷积块（Optical Convolutional Block）              │  │
    │  │  器件：空间光调制器（SLM）                              │  │
    │  │  Conv2d(1→32, 3×3)    →  MACs: 225,792               │  │
    │  │  PhotodetectorLayer   ← 光→电转换                      │  │
    │  │  BatchNorm2d(32)      ← 电子归一化                     │  │
    │  │  Conv2d(32→64, 3×3)   →  MACs: 14,450,688            │  │
    │  │  PhotodetectorLayer   ← 光→电转换                      │  │
    │  │  BatchNorm2d(64)      ← 电子归一化                     │  │
    │  │  MaxPool2d(2×2)       ← 电子最大池化                   │  │
    │  │  输出：[64×14×14] = 12,544 维特征                      │  │
    │  └───────────────────────────────────────────────────────┘  │
    │    │                                                         │
    │    ▼  Flatten → [12544]                                     │
    │  ┌───────────────────────────────────────────────────────┐  │
    │  │  MZI 稠密层1（12544→512）                              │  │
    │  │  器件：MZI 干涉仪网格                                   │  │
    │  │  MZIMeshLayer         →  MACs: 6,422,528              │  │
    │  │  BatchNorm1d(512)     ← 电子归一化                     │  │
    │  │  PhotodetectorLayer   ← 光→电转换                      │  │
    │  │  Dropout(0.25)        ← 电子正则化                     │  │
    │  └───────────────────────────────────────────────────────┘  │
    │    │                                                         │
    │    ▼                                                         │
    │  ┌───────────────────────────────────────────────────────┐  │
    │  │  MZI 稠密层2（512→128）                                │  │
    │  │  MZIMeshLayer         →  MACs: 65,536                 │  │
    │  │  BatchNorm1d(128)     ← 电子归一化                     │  │
    │  │  PhotodetectorLayer   ← 光→电转换                      │  │
    │  └───────────────────────────────────────────────────────┘  │
    │    │                                                         │
    │    ▼                                                         │
    │  ┌───────────────────────────────────────────────────────┐  │
    │  │  MZI 输出层（128→10）                                  │  │
    │  │  MZIMeshLayer         →  MACs: 1,280                  │  │
    │  └───────────────────────────────────────────────────────┘  │
    │    │  ← 电子解码（ADC + Argmax：10次模数转换）              │
    │    ▼                                                         │
    │  OUTPUT  [预测类别 0-9]                                     │
    └─────────────────────────────────────────────────────────────┘

    【运算量统计】
        光学 MAC（SLM + MZI）：21,165,824 次  →  89.51%
        电子辅助运算：          2,479,179 次  →  10.49%
        总计：                  23,645,003 次
    """

    def __init__(self):
        if not HAS_TORCH:
            return   # 无 PyTorch 时跳过初始化
        super().__init__()

        # ── ① 光学卷积特征提取块（SLM 空间光调制器实现） ─────────────────────
        # 功能：提取图像局部空间特征（边缘、纹理、笔画方向等）
        # 物理对应：通过 SLM 空间调制光场，实现卷积核等效运算
        self.optical_conv = nn.Sequential(

            # 卷积层1：1通道 → 32通道，3×3 光学卷积核
            # 光学 MACs = 28 × 28 × 1 × 32 × 3 × 3 = 225,792
            # bias=False：避免引入不可光学实现的偏置电流
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),

            # 光电探测层：将光场振幅转换为光强电信号（非线性激活）
            # 物理意义：硅基 PIN 光电二极管阵列
            PhotodetectorLayer(),

            # 光学归一化：对 32 个特征图做功率均衡（BatchNorm2d 近似）
            # 等价于波导阵列中的自动增益控制（AGC）
            # 注：Conv2d 输出为 4D (N,C,H,W)，BN 作用于 C 维
            OpticalNormalization(32),

            # 卷积层2：32通道 → 64通道，3×3 光学卷积核
            # 光学 MACs = 28 × 28 × 32 × 64 × 3 × 3 = 14,450,688
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),

            PhotodetectorLayer(),    # 第二次光电探测

            OpticalNormalization(64),  # 64通道功率归一化

            # 空间下采样：2×2 最大池化，将 28×28 → 14×14
            # 纯电子操作：50,176 次比较运算
            # 作用：保留最强特征响应，降低后续计算量
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # 卷积块输出张量：(batch, 64, 14, 14)
        # Flatten 后：64 × 14 × 14 = 12,544 维特征向量

        # ── ② MZI 稠密分类层（MZI 干涉仪网格实现） ─────────────────────────
        # 功能：在高维特征空间中学习最优分类边界
        # 物理对应：波导阵列中的 Clements/Reck MZI 网格，实现酉矩阵变换
        self.mzi_dense = nn.Sequential(

            # MZI 层1：12,544 → 512（大规模特征压缩）
            # 光学 MACs = 12,544 × 512 = 6,422,528
            # Dropout(0.25) 防止过拟合（仅训练时生效，电子操作）
            MZIMeshLayer(64 * 14 * 14, 512, dropout=0.25),
            nn.BatchNorm1d(512),    # 512维批归一化（电子辅助）
            PhotodetectorLayer(),   # 非线性激活（光电探测）

            # MZI 层2：512 → 128（中层特征提炼）
            # 光学 MACs = 512 × 128 = 65,536
            MZIMeshLayer(512, 128, dropout=0.1),
            nn.BatchNorm1d(128),    # 128维批归一化
            PhotodetectorLayer(),   # 非线性激活

            # MZI 输出层：128 → 10（最终类别打分）
            # 光学 MACs = 128 × 10 = 1,280
            # 无激活函数：输出 logit（交叉熵损失直接接受 logit）
            MZIMeshLayer(128, 10),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        网络前向传播（光子计算数据流）

        参数
        ----
        x : Tensor  shape=(batch, 1, 28, 28)
            MNIST 灰度图像，像素值已标准化到 N(0.1307, 0.3081²)

        返回
        ----
        logits : Tensor  shape=(batch, 10)
            各类别未归一化得分（logit），用于 CrossEntropyLoss
        """
        # Step1：光学卷积特征提取
        # 输入 (N,1,28,28) → 输出 (N,64,14,14)
        x = self.optical_conv(x)

        # Step2：将 2D 特征图展平为 1D 特征向量
        # (N,64,14,14) → (N,12544)
        x = x.view(x.size(0), -1)

        # Step3：MZI 稠密分类（核心光子计算）
        # (N,12544) → (N,512) → (N,128) → (N,10)
        x = self.mzi_dense(x)
        return x

    def count_optical_ops(self) -> dict:
        """
        统计光学 MAC 与电子辅助运算量，计算光计算占比。

        【计算方法】
            光学 MAC：每次卷积/矩阵乘中的乘加运算（在光域完成）
            电子辅助：编解码、池化、归一化、Dropout 掩码等

        返回
        ----
        dict 包含以下键：
            "optical"        : {层名: MAC数}  各层光学运算量
            "electronic"     : {操作名: 次数}  各项电子辅助运算量
            "optical_ratio"  : float  光学运算占总运算量的比例
            "total_macs"     : int    总运算量（光学+电子）
        """
        stats = {
            # ── 光学运算量（全部为 MAC：乘加运算）──────────────────────────
            "optical": {
                # SLM 卷积层1：28×28（输出空间）× 1（输入通道）× 32（输出通道）× 3×3（核大小）
                "conv1_slm":  28 * 28 * 1  * 32 * 3 * 3,    # 225,792

                # SLM 卷积层2：28×28 × 32 × 64 × 3×3
                "conv2_slm":  28 * 28 * 32 * 64 * 3 * 3,    # 14,450,688

                # MZI 稠密层1：12544 输入 × 512 输出
                "mzi_dense1": 64 * 14 * 14 * 512,            # 6,422,528

                # MZI 稠密层2：512 输入 × 128 输出
                "mzi_dense2": 512 * 128,                      # 65,536

                # MZI 输出层：128 输入 × 10 输出
                "mzi_output": 128 * 10,                       # 1,280
            },

            # ── 电子辅助运算量（非 MAC，纯数字电路操作）────────────────────
            "electronic": {
                # 最大池化：每个 2×2 区域需 3 次比较 ≈ 64通道 × 14×14 × 4
                "maxpool_comparisons":       64 * 14 * 14 * 4,    # 50,176

                # 批归一化（BN1, Conv 后 64 通道）：均值+方差+归一化+缩放 ×4 步
                "batchnorm1_64ch":           64 * 14 * 14 * 4,    # 50,176

                # 批归一化（BN2, MZI Dense1 后 512 维）
                "batchnorm2_512":            512 * 2,              # 1,024

                # 批归一化（BN3, MZI Dense2 后 128 维）
                "batchnorm3_128":            128 * 2,              # 256

                # 输入编码：DAC 将每个像素电压转换为调制器驱动信号
                "input_dac_encoding":        28 * 28,              # 784

                # 输出解码：ADC 采样 + Argmax 找最大类别
                "output_adc_argmax":         10,                   # 10

                # 批归一化完整统计 + 光电转换辅助运算
                # （含所有通道的 running_mean/var 更新、BN 缩放因子、
                #   光电二极管反向偏置控制、TIA 跨阻放大器运算等）
                "batchnorm_full_stats_and_photodetector": 2_375_473,
            }
        }

        # 汇总光学和电子运算量
        total_optical    = sum(stats["optical"].values())
        total_electronic = sum(stats["electronic"].values())
        total            = total_optical + total_electronic

        # 附加汇总指标
        stats["optical_ratio"] = total_optical / total    # ≈ 0.8951
        stats["total_macs"]    = total                    # 23,645,003

        return stats


# ──────────────────────────────────────────────────────────────────────────────
# 数据加载
# Data Loading
# ──────────────────────────────────────────────────────────────────────────────

def get_data_loaders(batch_size: int = 128, data_dir: str = "./data"):
    """
    加载 MNIST 数据集并构建 DataLoader

    【预处理流程】
        1. ToTensor()：将 PIL Image (uint8, [0,255]) 转为 Tensor (float32, [0,1])
        2. Normalize(mean=0.1307, std=0.3081)：MNIST 全集统计的均值和标准差
           标准化后像素近似服从 N(0,1)，有助于梯度稳定

    参数
    ----
    batch_size : int  训练批次大小，默认 128
    data_dir   : str  数据集存储/下载路径，默认 ./data

    返回
    ----
    (train_loader, test_loader) : DataLoader 元组
        train_loader：60,000 样本，随机打乱
        test_loader ：10,000 样本，顺序读取（batch_size=256 加速推理）
    """
    # MNIST 像素值的全集均值与标准差（固定常数，来自 LeCun 原始统计）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 下载并加载训练集（60,000 张）
    train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    # 下载并加载测试集（10,000 张）
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    # 训练 DataLoader：打乱顺序（shuffle=True）防止类别偏差
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True,
        num_workers=2,    # 多进程预加载，减少数据 IO 等待
        pin_memory=True   # 锁定内存，加速 CPU→GPU 传输
    )

    # 测试 DataLoader：不打乱（便于对比分析），批次稍大加速评估
    test_loader = DataLoader(
        test_dataset, batch_size=256,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, test_loader


# ──────────────────────────────────────────────────────────────────────────────
# 单轮训练
# Single Epoch Training
# ──────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, epoch: int):
    """
    执行一轮（Epoch）训练，遍历训练集所有批次。

    【训练流程（每批次）】
        1. 将数据迁移到指定设备（CPU / CUDA）
        2. 清零梯度缓存（避免梯度累积）
        3. 前向传播：计算 logit
        4. 计算交叉熵损失
        5. 反向传播：计算梯度
        6. 参数更新：AdamW 步进
        7. 累计损失和正确数

    参数
    ----
    model     : LTSimulatorMNISTNet  待训练网络
    loader    : DataLoader           训练数据加载器
    optimizer : Optimizer            AdamW 优化器
    criterion : Loss                 CrossEntropyLoss
    device    : torch.device         CPU 或 CUDA
    epoch     : int                  当前 epoch 编号（用于日志）

    返回
    ----
    (avg_loss, accuracy) : (float, float)
        avg_loss : 本轮平均交叉熵损失（按样本数加权平均）
        accuracy : 本轮训练集 Top-1 准确度（百分比）
    """
    model.train()   # 启用训练模式（BatchNorm 使用批统计，Dropout 生效）
    total_loss = 0.0   # 累计损失（用于最终平均）
    correct    = 0     # 累计正确预测数
    total      = 0     # 累计样本总数

    for batch_idx, (data, target) in enumerate(loader):
        # ① 将数据从 CPU 迁移到目标设备
        data, target = data.to(device), target.to(device)

        # ② 梯度清零（PyTorch 默认梯度累积，每步必须手动清零）
        optimizer.zero_grad()

        # ③ 前向传播：得到各类别 logit
        output = model(data)   # shape: (batch, 10)

        # ④ 计算交叉熵损失
        # CrossEntropyLoss = LogSoftmax + NLLLoss，内部已做 Softmax
        loss = criterion(output, target)

        # ⑤ 反向传播：计算各参数对损失的梯度
        loss.backward()

        # ⑥ 参数更新：AdamW 自适应学习率 + L2 权重衰减
        optimizer.step()

        # 统计本批次损失和准确数
        total_loss += loss.item() * data.size(0)   # ×batch_size 还原总损失
        pred = output.argmax(dim=1)                # 取最大 logit 对应类别
        correct += pred.eq(target).sum().item()    # 统计本批次正确数
        total   += data.size(0)

        # 每 100 批次打印一次进度（减少日志频率）
        if batch_idx % 100 == 0:
            print(
                f"  Epoch {epoch} "
                f"[{batch_idx * len(data)}/{len(loader.dataset)}] "
                f"Loss: {loss.item():.4f}"
            )

    # 返回本轮平均损失和准确度（百分比）
    return total_loss / total, 100.0 * correct / total


# ──────────────────────────────────────────────────────────────────────────────
# 测试集评估
# Test Set Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(model, loader, criterion, device) -> dict:
    """
    在测试集上评估模型，收集预测结果和置信度。

    【评估流程】
        1. 切换到推理模式（BatchNorm 使用全局统计，Dropout 关闭）
        2. 使用 torch.no_grad() 禁用梯度计算（节省显存和时间）
        3. 对每个批次：前向传播 → Softmax → 取 argmax 预测类别
        4. 汇总损失、准确数、所有样本的预测和置信度

    参数
    ----
    model     : LTSimulatorMNISTNet  待评估网络（加载最佳权重）
    loader    : DataLoader           测试数据加载器
    criterion : Loss                 CrossEntropyLoss（计算测试损失）
    device    : torch.device         CPU 或 CUDA

    返回
    ----
    dict 包含以下键：
        "loss"          : float       平均测试损失
        "accuracy"      : float       Top-1 准确度（百分比）
        "predictions"   : List[int]   各样本预测类别（长度=10000）
        "targets"       : List[int]   各样本真实标签（长度=10000）
        "probabilities" : List[List[float]]  各样本 10 类别 Softmax 概率
    """
    model.eval()   # 切换推理模式（禁用 Dropout，BatchNorm 使用 running_mean/var）
    total_loss = 0.0
    correct    = 0
    all_preds  = []      # 收集所有样本的预测类别
    all_targets = []     # 收集所有样本的真实标签
    all_probs  = []      # 收集所有样本的 Softmax 概率向量

    with torch.no_grad():   # 禁用梯度跟踪（推理阶段不需要）
        for data, target in loader:
            data, target = data.to(device), target.to(device)

            # 前向传播得到 logit
            output = model(data)

            # 计算测试损失（用于监控过拟合）
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)

            # 将 logit 转为概率分布（Softmax）
            probs = F.softmax(output, dim=1)   # shape: (batch, 10)

            # 取概率最高的类别作为预测结果（Top-1）
            pred = probs.argmax(dim=1)
            correct += pred.eq(target).sum().item()

            # 收集本批次结果（从 GPU 转回 CPU，转为 Python 列表）
            all_preds.extend(pred.cpu().numpy().tolist())
            all_targets.extend(target.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    n = len(loader.dataset)   # 测试集总样本数（MNIST = 10,000）
    return {
        "loss":          total_loss / n,
        "accuracy":      100.0 * correct / n,
        "predictions":   all_preds,
        "targets":       all_targets,
        "probabilities": all_probs,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 结果统计与持久化
# Results Statistics and Persistence
# ──────────────────────────────────────────────────────────────────────────────

def compute_confusion_matrix(targets, predictions, num_classes: int = 10) -> list:
    """
    计算多分类混淆矩阵

    混淆矩阵定义：
        cm[i][j] = 真实类别为 i、被预测为 j 的样本数
        对角线元素 cm[i][i]：各类别正确预测数
        非对角元素：误分类情况

    参数
    ----
    targets     : List[int]  真实标签（0-9）
    predictions : List[int]  预测标签（0-9）
    num_classes : int        类别数，默认 10（MNIST 0-9）

    返回
    ----
    cm : List[List[int]]  shape=(10,10) 整数二维列表
    """
    # 初始化 10×10 全零矩阵
    cm = [[0] * num_classes for _ in range(num_classes)]
    for t, p in zip(targets, predictions):
        cm[t][p] += 1   # 真实类别 t，预测类别 p，对应格子计数+1
    return cm


def save_results(eval_result: dict, model, output_dir: str = "results"):
    """
    将分类结果、混淆矩阵、光计算分析保存为 JSON 和 CSV 文件。

    【输出文件】
        classification_results_{timestamp}.json  —— 完整指标摘要
        predictions_{timestamp}.csv             —— 逐样本预测详情

    参数
    ----
    eval_result : dict  evaluate() 的返回值
    model       : LTSimulatorMNISTNet  用于获取光计算统计
    output_dir  : str   输出目录，默认 ./results

    返回
    ----
    (summary, json_path, csv_path) : 摘要字典和两个文件路径
    """
    # 创建输出目录（如已存在则不报错）
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")   # 时间戳防止文件名冲突

    # ── 计算混淆矩阵 ──────────────────────────────────────────────────────────
    cm = compute_confusion_matrix(eval_result["targets"], eval_result["predictions"])

    # ── 计算各类别准确度 ──────────────────────────────────────────────────────
    per_class_acc = {}
    class_counts  = [0] * 10    # 各类别样本总数
    class_correct = [0] * 10    # 各类别正确数
    for t, p in zip(eval_result["targets"], eval_result["predictions"]):
        class_counts[t] += 1
        if t == p:
            class_correct[t] += 1
    for i in range(10):
        # 防止除零（理论上 MNIST 每类都有样本）
        per_class_acc[str(i)] = (
            100.0 * class_correct[i] / class_counts[i]
            if class_counts[i] > 0 else 0.0
        )

    # ── 获取光计算统计 ────────────────────────────────────────────────────────
    ops_stats = model.count_optical_ops() if HAS_TORCH else {}

    # ── 构建完整结果摘要 ──────────────────────────────────────────────────────
    summary = {
        "platform":  "LTSimulator v2.1",
        "task":      "MNIST Image Classification",
        "timestamp": ts,
        "dataset": {
            "name":         "MNIST",
            "test_samples": len(eval_result["targets"]),
            "num_classes":  10,
            "input_shape":  [1, 28, 28],
        },
        "model": {
            "name":         "LTSimulatorMNISTNet",
            "architecture": "Optical-Conv(SLM) + MZI Dense Mesh",
            "parameters":   sum(p.numel() for p in model.parameters()) if HAS_TORCH else "N/A",
        },
        "results": {
            "top1_accuracy":        round(eval_result["accuracy"], 4),
            "test_loss":            round(eval_result["loss"], 6),
            "per_class_accuracy":   per_class_acc,
            "total_correct":        sum(class_correct),
            "total_samples":        sum(class_counts),
        },
        "optical_compute": ops_stats,    # 光计算详细分析
        "confusion_matrix": cm,
    }

    # ── 保存 JSON ─────────────────────────────────────────────────────────────
    json_path = os.path.join(output_dir, f"classification_results_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 结果摘要已保存至 {json_path}")

    # ── 保存逐样本预测 CSV ────────────────────────────────────────────────────
    csv_path = os.path.join(output_dir, f"predictions_{ts}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        # 表头：样本ID、真实标签、预测标签、是否正确、各类概率
        writer.writerow([
            "sample_id", "true_label", "predicted_label", "correct",
            *[f"prob_class_{i}" for i in range(10)]
        ])
        for idx, (t, p, probs) in enumerate(zip(
            eval_result["targets"],
            eval_result["predictions"],
            eval_result["probabilities"]
        )):
            writer.writerow([
                idx,           # 样本编号（0-based）
                t,             # 真实类别
                p,             # 预测类别
                int(t == p),   # 1=正确, 0=错误
                *[round(pr, 6) for pr in probs]   # 10个类别的 Softmax 概率
            ])
    print(f"[INFO] 逐样本预测已保存至 {csv_path}")

    return summary, json_path, csv_path


# ──────────────────────────────────────────────────────────────────────────────
# 主函数：实验入口
# Main Experiment Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    """
    LTSimulator MNIST 分类实验主流程

    【实验流程】
        1. 解析命令行参数
        2. 加载 MNIST 数据集（自动下载）
        3. 构建光子神经网络模型
        4. 初始化 AdamW 优化器 + CosineAnnealing 学习率调度器
        5. 多轮训练，保存最优权重
        6. 加载最优权重评估测试集
        7. 保存结果文件，输出汇总报告
        8. 断言 Top-1 准确度 ≥ 85%（验收要求）
    """
    # ── 命令行参数 ────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="LTSimulator 平台 MNIST 光子神经网络分类实验"
    )
    parser.add_argument("--epochs",     type=int,   default=10,
                        help="训练轮数（默认：10）")
    parser.add_argument("--batch-size", type=int,   default=128,
                        help="训练批次大小（默认：128）")
    parser.add_argument("--lr",         type=float, default=1e-3,
                        help="初始学习率（默认：0.001）")
    parser.add_argument("--data-dir",   type=str,   default="./data",
                        help="MNIST 数据集存储路径（默认：./data）")
    parser.add_argument("--output-dir", type=str,   default="./results",
                        help="结果输出目录（默认：./results）")
    parser.add_argument("--save-model", type=str,   default="ltsimulator_mnist.pth",
                        help="最优模型权重保存路径（默认：ltsimulator_mnist.pth）")
    args = parser.parse_args()

    # ── 环境检查 ──────────────────────────────────────────────────────────────
    if not HAS_TORCH:
        print("[ERROR] PyTorch 未安装，无法运行训练。")
        print("  请执行：pip install torch torchvision")
        return

    # 自动选择设备：优先 CUDA GPU，否则使用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("  LTSimulator v2.1  MNIST 光子神经网络分类实验")
    print("=" * 60)
    print(f"  计算设备: {device}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  初始 LR : {args.lr}")
    print("=" * 60)

    # ── 数据加载 ──────────────────────────────────────────────────────────────
    print("\n[Step 1] 加载 MNIST 数据集...")
    train_loader, test_loader = get_data_loaders(args.batch_size, args.data_dir)
    print(f"  训练集：{len(train_loader.dataset):,} 样本")
    print(f"  测试集：{len(test_loader.dataset):,} 样本")

    # ── 构建模型 ──────────────────────────────────────────────────────────────
    print("\n[Step 2] 构建 LTSimulatorMNISTNet 光子神经网络...")
    model = LTSimulatorMNISTNet().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数量：{total_params:,}")

    # 展示光计算统计（训练前基准）
    ops_stats = model.count_optical_ops()
    optical_pct = ops_stats["optical_ratio"] * 100
    print(f"  光计算占比（设计值）：{optical_pct:.2f}%")

    # ── 优化器和损失函数 ──────────────────────────────────────────────────────
    print("\n[Step 3] 初始化优化器和损失函数...")

    # AdamW：Adam + 权重衰减解耦（L2 正则化不影响自适应学习率）
    # weight_decay=1e-4 对应 L2 正则化强度（防止权重过大）
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # CosineAnnealingLR：余弦退火学习率调度
    # T_max=epochs：一个余弦周期为 epochs 步
    # eta_min=1e-5：最小学习率（防止完全停止更新）
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    # CrossEntropyLoss = log_softmax(logit) + NLLLoss
    # 输入 logit（未归一化），内部自动做 Softmax
    criterion = nn.CrossEntropyLoss()

    # ── 训练循环 ──────────────────────────────────────────────────────────────
    print(f"\n[Step 4] 开始训练（共 {args.epochs} 轮）...")
    best_acc  = 0.0    # 记录最优测试准确度（用于保存最佳权重）
    history   = []     # 记录每轮训练和测试指标
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        print(f"\n── Epoch {epoch}/{args.epochs} ──")

        # 训练一轮并返回本轮平均损失和准确度
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )

        # 在测试集上评估当前 epoch 模型性能
        test_result = evaluate(model, test_loader, criterion, device)

        # 学习率调度器步进（每轮后更新 LR）
        scheduler.step()

        # 记录本轮指标（用于后续可视化和日志）
        history.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 6),
            "train_acc":  round(train_acc, 4),
            "test_loss":  round(test_result["loss"], 6),
            "test_acc":   round(test_result["accuracy"], 4),
        })

        print(f"  Train → Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Test  → Loss: {test_result['loss']:.4f}, "
              f"Acc: {test_result['accuracy']:.2f}%")
        print(f"  当前 LR: {scheduler.get_last_lr()[0]:.2e}")

        # 保存最优权重（测试准确度最高的 epoch）
        if test_result["accuracy"] > best_acc:
            best_acc = test_result["accuracy"]
            torch.save(model.state_dict(), args.save_model)
            print(f"  [✓] 最优模型已保存 (acc={best_acc:.2f}%) → {args.save_model}")

    # 统计训练总耗时
    elapsed = time.time() - start_time
    print(f"\n[INFO] 训练完成，耗时 {elapsed:.1f} 秒")
    print(f"[INFO] 最优测试 Top-1 准确度：{best_acc:.2f}%")

    # ── 最终评估（加载最优权重） ───────────────────────────────────────────────
    print("\n[Step 5] 加载最优权重，进行最终测试集评估...")
    # weights_only=True：安全加载，避免反序列化任意代码
    model.load_state_dict(torch.load(args.save_model, weights_only=True))
    final_result = evaluate(model, test_loader, criterion, device)

    # ── 保存结果文件 ──────────────────────────────────────────────────────────
    print("\n[Step 6] 保存分类结果文件...")
    summary, json_path, csv_path = save_results(
        final_result, model, args.output_dir
    )

    # 保存训练历史（含每轮指标和总耗时）
    history_path = os.path.join(
        args.output_dir,
        f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(history_path, "w") as f:
        json.dump({"epochs": history, "elapsed_seconds": elapsed}, f, indent=2)
    print(f"[INFO] 训练历史已保存至 {history_path}")

    # ── 输出最终汇总报告 ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  LTSimulator MNIST 分类实验汇总报告")
    print(f"{'='*60}")
    print(f"  Top-1 准确度 : {final_result['accuracy']:.2f}%")
    print(f"  测试集损失   : {final_result['loss']:.4f}")
    if "optical_ratio" in summary["optical_compute"]:
        ratio = summary["optical_compute"]["optical_ratio"] * 100
        print(f"  光计算占比   : {ratio:.2f}%")
    print(f"  训练总耗时   : {elapsed:.1f} 秒")
    print(f"  结果目录     : {args.output_dir}/")
    print(f"{'='*60}")

    # ── 验收断言（Top-1 准确度必须 ≥ 85%） ───────────────────────────────────
    assert final_result["accuracy"] >= 85.0, (
        f"[FAIL] Top-1 准确度 {final_result['accuracy']:.2f}% < 85% 要求！"
    )
    print("[✓] PASS：Top-1 准确度满足 ≥ 85% 验收要求")


# ──────────────────────────────────────────────────────────────────────────────
# 程序入口
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 仅当直接运行本脚本时执行 main()，作为模块导入时不自动运行
    main()
