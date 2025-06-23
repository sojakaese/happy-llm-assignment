## "归一化"**Normalization**

---

### 一、核心术语
| 中文       | 英语               | 定义                                                                 |
|------------|--------------------|----------------------------------------------------------------------|
| **归一化** | **Normalization**  | 将数据按比例缩放至特定范围（如 [0,1] 或 [-1,1]）的通用过程           |

---

### 二、常见场景及具体表达
#### 1. **数据预处理**
- **Min-Max Normalization**  
  \( X_{\text{norm}} = \frac{X - X_{\min}}{X_{\max} - X_{\min}} \)  
  *将数据线性映射到 [0,1] 区间*

- **Standardization (Z-score Normalization)**  
  \( X_{\text{std}} = \frac{X - \mu}{\sigma} \)  
  *将数据转换为均值为0、标准差为1的分布*

#### 2. **深度学习**
- **Batch Normalization (BN)**  
  *对神经网络每批输入进行归一化，加速训练*  
  `torch.nn.BatchNorm1d/2d/3d` in PyTorch

- **Layer Normalization (LN)**  
  *对单个样本所有特征进行归一化（Transformer 常用）*  
  `torch.nn.LayerNorm`

#### 3. **注意力机制**（您关注的上下文）
```python
# Transformer 中的归一化：
p_attn = scores.softmax(dim=-1)  # 英语：Apply softmax normalization
```
- **Softmax Normalization**  
  将向量转换为概率分布（和为1）：  
  \( \text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}} \)

---

### 三、技术文档中的典型表述
在您提供的 Transformer 代码中：
```python
# 原始代码：
p_attn = scores.softmax(dim=-1)

# 英语描述：
"The attention scores are normalized using softmax along the last dimension"
```
- **关键动词**：  
  - `normalize`（归一化）  
  - `scale`（缩放，如除以 \(\sqrt{d_k}\))  

---

### 四、易混淆概念对比
| 术语              | 英语                     | 目的                          | 示例场景               |
|-------------------|--------------------------|-----------------------------|------------------------|
| **归一化**        | Normalization            | 改变数据分布范围              | Min-Max 缩放           |
| **标准化**        | Standardization          | 使数据符合标准正态分布        | Z-score 变换           |
| **正则化**        | Regularization           | 防止模型过拟合                | L1/L2 正则化           |
| **激活函数**      | Activation Function      | 引入非线性                    | Softmax/ReLU           |

---

### 五、在 Transformer 中的具体应用
在注意力计算中：
```python
scores = torch.matmul(Q, K.T) / math.sqrt(d_k)  # Step 1: 缩放（Scaling）
p_attn = scores.softmax(dim=-1)                 # Step 2: 归一化（Normalization）
```
1. **Scaling**：通过除以 \(\sqrt{d_k}\) 控制数值范围  
2. **Normalization**：通过 softmax 将分数转换为概率分布  

这个过程在论文中描述为：  
> "Scaled Dot-Product Attention with softmax normalization"

---

### 总结
- **通用术语**：**Normalization**  
- **Transformer 场景**：  
  - 缩放：**Scaling** (除以 \(\sqrt{d_k}\))  
  - 概率归一化：**Softmax Normalization**  
- **代码中的体现**：  
  `scores.softmax(dim=-1)` 就是归一化的具体实现