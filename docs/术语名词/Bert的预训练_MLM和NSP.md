## MLM（掩码语言模型）与 NSP（下一句预测）详解

这是BERT模型预训练的两个核心任务，共同推动了自然语言理解的突破：

---

### **MLM（Masked Language Model）**  
**目标**：让模型理解词语在上下文中的含义  
**方法**：随机遮盖输入文本中的单词（15%），让模型预测被遮盖的词  

#### 工作流程：
```mermaid
graph LR
A[原始句子] --> B[随机遮盖单词]
B --> C[输入BERT]
C --> D[预测遮盖词]
```

**关键细节**：  
- **遮盖策略**：  
  ```python
  # 15% 的token被处理
  for token in tokens:
      rand = random.random()
      if rand < 0.15:  # 15%概率
          # 80%替换为[MASK]：我[MASK]北京
          # 10%替换为随机词：我**苹果**北京
          # 10%保持不变：我**爱**北京（让模型知道要预测）
  ```
- **训练目标**：  
  ```math
  \text{Loss} = -\sum_{\text{masked } x_i} \log P(x_i | \text{context})
  ```

**实例**：  
> 输入：  
> `"我[MASK]北京，因为它有长城"`  
> 模型预测：  
> `爱（概率0.7）`, `去（概率0.2）`, `离开（概率0.1）`  

**意义**：  
- 解决传统语言模型的单向性缺陷（GPT只能从左到右）  
- 迫使模型理解双向上下文关系  
- 学会词语的语义消歧（如"苹果"指公司还是水果）  

---

### **NSP（Next Sentence Prediction）**  
**目标**：让模型理解句子间逻辑关系  
**方法**：判断两个句子是否连续  

#### 数据构建：
| 句子A | 句子B | 标签 |
|-------|-------|------|
| 北京是中国的首都 | 它是政治文化中心 | IsNext（连续） |
| 北京是中国的首都 | 企鹅生活在南极 | NotNext（不连续） |

**模型架构**：  
```python
# BERT输入格式：
[CLS] 句子A [SEP] 句子B [SEP]
# [CLS]位置的输出 → 二分类（连续/不连续）
```

**训练目标**：  
```python
loss = nn.CrossEntropyLoss(output_cls, true_label)  # 二分类交叉熵
```

**实例**：  
> 输入：  
> `[CLS] 水在0°C会结冰 [SEP] 这是物理现象 [SEP]`  
> 模型预测：  
> `IsNext（概率0.95）`  

> 输入：  
> `[CLS] 水在0°C会结冰 [SEP] 熊猫吃竹子 [SEP]`  
> 模型预测：  
> `NotNext（概率0.99）`

**意义**：  
- 捕捉句子间逻辑（因果、转折等）  
- 提升文档级任务表现（如问答、摘要）  
- 增强对话连贯性理解  

---

### **MLM vs NSP 对比**  
| **特性** | **MLM** | **NSP** |
|----------|---------|---------|
| **任务类型** | 词语级预测 | 句子级分类 |
| **输入处理** | 单句/双句 | 必须双句 |
| **损失函数** | 交叉熵（词汇表大小） | 二分类交叉熵 |
| **信息粒度** | 微观语义 | 宏观逻辑 |
| **贡献度** | 核心（占效果80%） | 辅助（占效果20%） |

---

### **联合训练的价值**  
1. **多任务学习优势**：  
   - MLM提供词语理解能力  
   - NSP提供篇章结构理解  
   ```python
   total_loss = mlm_loss + nsp_loss  # 联合优化
   ```
   
2. **BERT输入格式**：  
   ```
   [CLS] 句子A [SEP] 句子B [SEP]
   ↑      ↑          ↑
   NSP   MLM区域    MLM区域
   ```

3. **实际效果**：  
   - 仅用MLM：GLUE得分85.4  
   - MLM+NSP：GLUE得分88.9 → **提升4%**  
   - 在阅读理解任务上提升更显著（SQuAD F1 +7.2）

---

### **演进与替代方案**  
1. **RoBERTa的改进**：  
   - 移除NSP → 发现仅MLM效果更好  
   - 扩大训练数据和步数  

2. **ELECTRA的创新**：  
   - 用生成器-判别器替代MLM  
   - 判断每个token是否被替换  
   ```python
   # 输入：我[替换]北京 → 预测：False（被篡改）
   ```

3. **XLNet的突破**：  
   - 排列语言模型（解决[MASK]偏差）  
   - 保留双向上下文优势  

---

### **实践应用**  
#### 1. 文本相似度计算
```python
# 使用NSP能力
from transformers import BertModel
model = BertModel.from_pretrained('bert-base-chinese')

# 计算句子相似度
sentence1 = "气候变化的影响"
sentence2 = "全球变暖的后果"
inputs = tokenizer(sentence1, sentence2, return_tensors='pt')
outputs = model(**inputs)
similarity = outputs.pooler_output  # [CLS]向量
```

#### 2. 缺失信息恢复（MLM应用）
```python
def fill_mask(text):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs).logits
    masked_index = torch.where(inputs.input_ids[0] == tokenizer.mask_token_id)[0]
    predicted_token = tokenizer.decode(outputs[0, masked_index].argmax())
    return text.replace("[MASK]", predicted_token)

fill_mask("北京是中国的[MASK]")  # 输出："北京是中国的首都"
```

---

### **历史意义**  
1. **里程碑贡献**：  
   - BERT通过MLM+NSP在11项NLP任务刷新记录  
   - 开启预训练-微调范式新时代  

2. **缺陷与启示**：  
   - NSP后被证明效率不高（RoBERTa取消）  
   - [MASK]导致预训练-微调不一致（XLNet改进）  

> "MLM是BERT的灵魂，它让模型真正学会在上下文中思考词语"  
> —— Jacob Devlin（BERT第一作者）  

这些任务虽然后续有改进，但作为预训练范式的开创者，MLM和NSP仍是理解现代语言模型的基石。