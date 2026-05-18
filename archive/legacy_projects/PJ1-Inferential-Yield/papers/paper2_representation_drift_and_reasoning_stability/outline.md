建议完成：

- 3.2.1 representation drift
- 3.2.2 layerwise invariance
- 1.8 reasoning path stability
- 1.9 representation invariance

最好附带：

- 3.2.3 intervention response（初版）

---

# 论文定位

## “Mechanistic Stability Paper”

---

# 核心 claim

> unstable reasoning  
> 对应：  
> unstable latent representations。

---

# 推荐标题

### Representation Drift and Reasoning Stability in Language Models

或：

### Toward Representation-Level Analysis of Stable Inference

---

# Draft Framework

---

## 1. Introduction

behavioral robustness 不足以解释 reasoning。

需要：  
representation-level analysis。

---

## 2. Stable Representation Hypothesis

假设：

stable reasoning  
↔  
stable latent representations。

---

## 3. Representation Drift Analysis

- hidden state extraction
- cosine similarity
- perturbation drift

---

## 4. Layerwise Invariance

分析：

- 哪些 layers stable
- invariance emergence

---

## 5. Reasoning Path Stability

不同 reasoning paths：  
是否 converge to similar latent regions。

---

## 6. Preliminary Intervention Analysis

activation perturbation：  
如何影响 reasoning。

---

## 7. Discussion

从：  
behavioral robustness

走向：  
mechanistic stability。