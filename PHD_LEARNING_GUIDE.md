# PhD 申请准备：数学与编程学习指南

**目标**：为 2026 年 11 月 IMPRS-IS/CLS/MCML 申请做准备
**总投入**：约 180 小时 | 每周 8-10 小时 | 6 个月

**你的学习档案**：
- 数学基础：概念记得，细节不清楚 (CCC)
- 编程水平：Python 会用，NumPy 基础，PyTorch 零基础 (BAA)
- 学习风格：混合（视频+书+练习）
- 时间投入：每天 3-5 小时
- 目标：能独立做研究项目

---

## 📐 第一部分：数学基础

### **第一阶段：线性代数深化（3周，30小时）**

**目标**：从"知道概念"→ "能用数学语言推导"

#### **1.1 | 向量和矩阵基础**

**学习资源选择（按顺序）**：

1. **视频讲解**（1 周）- 3Blue1Brown "线性代数本质"
   - 链接：https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab
   - 第 1-3 集：向量、矩阵乘法、3D 变换
   - 第 4-5 集：矩阵乘法再看一遍
   - ⏱️ 投入：8 小时
   - 📌 关键：理解"矩阵是线性变换"的概念

2. **教科书补充**（3-4 天）- MIT 18.06 Gilbert Strang
   - 链接：https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/
   - **Lecture 1-3**：向量、矩阵、消元法
   - **Lecture 4-5**：LU 分解、置换矩阵
   - 📚 读：讲义 Sections 1.1-1.3（2-3 小时阅读）
   - 📌 这里学 Gaussian elimination 的细节

3. **练习**（2 天）
   - MIT 18.06 Problem Set 1-2
   - 链接：https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/lecture-notes-and-readings/
   - 做 10 道基础题
   - ⏱️ 投入：4 小时

**测试题**：
```
给定矩阵 A = [[1, 2], [3, 4]]
1. 计算 A 的行列式
2. 计算 A 的逆矩阵
3. 你为什么这样计算？每一步是什么意思？
```

---

#### **1.2 | 特征值和特征向量**

**学习资源选择**：

1. **视频讲解**（2 天）- 3Blue1Brown
   - 第 13-14 集：特征值和特征向量
   - 链接同上
   - ⏱️ 投入：3 小时
   - 📌 关键：理解"特征向量是不变方向"

2. **MIT 课程**（2 天）
   - **Lecture 21-23**：特征值、对角化、应用
   - 讲义 Sections 5.1-5.3
   - 📚 重点读：为什么对角化很重要？
   - ⏱️ 投入：3 小时

3. **动手计算**（1 天）
   - MIT 18.06 Problem Set 7-8
   - 手算 3-4 个 2x2 矩阵的特征值
   - ⏱️ 投入：2 小时

**测试题**：
```
矩阵 A = [[2, 1], [1, 2]]
1. 求特征值和特征向量
2. 计算 A^10 的结果（提示：用对角化）
3. 在你的 Paper 1 实验中，哪里用到特征值？
```

---

#### **1.3 | 矩阵分解（SVD, QR）**

**学习资源选择**：

1. **视频讲解**（1 天）- 3Blue1Brown
   - 第 15-16 集：SVD 的直观理解
   - ⏱️ 投入：2 小时
   - 📌 关键："SVD 把任何矩阵分解成三个简单变换"

2. **MIT 课程**（2 天）
   - **Lecture 29-31**：SVD、应用、数值方法
   - 讲义 Sections 6.3-6.5
   - 📚 重点读：SVD 的几何意义
   - ⏱️ 投入：4 小时

3. **实践操作**（2 天）
   ```python
   import numpy as np
   
   # 理解 SVD
   A = np.array([[1, 0, 0], [0, 2, 0]])
   U, S, Vh = np.linalg.svd(A)
   # 问题 1：U 是什么？S 是什么？Vh 是什么？
   # 问题 2：A = U @ np.diag(S) @ Vh，验证这个等式
   # 问题 3：如果 S = [3, 2, 0]，为什么最后一个是 0？
   
   # 应用：图片压缩
   # 加载一张灰度图片，用 SVD 压缩，看效果
   ```
   - ⏱️ 投入：3 小时

**测试题**：
```
1. 手工计算一个 2x2 矩阵的 SVD
2. 用 numpy.linalg.svd 验证你的手算结果
3. 在 PCA 中，为什么用 SVD 而不是特征值分解？
```

---

#### **1.4 | 范数、向量空间、正交性**

**学习资源选择**：

1. **MIT 课程**（2 天）
   - **Lecture 5-6**：正交向量、正交矩阵
   - **Lecture 15-16**：投影、最小二乘法
   - 讲义 Sections 3.1-3.5
   - ⏱️ 投入：4 小时
   - 📌 关键：正交性为什么重要？投影的几何意义

2. **视频补充**（1 天）
   - 3Blue1Brown 第 7 集：点积和二次性
   - ⏱️ 投入：2 小时

3. **实践**（1 天）
   ```python
   import numpy as np
   
   # 理解范数
   x = np.array([3, 4])
   L2_norm = np.linalg.norm(x)  # 应该是 5
   
   # 理解投影
   u = np.array([1, 0])
   v = np.array([3, 4])
   proj_u_v = np.dot(v, u) / np.dot(u, u) * u
   # 这是 v 在 u 上的投影
   
   # 理解正交
   a = np.array([1, 0])
   b = np.array([0, 1])
   # a 和 b 正交吗？为什么？
   ```
   - ⏱️ 投入：2 小时

**测试题**：
```
1. 定义 L1、L2、L-infinity 范数
2. 为什么深度学习中常用 L2 范数？
3. 计算向量 [1,2,3] 在 [1,0,0] 上的投影
```

---

### **第二阶段：概率论与统计（2周，20小时）**

**目标**：从"知道公式"→ "能用概率语言思考"

#### **2.1 | 概率基础**

**学习资源选择**：

1. **视频讲解**（2 天）- StatQuest with Josh Starmer
   - 频道：https://www.youtube.com/@statquest
   - 播放列表："Probability"
   - 视频 1-5：概率公理、条件概率、独立性
   - ⏱️ 投入：4 小时
   - 📌 风格：非常直观，用现实例子

2. **教科书**（2 天）- MIT 6.041
   - 链接：https://ocw.mit.edu/courses/6-041sc-probabilistic-systems-analysis-and-applied-probability-fall-2013/
   - **Lecture 1-4**：样本空间、概率、条件概率
   - 讲义 Sections 1.1-1.3
   - ⏱️ 投入：4 小时

3. **练习**（1 天）
   ```python
   # 概率计算练习
   # P(A|B) = P(A,B) / P(B)
   # 练习计算：给定条件概率，反推联合概率
   ```
   - ⏱️ 投入：2 小时

**测试题**：
```
1. 100 个人中 10 个患病，检测准确率 99%。
   某人检测呈阳性，他真的患病的概率是多少？
   （提示：用贝叶斯定理，需要考虑 base rate）

2. 抛两个骰子，P(和为 7 | 至少一个 6) = ?
```

---

#### **2.2 | 贝叶斯定理和推断**

**学习资源选择**：

1. **视频讲解**（1 天）- StatQuest
   - "Bayes' Theorem" 系列（3 个视频）
   - ⏱️ 投入：2 小时
   - 📌 这是最关键的一个概念

2. **3Blue1Brown 视角**（1 天）
   - 《概率的本质》播放列表
   - "Bayes theorem, the geometry of changing beliefs"
   - ⏱️ 投入：1.5 小时

3. **深度阅读**（1 天）- MIT 6.041
   - **Lecture 5-7**：贝叶斯推断
   - ⏱️ 投入：2 小时

4. **实践**（1 天）
   ```python
   # 实现贝叶斯更新
   # 初始：某病患率 P(disease) = 0.01
   # 测试阳性：P(positive|disease) = 0.95, P(positive|no disease) = 0.05
   # 求：P(disease|positive) = ?
   ```
   - ⏱️ 投入：1.5 小时

**测试题**：
```
在你的因果推理中，干预（do-calculus）和条件概率的区别是什么？
为什么这个区别很重要？
```

---

#### **2.3 | 分布和期望**

**学习资源选择**：

1. **视频讲解**（2 天）- StatQuest
   - "Distributions" 系列
   - 高斯分布、伯努利分布、多项分布
   - ⏱️ 投入：3 小时

2. **MIT 课程**（1 天）
   - **Lecture 8-10**：离散分布、连续分布、期望
   - ⏱️ 投入：2 小时

3. **实践**（1 天）
   ```python
   import numpy as np
   from scipy import stats
   
   # 理解正态分布
   data = np.random.normal(loc=0, scale=1, size=1000)
   mu = np.mean(data)
   sigma = np.std(data)
   # 约 68% 的数据在 [mu-sigma, mu+sigma] 中吗？
   
   # 理解期望和方差
   E_X = np.mean(data)  # 期望
   Var_X = np.var(data)  # 方差
   ```
   - ⏱️ 投入：2 小时

**测试题**：
```
1. X ~ N(0, 1)（标准正态分布）
   P(X > 1.96) = ?
   
2. 为什么交叉熵损失函数用负对数概率？
   (Hint: 从信息论的角度思考)
```

---

#### **2.4 | 极大似然估计（MLE）**

**学习资源选择**：

1. **视频讲解**（1 天）- StatQuest
   - "Maximum Likelihood Estimation"
   - ⏱️ 投入：2 小时
   - 📌 这是机器学习最基本的概念

2. **MIT 课程**（1 天）
   - **Lecture 14-15**：参数估计、MLE
   - ⏱️ 投入：2 小时

3. **实践**（1 天）
   ```python
   # 实现 MLE
   # 给定数据，估计高斯分布的参数
   import numpy as np
   from scipy.stats import norm
   
   # 真实数据来自 N(mu=2, sigma=1)
   data = np.random.normal(2, 1, 100)
   
   # MLE 估计 mu 和 sigma
   mu_mle = np.mean(data)
   sigma_mle = np.std(data)
   
   # 验证这就是 MLE（对比优化）
   ```
   - ⏱️ 投入：2 小时

**测试题**：
```
1. 逻辑回归的损失函数为什么是交叉熵？
   （提示：推导 MLE 目标）

2. 在你的 Paper 1 中，哪里用到了概率？
```

---

### **第三阶段：微积分与优化（2周，20小时）**

**目标**：理解梯度下降的工作原理

#### **3.1 | 偏导数和梯度**

**学习资源选择**：

1. **视频讲解**（1 天）- 3Blue1Brown
   - "Derivatives" 播放列表第 10-12 集
   - 多变量微积分、偏导数、梯度
   - ⏱️ 投入：2 小时

2. **Khan Academy**（1 天）
   - 链接：https://www.khanacademy.org/math/multivariable-calculus
   - "Partial derivatives" 单元
   - ⏱️ 投入：2 小时

3. **实践**（1 天）
   ```python
   import numpy as np
   
   # 理解梯度
   def f(x, y):
       return x**2 + 2*y**2
   
   # 在点 (1, 2) 的梯度
   # df/dx = 2x = 2
   # df/dy = 4y = 8
   # 梯度 = [2, 8]，指向最陡上升方向
   
   # 数值梯度计算（验证）
   eps = 1e-5
   grad_x = (f(1+eps, 2) - f(1, 2)) / eps
   grad_y = (f(1, 2+eps) - f(1, 2)) / eps
   ```
   - ⏱️ 投入：2 小时

**测试题**：
```
1. 给定 f(x,y) = x^2 + xy + y^2，求 ∇f
2. 在点 (1, -1) 的梯度是什么？
3. 梯度的方向是什么意思（为什么是"最陡上升"方向）？
```

---

#### **3.2 | 链式法则和反向传播**

**学习资源选择**：

1. **视频讲解**（1 天）- 3Blue1Brown
   - "Chain Rule" 第 3 集
   - ⏱️ 投入：1.5 小时
   - 📌 超级重要，看 3 遍都不过分

2. **深度解释**（1 天）- Jeremy Jordan 的博客
   - 链接：https://www.jeremyjordan.me/neural-networks-training/
   - 讲解：如何用链式法则计算神经网络的梯度
   - ⏱️ 投入：2 小时

3. **手工计算**（2 天）
   ```python
   # 手工计算反向传播
   # y = sigmoid(w*x + b)
   # L = (y - target)^2
   
   # 前向传播：
   x = 2
   w = 0.5
   b = 0.1
   z = w*x + b  # = 1.1
   y = 1 / (1 + np.exp(-z))  # sigmoid
   target = 1
   L = (y - target)**2
   
   # 反向传播（手工计算）：
   # dL/dy = 2(y - target)
   # dy/dz = y(1-y)  （sigmoid 导数）
   # dz/dw = x
   # dz/db = 1
   # 用链式法则：dL/dw = dL/dy * dy/dz * dz/dw
   ```
   - ⏱️ 投入：3 小时

**测试题**：
```
1. 给定：y = relu(w*x + b)，L = (y - target)^2
   用链式法则计算 dL/dw 和 dL/db

2. 为什么需要计算这些梯度？
   （提示：梯度下降）

3. 在 PyTorch 中，loss.backward() 做的是什么？
```

---

#### **3.3 | 梯度下降和优化**

**学习资源选择**：

1. **视频讲解**（1 天）- StatQuest
   - "Gradient Descent" 系列（3 个视频）
   - ⏱️ 投入：2 小时

2. **深度解释**（1 天）- Sebastian Ruder 的博客
   - 链接：https://ruder.io/optimizers/
   - 讲解：各种优化算法（GD, SGD, Adam 等）
   - ⏱️ 投入：2 小时

3. **实践**（2 天）
   ```python
   import numpy as np
   
   # 从零实现梯度下降
   def objective(w):
       return (w - 3)**2  # 最小值在 w=3
   
   def gradient(w):
       return 2 * (w - 3)
   
   # 梯度下降
   w = 0
   learning_rate = 0.1
   for step in range(50):
       grad = gradient(w)
       w = w - learning_rate * grad
       if step % 10 == 0:
           print(f"Step {step}: w={w:.4f}, loss={objective(w):.4f}")
   
   # 问题：
   # 1. w 最终收敛到 3 吗？
   # 2. 如果 learning_rate 太大会怎样？
   # 3. 如果 learning_rate 太小会怎样？
   ```
   - ⏱️ 投入：2 小时

**测试题**：
```
1. 什么是学习率？太大和太小各有什么问题？
2. Adam 优化器相比 SGD 有什么优势？
3. 为什么在神经网络中用 SGD 而不是完整的梯度下降？
```

---

#### **3.4 | 凸函数和凸优化**

**学习资源选择**：

1. **视频讲解**（1 天）- Boyd 的凸优化课程
   - 链接：https://www.youtube.com/playlist?list=PL3940DD956CDF0BB5
   - Lecture 1-3：凸函数、凸优化问题
   - ⏱️ 投入：2 小时

2. **书籍**（1 天）- Boyd & Vandenberghe "Convex Optimization"
   - 链接：https://web.stanford.edu/~boyd/cvxbook/
   - Chapter 2：凸集和凸函数（skim，不需要深入）
   - ⏱️ 投入：2 小时

3. **直观理解**（1 天）
   ```python
   import numpy as np
   import matplotlib.pyplot as plt
   
   # 凸函数 vs 非凸函数
   x = np.linspace(-5, 5, 100)
   
   # 凸函数：f(x) = x^2
   y_convex = x**2
   
   # 非凸函数：f(x) = sin(x)
   y_nonconvex = np.sin(x)
   
   # 对于凸函数，梯度下降保证找到全局最小值
   # 对于非凸函数，可能陷入局部最小值
   plt.plot(x, y_convex, label='convex')
   plt.plot(x, y_nonconvex, label='non-convex')
   plt.legend()
   plt.show()
   ```
   - ⏱️ 投入：1 小时

**测试题**：
```
1. 什么是凸函数？
2. 神经网络的损失函数是凸的吗？为什么这很重要？
3. 即使损失函数不是凸的，为什么梯度下降在深度学习中还能工作？
```

---

### **第四阶段：信息论基础（1周，8小时）**

**目标**：理解为什么用交叉熵而不是其他损失函数

#### **4.1 | 熵和 KL 散度**

**学习资源选择**：

1. **视频讲解**（1 天）- StatQuest
   - "Entropy" 和 "Cross Entropy" 视频
   - ⏱️ 投入：2 小时

2. **3Blue1Brown**（半天）
   - 信息论视频（如果有的话）
   - ⏱️ 投入：1.5 小时

3. **实践**（1 天）
   ```python
   import numpy as np
   
   # 理解熵
   # 高熵 = 不确定性大
   # 低熵 = 不确定性小
   
   p1 = np.array([0.5, 0.5])  # 高熵
   p2 = np.array([0.99, 0.01])  # 低熵
   
   def entropy(p):
       return -np.sum(p * np.log(p + 1e-10))
   
   print(f"Entropy of p1: {entropy(p1):.4f}")
   print(f"Entropy of p2: {entropy(p2):.4f}")
   
   # 理解 KL 散度（相对熵）
   # KL(p || q) 衡量分布 p 和 q 的差异
   def kl_divergence(p, q):
       return np.sum(p * (np.log(p + 1e-10) - np.log(q + 1e-10)))
   
   # 交叉熵 = 熵 + KL 散度
   # cross_entropy(p, q) = entropy(p) + kl_divergence(p, q)
   ```
   - ⏱️ 投入：2 小时

4. **应用**（1 天）
   ```python
   # 为什么机器学习用交叉熵？
   # 因为最小化交叉熵 = 最小化 KL 散度（在给定真实分布时）
   # 这意味着我们在让模型输出的分布接近真实分布
   
   # 例子：分类问题
   true_label = np.array([0, 1, 0])  # 真实标签（one-hot）
   pred_prob = np.array([0.1, 0.8, 0.1])  # 模型预测的概率
   
   cross_entropy_loss = -np.sum(true_label * np.log(pred_prob + 1e-10))
   print(f"Cross entropy loss: {cross_entropy_loss:.4f}")
   ```
   - ⏱️ 投入：1.5 小时

**测试题**：
```
1. 为什么 KL(p || q) ≠ KL(q || p)？
2. 在分类问题中，为什么用交叉熵而不是 MSE？
3. 在你的 Paper 1 中，用的是什么损失函数？为什么？
```

---

### **第五阶段：因果推理入门（1周，10小时）**

**目标**：理解干预 vs 观察、因果图基础

#### **5.1 | 因果图和 DAG**

**学习资源选择**：

1. **书籍**（2 天）- Pearl "Causal Inference in Statistics: A Primer"
   - Chapter 1: 因果图的基础
   - ⏱️ 投入：4 小时
   - 📌 这本书就是为你这样的人写的，非常清晰

2. **视频补充**（1 天）- Brady Neal 的因果推理课程
   - 链接：https://www.bradyneal.com/causal-inference-course
   - Lesson 1-2：因果图、干预
   - ⏱️ 投入：2 小时

3. **实践**（1 天）
   ```
   理解因果图：
   
   例子 1：混淆变量
   X ← Z → Y  （Z 是混淆变量）
   - 观察相关性不能推断因果性
   - 需要控制 Z
   
   例子 2：中介变量
   X → Z → Y  （Z 是中介）
   - X 对 Y 有直接和间接效应
   - 需要区分两者
   
   例子 3：碰撞器
   X → Z ← Y  （Z 是碰撞器）
   - X 和 Y 独立，但条件 Z 后相关
   
   你的 Paper 1 中：
   快捷特征 S ← ？→ 标签 Y
   语义特征 C ← ？→ 标签 Y
   这是什么因果关系？
   ```
   - ⏱️ 投入：2 小时

**测试题**：
```
1. 区别：P(Y | do(X=x)) 和 P(Y | X=x)
   （干预 vs 观察）

2. 给定因果图：
   A → C, B → C
   A 和 B 独立吗？
   如果条件 C，A 和 B 还独立吗？

3. 在你的 Paper 1 中，为什么要做"反向"实验？
   这是在做什么因果操作？
```

---

#### **5.2 | 后门调整和因果效应识别**

**学习资源选择**：

1. **书籍**（1 天）- Pearl "Causal Inference"
   - Chapter 2: 识别因果效应
   - 后门准则、前门准则
   - ⏱️ 投入：2 小时

2. **论文阅读**（1 天）
   - Schölkopf 的论文（你的目标导师！）
   - 搜索："Causal Representation Learning" 或 "Toward Causal Representation Learning"
   - ⏱️ 投入：2 小时
   - 📌 这会给你方向感

**测试题**：
```
1. 什么是后门路径？为什么需要阻断它？
2. 在随机对照试验（RCT）中，为什么可以推断因果效应？
3. 在你的 Paper 1 中，你是在做隐式的 RCT 吗？
```

---

## 💻 第二部分：编程与工程

### **第一阶段：Python 和 NumPy 深化（1周，12小时）**

**目标**：能独立解决数值计算问题

#### **1.1 | Python 面向对象编程**

**学习资源选择**：

1. **视频讲解**（2 天）- Corey Schafer
   - YouTube 频道：https://www.youtube.com/@CoreySchafer
   - 播放列表：Python OOP Tutorial
   - 第 1-6 集：类、继承、魔法方法
   - ⏱️ 投入：3 小时

2. **实践**（2 天）
   ```python
   # 实现一个简单的数据集类
   class Dataset:
       def __init__(self, X, y):
           self.X = X
           self.y = y
       
       def __len__(self):
           return len(self.X)
       
       def __getitem__(self, idx):
           return self.X[idx], self.y[idx]
       
       def shuffle(self):
           indices = np.random.permutation(len(self))
           self.X = self.X[indices]
           self.y = self.y[indices]
   
   # 使用这个类
   X = np.random.randn(100, 10)
   y = np.random.randint(0, 2, 100)
   dataset = Dataset(X, y)
   print(len(dataset))
   print(dataset[0])
   ```
   - ⏱️ 投入：2 小时

**测试**：能写出一个有 `__init__`、`__len__`、`__getitem__` 的类吗？

---

#### **1.2 | NumPy 高级用法**

**学习资源选择**：

1. **官方教程**（1 天）
   - https://numpy.org/doc/stable/user/basics.broadcasting.html
   - 重点：广播机制、高级索引
   - ⏱️ 投入：2 小时

2. **实践题目**（2 天）- Numpy 100 Exercises
   - 链接：https://github.com/rougier/numpy-100
   - 做第 1-40 题（涵盖基本和进阶）
   - ⏱️ 投入：3 小时

3. **实战**（1 天）
   ```python
   # 广播演示
   a = np.array([[1, 2, 3]])  # (1, 3)
   b = np.array([[1], [2], [3]])  # (3, 1)
   c = a + b  # 广播成 (3, 3)
   
   # 高级索引
   arr = np.arange(20).reshape(4, 5)
   mask = arr > 10
   arr[mask] = 0  # 条件赋值
   
   # 实现矩阵乘法的另一种方式
   A = np.random.randn(5, 3)
   B = np.random.randn(3, 4)
   C = np.einsum('ij,jk->ik', A, B)  # 爱因斯坦求和
   ```
   - ⏱️ 投入：2 小时

**测试题**：
```python
# 能不用 for 循环实现吗？
# 计算每行与某个向量的距离
X = np.random.randn(100, 10)  # 100 个样本，10 维
v = np.random.randn(10)  # 1 个向量
distances = ?  # 应该是 (100,)
```

---

#### **1.3 | Pandas 数据处理**

**学习资源选择**：

1. **官方教程**（1 天）
   - https://pandas.pydata.org/docs/user_guide/basics.html
   - 或者看 Corey Schafer 的 Pandas 教程
   - ⏱️ 投入：2 小时

2. **实践**（2 天）
   ```python
   import pandas as pd
   
   # 读取数据
   df = pd.read_csv('data.csv')
   
   # 基本操作
   print(df.head())
   print(df.info())
   print(df.describe())
   
   # 选择列
   df['column_name']
   df[['col1', 'col2']]
   df.loc[0]  # 按标签选
   df.iloc[0]  # 按位置选
   
   # 过滤
   df[df['age'] > 30]
   
   # 分组
   df.groupby('category')['value'].mean()
   
   # 合并
   df1.merge(df2, on='key')
   
   # 清理
   df.dropna()
   df.fillna(0)
   df.drop_duplicates()
   ```
   - ⏱️ 投入：2 小时

**测试**：能从 CSV 加载数据，清理，分组统计吗？

---

### **第二阶段：PyTorch 从零开始（3周，35小时）**

**目标**：理解自动求导，能实现简单神经网络

#### **2.1 | PyTorch 基础和张量**

**学习资源选择**：

1. **官方教程**（2 天）
   - https://pytorch.org/tutorials/beginner/basics/intro.html
   - 前 3 个教程
   - ⏱️ 投入：3 小时

2. **视频讲解**（2 天）- Aladdin Persson
   - YouTube 频道：https://www.youtube.com/@AladdinPersson
   - 播放列表：PyTorch Tutorials for Deep Learning
   - 第 1-3 集：基础、张量操作
   - ⏱️ 投入：3 小时

3. **实践**（2 天）
   ```python
   import torch
   
   # 张量创建
   x = torch.randn(3, 4)
   x = torch.zeros(2, 3)
   x = torch.ones(5)
   x = torch.arange(10)
   
   # 张量操作
   x.shape, x.dtype, x.device
   x.reshape(-1, 5)
   x.T  # 转置
   
   # 数学运算
   torch.mm(A, B)  # 矩阵乘法
   torch.dot(a, b)  # 向量点积
   torch.sum(x), torch.mean(x)
   
   # GPU（如果有）
   x = x.to('cuda')
   x = x.cpu()
   
   # NumPy 互转
   x_np = x.numpy()
   x_torch = torch.from_numpy(x_np)
   ```
   - ⏱️ 投入：3 小时

**测试**：能创建、操作张量，理解形状和数据类型吗？

---

#### **2.2 | 自动求导（Autograd）**

**学习资源选择**：

1. **官方教程**（1 天）
   - https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html
   - ⏱️ 投入：2 小时

2. **深度讲解**（1 天）- Aladdin Persson
   - 第 4 集：Autograd 和 Gradient
   - ⏱️ 投入：1.5 小时

3. **动手实现**（2 天）
   ```python
   import torch
   
   # 基础：自动求导
   x = torch.tensor(2.0, requires_grad=True)
   y = x ** 3
   z = y + 2
   z.backward()
   print(x.grad)  # dz/dx = 3*x^2 = 12 在 x=2
   
   # 更复杂的例子
   x = torch.randn(3, requires_grad=True)
   y = x ** 2
   z = y.sum()
   z.backward()
   print(x.grad)  # dz/dx = 2*x
   
   # 与 NumPy 数值求导对比验证
   eps = 1e-5
   def numerical_gradient(f, x, eps=1e-5):
       grad = np.zeros_like(x)
       for i in range(len(x)):
           x_plus = x.copy()
           x_plus[i] += eps
           x_minus = x.copy()
           x_minus[i] -= eps
           grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
       return grad
   
   # 验证 autograd 是否正确
   ```
   - ⏱️ 投入：3 小时

**测试题**：
```python
# 计算梯度
x = torch.randn(2, 2, requires_grad=True)
y = (x ** 2).sum()
y.backward()
# x.grad 应该是什么？
```

---

#### **2.3 | 神经网络模块（nn.Module）**

**学习资源选择**：

1. **官方教程**（1 天）
   - https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
   - ⏱️ 投入：2 小时

2. **视频讲解**（1 天）- Aladdin Persson
   - 第 5-6 集：nn.Module, 常见层
   - ⏱️ 投入：1.5 小时

3. **实践**（2 天）
   ```python
   import torch
   import torch.nn as nn
   
   # 定义模型
   class SimpleNet(nn.Module):
       def __init__(self):
           super(SimpleNet, self).__init__()
           self.fc1 = nn.Linear(784, 128)  # 28x28=784 -> 128
           self.relu = nn.ReLU()
           self.fc2 = nn.Linear(128, 10)  # 128 -> 10 classes
       
       def forward(self, x):
           x = x.view(-1, 784)  # 展平
           x = self.fc1(x)
           x = self.relu(x)
           x = self.fc2(x)
           return x
   
   # 使用模型
   model = SimpleNet()
   x = torch.randn(32, 1, 28, 28)  # 32 个 28x28 的图片
   output = model(x)  # (32, 10)
   
   # 查看参数
   for name, param in model.named_parameters():
       print(f"{name}: {param.shape}")
   ```
   - ⏱️ 投入：2 小时

**测试**：能定义一个 3 层的全连接网络吗？

---

#### **2.4 | 优化器和训练循环**

**学习资源选择**：

1. **官方教程**（1 天）
   - https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
   - ⏱️ 投入：2 小时

2. **视频讲解**（1 天）- Aladdin Persson
   - 第 7 集：Loss Functions 和 Optimizers
   - ⏱️ 投入：1.5 小时

3. **实现一个完整的训练循环**（3 天）
   ```python
   import torch
   import torch.nn as nn
   import torch.optim as optim
   
   # 模型、损失函数、优化器
   model = SimpleNet()
   criterion = nn.CrossEntropyLoss()
   optimizer = optim.Adam(model.parameters(), lr=0.001)
   
   # 虚拟数据
   X_train = torch.randn(100, 1, 28, 28)
   y_train = torch.randint(0, 10, (100,))
   
   # 训练循环
   num_epochs = 5
   batch_size = 32
   
   for epoch in range(num_epochs):
       total_loss = 0
       for i in range(0, len(X_train), batch_size):
           # 前向传播
           batch_x = X_train[i:i+batch_size]
           batch_y = y_train[i:i+batch_size]
           
           output = model(batch_x)
           loss = criterion(output, batch_y)
           
           # 反向传播
           optimizer.zero_grad()  # 清除旧梯度
           loss.backward()  # 计算新梯度
           optimizer.step()  # 更新参数
           
           total_loss += loss.item()
       
       avg_loss = total_loss / (len(X_train) // batch_size)
       print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
   ```
   - ⏱️ 投入：3 小时

**测试**：能从零实现训练循环，包括数据加载、前向/反向传播吗？

---

#### **2.5 | 使用预训练模型**

**学习资源选择**：

1. **官方教程**（1 天）
   - https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
   - ⏱️ 投入：2 小时

2. **实践**（1 天）
   ```python
   import torch
   import torchvision.models as models
   
   # 加载预训练模型
   resnet = models.resnet18(pretrained=True)
   
   # 修改最后一层用于自己的任务
   num_classes = 5
   resnet.fc = torch.nn.Linear(512, num_classes)
   
   # Fine-tuning
   optimizer = torch.optim.Adam(resnet.parameters(), lr=0.001)
   # ... 训练循环
   ```
   - ⏱️ 投入：1.5 小时

---

### **第三阶段：计算机视觉和 NLP 基础（2周，15小时）**

**目标**：理解 CNN、Transformer 基础

#### **3.1 | 卷积神经网络（CNN）**

**学习资源选择**：

1. **视频讲解**（1 天）- StatQuest
   - "Convolutional Neural Networks" 系列
   - ⏱️ 投入：2 小时

2. **深度讲解**（1 天）- Aladdin Persson
   - 第 8-10 集：CNN 架构
   - ⏱️ 投入：2 小时

3. **实践**（2 天）
   ```python
   import torch
   import torch.nn as nn
   
   # 定义 CNN
   class SimpleCNN(nn.Module):
       def __init__(self):
           super(SimpleCNN, self).__init__()
           # 卷积层
           self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
           self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
           # 池化层
           self.pool = nn.MaxPool2d(2, 2)
           # 全连接层
           self.fc1 = nn.Linear(64 * 7 * 7, 128)
           self.fc2 = nn.Linear(128, 10)
       
       def forward(self, x):
           x = self.pool(torch.relu(self.conv1(x)))
           x = self.pool(torch.relu(self.conv2(x)))
           x = x.view(-1, 64 * 7 * 7)  # 展平
           x = torch.relu(self.fc1(x))
           x = self.fc2(x)
           return x
   
   # 测试
   model = SimpleCNN()
   x = torch.randn(32, 1, 28, 28)  # 32 个灰度图
   output = model(x)  # (32, 10)
   ```
   - ⏱️ 投入：2 小时

---

#### **3.2 | Transformer 和 Attention（对你特别重要！）**

**学习资源选择**：

1. **视频讲解**（2 天）- StatQuest 或 3Blue1Brown
   - "Attention is All You Need" 讲解视频
   - ⏱️ 投入：2 小时

2. **深度讲解**（2 天）- Aladdin Persson
   - Transformer 系列视频
   - ⏱️ 投入：2 小时

3. **博客**（1 天）- Jay Alammar
   - 链接：http://jalammar.github.io/illustrated-transformer/
   - 超级清楚的图解说明
   - ⏱️ 投入：2 小时

4. **实践**（2 天）
   ```python
   import torch
   import torch.nn as nn
   
   # 简化版 Attention
   class Attention(nn.Module):
       def __init__(self, d_model):
           super().__init__()
           self.d_k = d_model ** 0.5
       
       def forward(self, Q, K, V):
           # Q: (batch, seq_len, d_model)
           # K: (batch, seq_len, d_model)
           # V: (batch, seq_len, d_model)
           
           # 计算注意力权重
           scores = torch.matmul(Q, K.transpose(-2, -1)) / self.d_k
           weights = torch.softmax(scores, dim=-1)
           
           # 加权求和
           output = torch.matmul(weights, V)
           return output
   
   # 测试
   d_model = 64
   batch_size = 32
   seq_len = 10
   
   Q = torch.randn(batch_size, seq_len, d_model)
   K = torch.randn(batch_size, seq_len, d_model)
   V = torch.randn(batch_size, seq_len, d_model)
   
   attention = Attention(d_model)
   output = attention(Q, K, V)
   print(output.shape)  # (32, 10, 64)
   ```
   - ⏱️ 投入：2 小时

**测试题**：
```
1. Attention 机制的三个输入（Q, K, V）分别是什么？
2. 为什么要除以 √d_k？
3. 在 BERT 中，为什么使用 Multi-Head Attention？
```

---

#### **3.3 | NLP 和 Transformers 库**

**学习资源选择**：

1. **官方教程**（1 天）
   - https://huggingface.co/docs/transformers/quicktour
   - ⏱️ 投入：2 小时

2. **实践**（1 天）
   ```python
   from transformers import AutoTokenizer, AutoModel
   import torch
   
   # 加载预训练模型
   model_name = "bert-base-uncased"
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   model = AutoModel.from_pretrained(model_name)
   
   # 文本编码
   text = "Hello, this is a test sentence."
   inputs = tokenizer(text, return_tensors="pt")
   # inputs: {'input_ids': [...], 'attention_mask': [...], ...}
   
   # 前向传播
   with torch.no_grad():
       outputs = model(**inputs)
   
   # 获取最后一层隐藏状态
   last_hidden_state = outputs.last_hidden_state
   print(last_hidden_state.shape)
   
   # 获取 [CLS] 令牌的表示（用于分类）
   cls_representation = last_hidden_state[:, 0, :]
   ```
   - ⏱️ 投入：1.5 小时

---

### **第四阶段：调试和最佳实践（1周，8小时）**

**目标**：能独立解决编程问题

#### **4.1 | 调试技巧**

**学习资源选择**：

1. **视频讲解**（1 天）
   - Python 和 PyTorch 调试
   - ⏱️ 投入：1 小时

2. **实践**（1 天）
   ```python
   # 使用 print debugging
   def forward(self, x):
       print(f"Input shape: {x.shape}")
       x = self.conv1(x)
       print(f"After conv1: {x.shape}")
       x = self.pool(x)
       print(f"After pool: {x.shape}")
       return x
   
   # 使用 PyTorch 调试器（更高级）
   import pdb
   
   def forward(self, x):
       pdb.set_trace()  # 代码在这里停止
       x = self.layer1(x)
       return x
   
   # 使用 assert 检查
   assert x.shape == (batch_size, seq_len, d_model), f"Wrong shape: {x.shape}"
   ```
   - ⏱️ 投入：1 小时

---

#### **4.2 | 代码组织和最佳实践**

**学习资源选择**：

1. **阅读**（1 天）
   - "Clean Code" 的 Python 版本概念
   - PEP 8（Python 代码风格指南）
   - ⏱️ 投入：1 小时

2. **实践**（1 天）
   ```python
   # 好的代码结构
   class DataModule:
       def __init__(self, data_path, batch_size):
           self.data_path = data_path
           self.batch_size = batch_size
       
       def load_data(self):
           # ...
           return train_loader, val_loader, test_loader
   
   class TrainingModule:
       def __init__(self, model, device):
           self.model = model
           self.device = device
       
       def train_epoch(self, train_loader, optimizer, criterion):
           # ...
           return avg_loss
       
       def validate(self, val_loader, criterion):
           # ...
           return avg_acc
   
   # 主脚本
   if __name__ == "__main__":
       # 配置
       config = {
           'data_path': './data',
           'batch_size': 32,
           'lr': 0.001,
           'epochs': 10
       }
       
       # 初始化
       data = DataModule(config['data_path'], config['batch_size'])
       train_loader, val_loader, test_loader = data.load_data()
       
       model = SimpleNet()
       trainer = TrainingModule(model, 'cuda')
       
       # 训练
       for epoch in range(config['epochs']):
           train_loss = trainer.train_epoch(train_loader, optimizer, criterion)
           val_acc = trainer.validate(val_loader, criterion)
   ```
   - ⏱️ 投入：2 小时

---

#### **4.3 | 复现论文代码**

**学习资源选择**：

1. **从你的 Paper 1 开始**（2 天）
   - 打开 PJ1 和 PJ3 的代码
   - 理解每一行
   - 如果有不明白的，查文档、Google、或问 ChatGPT
   - ⏱️ 投入：3 小时

2. **尝试复现其他论文**（1 天）
   - Papers with Code（https://paperswithcode.com/）
   - 找一篇简单的论文，看他们提供的代码
   - 尝试自己重写一遍
   - ⏱️ 投入：2 小时

---

## 📊 6 个月学习时间表

### **5月（现在）- 6月：数学基础和编程基础**
```
周 1-2: 线性代数深化 (30h)
- 视频 + 教科书 + 练习题
- 每周 15 小时

周 3-4: NumPy 和 Python OOP (12h)
- 官方教程 + 练习
- 每周 6 小时

目标：完成 Paper 1 投稿
```

**总计：6月完成 42 小时**

---

### **7月：概率论和微积分**
```
周 1-2: 概率论 (20h)
- 贝叶斯定理、分布、MLE
- 每周 10 小时

周 3-4: 微积分和优化 (20h)
- 梯度、链式法则、反向传播
- 每周 10 小时

目标：能推导反向传播算法
```

**总计：7月完成 40 小时**

---

### **8月：PyTorch 深化**
```
周 1-2: PyTorch 基础 (15h)
- 张量、autograd、nn.Module
- 每周 7.5 小时

周 3-4: 训练循环和模型 (12h)
- 完整的训练脚本
- 在 Paper 1 数据上尝试简单模型
- 每周 6 小时

目标：能从零实现简单神经网络
```

**总计：8月完成 27 小时**

---

### **9月：高级主题和复现**
```
周 1-2: CNN 和 Transformer (15h)
- 理解 attention 机制
- 每周 7.5 小时

周 3-4: NLP 和 Transformers 库 (10h)
- 加载预训练模型
- 每周 5 小时

目标：能用 BERT 做简单 NLP 任务
```

**总计：9月完成 25 小时**

---

### **10月-11月：信息论、因果推理、复现论文**
```
周 1-2: 信息论和因果推理 (18h)
- 熵、KL 散度、因果图
- 每周 9 小时

周 3-4: 复现论文、调试、最佳实践 (15h)
- 理解 Paper 1 的每一行代码
- 尝试修改和改进
- 准备面试
- 每周 7.5 小时

目标：能自信地讲解和修改你的研究代码
```

**总计：10-11月完成 33 小时**

---

**6月总投入时间**：
- 6月：42h
- 7月：40h
- 8月：27h
- 9月：25h
- 10月-11月：33h
- **总计：167 小时** ≈ **每周 6-8 小时**

---

## 🎯 学习验证检查表

### **6月底**
- [ ] 能推导 SVD
- [ ] 能解释特征值的含义
- [ ] 能用 NumPy 实现 PCA
- [ ] Paper 1 已投稿

### **7月底**
- [ ] 能用贝叶斯定理解决实际问题
- [ ] 能手工计算简单网络的反向传播
- [ ] 理解为什么用交叉熵损失

### **8月底**
- [ ] 能用 PyTorch 从零实现 MLP
- [ ] 理解自动求导的工作原理
- [ ] 能在你的 Paper 1 数据上训练模型

### **9月底**
- [ ] 理解 Attention 机制的数学
- [ ] 能用 Transformers 库加载 BERT
- [ ] 能理解 Transformer 的代码实现

### **10月底**
- [ ] 理解因果推理的基本概念
- [ ] 能完全讲解 Paper 1 的每一行代码
- [ ] 能修改代码并运行新实验

### **11月初（申请前）**
- [ ] 所有材料准备完成
- [ ] 能自信地回答技术面试题
- [ ] 能讲解你的研究想法和方法

---

## 📚 快速参考表

| 主题 | 时间 | 资源 | 优先级 |
|------|------|------|--------|
| 线性代数 | 30h | 3B1B + MIT | 🔴 必须 |
| 概率论 | 20h | StatQuest + MIT | 🔴 必须 |
| 微积分优化 | 20h | 3B1B + MIT | 🔴 必须 |
| NumPy | 12h | 官方 + 练习 | 🟠 强烈 |
| PyTorch | 27h | 官方 + Aladdin | 🟠 强烈 |
| 信息论 | 8h | StatQuest + 书 | 🟡 重要 |
| 因果推理 | 10h | Pearl 书 + Brady | 🟡 重要 |
| CNN/Transformer | 25h | 官方 + Aladdin | 🟡 有帮助 |

---

## 💡 学习建议

1. **按照时间表走，不要都学完才开始用**
   - 学完线性代数就开始 NumPy
   - 学完反向传播就开始 PyTorch

2. **混合学习方式**
   - 看视频（理解直观）
   - 读书籍（理解细节）
   - 做练习（记住并应用）

3. **定期回顾**
   - 每周末总结学过的内容
   - 每月做一个综合项目

4. **别完美主义**
   - 不需要 100% 理解所有东西
   - 70% 理解 + 能应用 > 90% 理解但没用过

5. **遇到卡住的地方**
   - Google 你的错误信息
   - 在 Stack Overflow 上搜索
   - 问 ChatGPT 或 Claude
   - 在你的研究 Discord/社区提问

---

## 🚀 开始行动

**这周立即做的事**：
1. 看 3Blue1Brown "线性代数本质" 第 1-3 集（3 小时）
2. 做 MIT 18.06 第 1-2 讲的练习题（2 小时）
3. 开始 NumPy 练习（1 小时）

**下周的目标**：
- 完成线性代数第 1-2 部分
- 完成 NumPy 基础
- 准备投稿 Paper 1

---

## 📞 遇到问题

- **理解概念卡住**：用不同的视角（YouTube、书、博客）
- **编程运行出错**：看错误信息、Google、实践调试
- **没有时间**：减少其他活动，这 6 个月很关键
- **太难了**：重新看更简单的教程，或者减少学习速度

你可以做到的！🚀

---

**最后：这份指南是为你定制的。如果你发现某部分太难或太简单，随时告诉我，我会调整。**

祝学习愉快！
