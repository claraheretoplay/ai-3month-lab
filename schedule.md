#总计划

走“通信信号 + 深度学习系统”的路线。3 个月目标不要定成“精通 PyTorch 内核”，而应定成：

1. 能独立写 Python/PyTorch 训练、验证、保存模型流程  
2. 能看懂常见 GitHub AI 项目结构  
3. 能手写一个小型 autograd / MLP 框架，理解反向传播、动态图、优化器  
4. 能写基础 C++，并把一个 C++ 函数或算子暴露给 Python  
5. 能做一个与你通信背景相关的项目，例如调制识别、信道估计、信号分类

按每周 10-15 小时设计。如果你每天能学 2 小时，周末再补 4-6 小时，刚好够。

**核心资料**
- Python：官方教程适合有一定技术基础的人，重点看语法、数据结构、函数、模块、类。官方也说明 Python 可用 C/C++ 扩展。([docs.python.org](https://docs.python.org/3/tutorial/))  
- PyTorch：官方 Learn the Basics 覆盖 tensor、dataset、dataloader、autograd、优化、保存模型等完整工作流。([pytorch.org](https://pytorch.org/tutorials/beginner/basics/intro.html))  
- 深度学习理论 + 代码：《Dive into Deep Learning》是开源交互式深度学习书，强调代码实践。([github.com](https://github.com/d2l-ai/d2l-en))  
- 框架底层：micrograd 是很小的 autograd 引擎，README 说明它用动态图 DAG 实现反向自动微分，并带一个类似 PyTorch API 的小神经网络库。([github.com](https://github.com/karpathy/micrograd))  
- C++：LearnCpp 从编译、调试、现代 C++ 基础开始，适合无 C++ 基础者。([learncpp.com](https://www.learncpp.com/))  
- Python/C++ 连接：pybind11 官方文档从把一个 C++ `add` 函数导出给 Python 开始。([pybind11.readthedocs.io](https://pybind11.readthedocs.io/en/stable/basics.html))  
- PyTorch C++：官方 C++ Frontend 教程说明 PyTorch 的 Python API 底层有大量 C++ 代码，并提供 C++17 API、tensor、autograd、优化器等能力。([pytorch.org](https://pytorch.org/tutorials/advanced/cpp_frontend.html))  

**第 1 个月：先能写，再能训练**
第 1 周：Python 基础  
学：变量、列表、字典、函数、类、文件读写、异常、包管理。  
做：写一个 `signal_tools.py`，生成正弦波、AWGN 噪声、BPSK/QPSK 简单信号。  
产出：GitHub 仓库 `ai-3month-lab`，每天 commit。

第 2 周：NumPy + 数据处理  
学：数组、广播、矩阵乘法、随机数、向量化。  
做：不用 sklearn，自己实现线性回归、逻辑回归、SVM hinge loss 的训练循环。  
重点：把你已有 SVM 知识迁移到“loss + gradient + optimizer”。

第 3 周：PyTorch 入门  
学：`torch.Tensor`、`requires_grad`、`nn.Module`、`Dataset/DataLoader`、`optim`。  
做：用 PyTorch 训练 MNIST 或 FashionMNIST；必须写训练集、验证集、保存模型、加载模型。  
不要只跑 notebook，要整理成命令行项目：`train.py`、`model.py`、`dataset.py`。

第 4 周：第一个通信 AI 项目  
做：合成 BPSK/QPSK/8PSK/QAM 信号，加不同 SNR 的噪声，用 1D-CNN 分类调制方式。  
你要画出：训练 loss、验证 accuracy、不同 SNR 下的准确率曲线。  
这周结束，你已经能把通信问题翻译成 AI 训练问题了。

**第 2 个月：理解深度学习和框架机制**
第 5 周：深度学习最小理论  
学：感知机、MLP、激活函数、交叉熵、反向传播、SGD/Adam、过拟合、正则化。  
做：不用 PyTorch，只用 NumPy 写一个两层 MLP，训练二维分类或调制分类的小版本。  
重点：每一层都手写 forward/backward。

第 6 周：CNN/RNN/Transformer 有选择地学  
你是通信方向，优先级建议：  
CNN：最高，适合频谱图、IQ 序列、信号分类。  
RNN/GRU/LSTM：了解，用于序列信号。  
Transformer：先理解 attention，不必一开始卷大模型。  
做：把第 4 周项目升级成 1D-CNN + 简单 attention 对比实验。

第 7 周：micrograd  
读 micrograd 源码，不要急着改。  
做三件事：  
1. 给每行核心代码写中文注释  
2. 改一个激活函数，例如 `sigmoid` 或 `tanh`  
3. 加一个小测试：用 PyTorch 的梯度结果对比 micrograd 的梯度结果  
这周你会真正理解 `loss.backward()` 背后在干什么。

第 8 周：写自己的 mini 框架  
做一个 `minitorch-lite`，只支持：`Tensor`、`Linear`、`ReLU`、`MSELoss`、`CrossEntropyLoss`、`SGD`。  
目标不是性能，是机制：计算图、梯度累积、参数更新、模块组合。  
参考 tinygrad 的项目结构即可，不建议此时深入 tinygrad 全部代码；tinygrad 是更完整的 PyTorch 风格小框架，代码里还涉及 C/CUDA/Metal 等后端。([github.com](https://github.com/tinygrad/tinygrad))  

**第 3 个月：建立 C++ 能力和系统视角**
第 9 周：C++ 基础  
学：编译、头文件、函数、类、引用、指针、`vector`、`string`、`const`、CMake。  
做：用 C++ 写矩阵类 `Matrix`，支持加法、乘法、转置。  
重点：不要追求模板元编程，先写清楚、编译通过、能测试。

第 10 周：Python 调 C++  
用 pybind11 把 C++ 的矩阵乘法暴露给 Python。  
做：Python 里调用 `cpp_matmul(a, b)`，和 NumPy/PyTorch 结果对比。  
再做：计时比较 Python 循环、NumPy、C++ 三者性能。

第 11 周：PyTorch C++/扩展入门  
学 PyTorch C++ Frontend 或 custom extension。  
做：写一个简单 C++ 扩展算子，例如 `scale_add(x, a, b) = a*x + b`，Python 里调用。  
目标：理解 PyTorch 不是“纯 Python 魔法”，Python 只是易用入口，底层 tensor、autograd、调度、算子大量依赖 C++。

第 12 周：最终项目整合  
做一个完整 GitHub 项目：`deep-signal-classifier`。  
内容包括：  
- Python/PyTorch 训练代码  
- 合成通信信号数据集  
- 1D-CNN 或 CNN+attention 模型  
- 训练曲线和 SNR-accuracy 曲线  
- 一个 C++/pybind11 小模块，比如快速生成噪声、矩阵乘法或简单滤波  
- README 写清楚环境、运行命令、结果图、你理解的框架机制

**每天怎么学**
建议固定节奏：

- 30 分钟：看资料，只看当天要用的部分  
- 60 分钟：写代码  
- 20 分钟：调 bug、加测试  
- 10 分钟：写学习日志，记录“今天卡在哪里、怎么解决的”

最重要的规则：每周必须有可运行代码。不要陷入“先把理论全学完”。深度学习和框架机制是写出来的，不是看出来的。

**GitHub 项目顺序**
按这个顺序来：

1. 自己的 `ai-3month-lab`：所有练习都放进去  
2. [d2l-ai/d2l-en](https://github.com/d2l-ai/d2l-en)：跟着改代码，学理论  
3. [karpathy/micrograd](https://github.com/karpathy/micrograd)：理解 autograd  
4. [tinygrad/tinygrad](https://github.com/tinygrad/tinygrad)：第 8 周后只读局部代码  
5. [pybind/pybind11](https://github.com/pybind/pybind11)：做 Python/C++ 连接  
6. [karpathy/llm.c](https://github.com/karpathy/llm.c)：3 个月后再看，它是 C/CUDA/LLM 训练项目，当前阶段只作为远期目标。([github.com](https://github.com/karpathy/llm.c))  

3 个月后，你应该不是“只会调包”的状态，而是能说清楚：数据怎么进模型、loss 怎么产生、梯度怎么回传、参数怎么更新、Python 如何调用 C++、为什么 PyTorch 能高效。这个路线比较硬，但很适合你。

##week01
下面是**第一周每日可达成计划**，目标只有一个：把 Python 基础、开发环境、GitHub 习惯建立起来，并写出一个“通信信号生成小工具”。

假设你每天 1.5-2 小时。如果某天时间少，优先完成“必须产出”。

**第 1 天：搭好环境 + 跑通第一个 Python 程序**
目标：电脑能稳定写、跑、管理代码。

学习内容：
- 安装/确认 Python 3.10+
- 安装 VS Code 或 PyCharm
- 理解 `.py` 文件、终端、虚拟环境
- 学会运行：`python hello.py`

必须产出：
```python
print("Hello AI and Communication")
```

再做一个小脚本：
```python
name = "BPSK"
snr = 10
print(f"Signal: {name}, SNR: {snr} dB")
```

当天结束你要能回答：
- Python 文件怎么运行？
- 终端当前目录是什么意思？
- `print()`、变量、字符串格式化是什么？

**第 2 天：Python 基础语法**
目标：能写简单逻辑。

学习内容：
- 数字、字符串、列表、字典
- `if/else`
- `for` 循环
- 函数 `def`

必须产出：写一个 `snr_level.py`

功能：
```python
def classify_snr(snr):
    if snr < 0:
        return "low"
    elif snr < 15:
        return "medium"
    else:
        return "high"
```

测试：
```python
for snr in [-5, 0, 10, 20]:
    print(snr, classify_snr(snr))
```

当天结束你要能回答：
- 列表和字典分别适合存什么？
- 函数输入和返回值是什么？
- 为什么要把重复逻辑写成函数？

**第 3 天：NumPy 入门**
目标：开始用数组表达信号。

学习内容：
- `numpy.array`
- `np.arange`
- `np.sin`
- `np.random.randn`
- 数组加减乘除

必须产出：写一个 `generate_sine.py`

功能：
生成一个 1 kHz 正弦波：

```python
import numpy as np

fs = 10000
f = 1000
duration = 0.01

t = np.arange(0, duration, 1 / fs)
x = np.sin(2 * np.pi * f * t)

print(t[:10])
print(x[:10])
```

进阶一点：
给正弦波加噪声：

```python
noise = 0.1 * np.random.randn(len(x))
y = x + noise
```

当天结束你要能回答：
- 为什么信号可以用数组表示？
- `fs`、`f`、`duration` 分别是什么意思？
- `np.random.randn` 生成的是什么？

**第 4 天：Matplotlib 画图**
目标：能把信号画出来，不再只看数字。

学习内容：
- `matplotlib.pyplot`
- 画时域波形
- 保存图片

必须产出：写一个 `plot_signal.py`

功能：
画出干净正弦波和加噪正弦波。

```python
import numpy as np
import matplotlib.pyplot as plt

fs = 10000
f = 1000
duration = 0.01

t = np.arange(0, duration, 1 / fs)
x = np.sin(2 * np.pi * f * t)
y = x + 0.2 * np.random.randn(len(x))

plt.plot(t, x, label="clean")
plt.plot(t, y, label="noisy", alpha=0.7)
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.savefig("sine_noise.png")
plt.show()
```

当天结束你要能回答：
- 时域图横轴和纵轴是什么？
- 噪声强度变大后图像有什么变化？
- `label`、`legend`、`savefig` 的作用是什么？

**第 5 天：生成 BPSK 信号**
目标：把通信基础真正接到 Python 上。

学习内容：
- 随机比特
- BPSK 映射：`0 -> -1`，`1 -> +1`
- 加性高斯白噪声 AWGN

必须产出：写一个 `bpsk.py`

```python
import numpy as np

def generate_bits(n):
    return np.random.randint(0, 2, size=n)

def bpsk_modulate(bits):
    return 2 * bits - 1

def add_noise(signal, noise_std):
    noise = noise_std * np.random.randn(len(signal))
    return signal + noise

bits = generate_bits(20)
symbols = bpsk_modulate(bits)
received = add_noise(symbols, 0.5)

print("bits:", bits)
print("symbols:", symbols)
print("received:", received)
```

当天结束你要能回答：
- BPSK 为什么可以看成二分类？
- `0/1` 比特为什么要映射成 `-1/+1`？
- 噪声标准差越大，接收信号会怎样？

**第 6 天：组织成小项目**
目标：不再写散乱脚本，开始有 GitHub 项目结构。

建议目录：

```text
ai-3month-lab/
  week01_python_signal/
    signal_tools.py
    demo_sine.py
    demo_bpsk.py
    README.md
```

必须产出：写 `signal_tools.py`

```python
import numpy as np

def generate_sine(fs, f, duration):
    t = np.arange(0, duration, 1 / fs)
    x = np.sin(2 * np.pi * f * t)
    return t, x

def generate_bits(n):
    return np.random.randint(0, 2, size=n)

def bpsk_modulate(bits):
    return 2 * bits - 1

def add_awgn(signal, noise_std):
    noise = noise_std * np.random.randn(len(signal))
    return signal + noise
```

再写 `demo_bpsk.py` 调用它。

当天结束你要能回答：
- 为什么要把函数放进 `signal_tools.py`？
- `import` 是什么？
- 一个项目为什么需要 README？

**第 7 天：复盘 + 上传 GitHub**
目标：形成第一周成果。

学习内容：
- Git 基础：`git init`、`git add`、`git commit`
- GitHub 创建仓库
- README 写清楚项目做了什么

必须产出：
README 包含这几段：

```markdown
# AI 3-Month Lab

## Week 01: Python and Communication Signal Basics

This week I learned:
- Python variables, functions, lists, dictionaries
- NumPy arrays
- Matplotlib plotting
- Sine wave generation
- BPSK modulation
- AWGN noise

## Files
- signal_tools.py: basic signal generation functions
- demo_sine.py: generate and plot sine wave
- demo_bpsk.py: generate BPSK symbols and noisy received signal
```

当天结束你要能回答：
- 这一周我写了哪些函数？
- 哪个函数最像以后机器学习里的“数据生成器”？
- 如果我要做调制识别，下一步需要生成哪些类别的数据？

**第一周最终验收标准**
你完成第一周后，应该有这些东西：

```text
signal_tools.py
demo_sine.py
demo_bpsk.py
sine_noise.png
README.md
GitHub 仓库
```

你应该能独立解释：

- Python 函数怎么写
- NumPy 数组怎么表示信号
- 正弦波怎么生成
- BPSK 怎么从比特映射成符号
- AWGN 噪声怎么加
- 怎么把代码上传 GitHub

第一周不要碰 PyTorch。先把 Python、NumPy、信号生成这条线跑顺。第二周再进入“数据集 + 简单机器学习训练循环”，这样后面学 PyTorch 会稳很多。

##week02
## 第二周目标

学习：

```text
数组、广播、矩阵乘法、随机数、向量化
```

完成：

```text
线性回归、逻辑回归、SVM hinge loss
```

理解核心流程：

```text
预测 → 计算 loss → 计算 gradient → 更新参数 → 重复训练
```

建议每天 `1.5～2小时`：概念学习30分钟、编程60分钟、复习与 Git 提交20分钟。

### 第1天：NumPy 数组与形状

学习：

- 创建一维、二维数组。
- 理解 `shape`、`ndim`、`dtype`。
- 学习索引、切片和 `reshape()`。
- 区分 Python 列表与 NumPy 数组。

创建：`day01_numpy_arrays.py`

实践：把多个接收信号组织成“行表示样本、列表示特征”的二维数组。

当天结束能回答：

- `shape` 和 `ndim` 分别表示什么？
- `(100,)` 和 `(100, 1)` 有什么区别？
- 为什么机器学习数据通常是二维数组？

### 第2天：随机数、广播与向量化

学习：

- 使用 `np.random.randn()` 和 `np.random.randint()`。
- 理解数组和标量之间的广播。
- 用数组运算代替 `for` 循环。
- 复习 BPSK 映射与 AWGN。

创建：`day02_vectorized_bpsk.py`

实践：

```python
bits = np.random.randint(0, 2, 1000)
symbols = 2 * bits - 1
noise = noise_std * np.random.randn(1000)
received = symbols + noise
detected_bits = (received > 0).astype(int)
```

当天结束能回答：

- 广播是什么？
- 向量化为什么比 Python 循环更适合数值计算？
- `randn()` 生成的数据具有什么分布？

### 第3天：矩阵乘法与机器学习数据

学习：

- 理解样本数、特征数和标签。
- 学习转置 `.T` 和矩阵乘法 `@`。
- 理解模型公式 `y_hat = X @ w + b`。
- 使用随机数生成简单训练数据。

创建：`day03_matrix_data.py`

重点检查形状：

```text
X：(样本数, 特征数)
w：(特征数, 1)
y：(样本数, 1)
```

当天结束能回答：

- `X @ w` 为什么能够一次计算所有样本？
- `X`、`w`、`b`、`y` 分别是什么？
- 矩阵相乘时形状需要满足什么条件？

### 第4天：手写线性回归

学习：

- 创建 `y = 3x + 2 + noise` 数据。
- 实现预测 `y_hat = X @ w + b`。
- 实现均方误差 MSE。
- 手写 `dw`、`db` 和参数更新。
- 记录并绘制 loss 曲线。

创建：`day04_linear_regression.py`

参数更新：

```python
w = w - learning_rate * dw
b = b - learning_rate * db
```

完成标准：loss 逐渐下降，训练出的 `w` 接近3、`b` 接近2。

### 第5天：手写逻辑回归

学习：

- 理解回归与二分类的区别。
- 实现 Sigmoid 函数。
- 实现二元交叉熵 loss。
- 手写逻辑回归训练循环。
- 用带噪 BPSK 接收值预测原始比特。

创建：`day05_logistic_regression.py`

当天结束能回答：

- Sigmoid 为什么能把输出变成概率？
- 为什么概率大于 `0.5` 可以判为类别1？
- loss 下降与分类准确率上升有什么关系？

### 第6天：手写 SVM hinge loss

学习：

- 把类别标签从 `0/1` 转换为 `-1/+1`。
- 复习分类分数 `score = X @ w + b`。
- 实现 hinge loss：`max(0, 1 - y * score)`。
- 计算梯度并更新 `w`、`b`。
- 统计 BPSK 比特检测准确率。

创建：`day06_svm_hinge.py`

当天结束能回答：

- `y * score >= 1` 表示什么？
- 哪些样本会产生 hinge loss？
- SVM 的间隔思想如何体现在 hinge loss 中？

### 第7天：比较模型并整理项目

学习与实践：

- 对比逻辑回归和 SVM 的分类结果。
- 测试 `noise_std = 0.2、0.5、1.0`。
- 绘制不同噪声下的准确率。
- 整理重复函数，编写 `README.md`。
- 检查并提交本周代码。

创建：`day07_compare_models.py`

```powershell
git add .
git commit -m "Complete week 02 NumPy machine learning"
```

最终目录：

```text
week02_numpy_ml/
├── day01_numpy_arrays.py
├── day02_vectorized_bpsk.py
├── day03_matrix_data.py
├── day04_linear_regression.py
├── day05_logistic_regression.py
├── day06_svm_hinge.py
├── day07_compare_models.py
└── README.md
```

本周所说的“optimizer”暂时不是 PyTorch 中的对象，而是你亲手写下的参数更新：

```python
parameter = parameter - learning_rate * gradient
```

掌握这个过程后，第3周学习 PyTorch 自动求导和 `torch.optim` 时，就能理解框架替你完成了哪些工作。