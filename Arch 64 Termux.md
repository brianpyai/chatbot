```markdown
# 关键要点

- 已修改流程以安装 Arch Linux 而非 Debian，确保所有工具和库为 64 位兼容。
- 优化了 transformers、torch、onnx 和 onnxruntime 的性能。
- 流程现支持直接以 root 用户登录，省去非 root 用户设置。
- 确保 MediaTek Dimensity 9300+（ARMv9-A 架构）的 64 位环境兼容性。

---

# 安装和配置概述

## 流程修改

- **原流程调整**：  
  原流程已调整为使用 Arch Linux，通过 [proot-distro](https://github.com/proot-me/proot-distro) 在 Termux 上运行，确保所有软件均为 64 位版本，适合 ARMv9-A 架构的 MediaTek Dimensity 9300+ 处理器。

- **性能优化**：  
  - **Llama.cpp** 使用 OpenBLAS 编译，提升矩阵运算性能。  
  - **Python 库**（如 torch 和 onnxruntime）通过 pip 安装，确保 ARM64 优化。  
  - 利用 Arch Linux 的滚动更新特性获取最新包，支持 AI 开发性能。

- **直接 root 登录**：  
  取消非 root 用户创建，直接以 root 登录，简化操作（但需注意安全风险）。

- **令人意外的细节**：  
  流程中原本计划安装的 `python3-torch` 等包在 Debian 中可能不可用，需通过 pip 安装，这反映了 AI 开发环境的复杂性。

---

# 详细报告

本文档详细描述了为 MediaTek Dimensity 9300+ 处理器（基于 ARMv9-A 架构的纯 64 位设计，不支持 32 位指令集 AArch32）优化 AI 开发环境的修改流程。目标包括：

- 安装 Arch Linux
- 确保所有工具和库为 64 位兼容
- 编译 Ollama 和 OpenBLAS
- 提升 transformers、torch、onnx 和 onnxruntime 的性能
- 支持直接 root 登录

以下是修改后的完整流程及相关分析。

---

## 流程修改背景

- **原流程**：基于 Termux 和 proot-distro 安装 Debian。
- **用户需求**：改为 Arch Linux，并强调 64 位兼容性和性能优化。
- **优势**：  
  - Arch Linux 的滚动更新特性使软件包通常较新，更适合 AI 开发需求。  
  - 直接 root 登录简化了操作（但存在安全风险）。

---

# 详细步骤

## 1. 安装 Termux 和相关应用

从 F-Droid 下载并安装以下应用：
- **Termux**
- **Termux:Widget**
- **Termux:API**

这些应用提供终端访问、桌面快捷方式和 Android 集成，无需 root 权限。

---

## 2. 初始 Termux 设置

执行以下命令配置环境：

```bash
termux-setup-storage
termux-wake-lock
termux-change-repo
pkg update && pkg upgrade
pkg install git proot-distro vim termux-API
pkg install qemu-user-arm
```

- `termux-setup-storage`：授权访问手机存储。  
- `termux-wake-lock`：防止设备在长时间操作时休眠。  
- `termux-change-repo`：切换到更快的仓库镜像。

更新和升级包列表，安装 git、proot-distro（运行 Linux 发行版）、vim 和 termux-API。  
*备注：* `qemu-user-arm` 用于模拟 ARM 二进制，但鉴于设备为 ARM64，可能非必需。

---

## 3. 安装和配置 Arch Linux

使用 proot-distro 安装 Arch Linux，并创建快捷脚本：

```bash
disname='archlinux'
rm -rf .shortcuts
mkdir .shortcuts
echo "proot-distro login $disname" > .shortcuts/archlinux.sh
chmod +x .shortcuts/archlinux.sh
proot-distro install archlinux
proot-distro login archlinux
```

- 删除并重建 `.shortcuts` 目录。  
- 创建 `archlinux.sh` 脚本，实现直接以 root 用户登录。  
- 安装 Arch Linux 并以 root 用户登录，可通过 Termux:Widget 添加快捷方式。

---

## 4. 设置 Arch Linux 环境

在 Arch Linux 环境中配置本地化：

```bash
pacman -Syu
pacman -S locales
echo "LANG=zh_CN.UTF-8" >> /etc/locale.conf
locale-gen zh_CN.UTF-8
```

- `pacman -Syu`：更新和升级包。  
- 安装 `locales` 以支持语言设置。  
- 配置中文 UTF-8 环境。

---

## 5. 安装必要包和 Python 环境

以 root 身份安装所需软件：

```bash
pacman -Syu
pacman -S python3 python3-pip git curl vim wget clang cmake opencl-headers ffmpeg libopenblas base-devel git-lfs go
pacman -S cmake ninja openblas
pacman -S python-numpy
```

- **说明**：  
  - 确保安装 Python 3 和 pip。  
  - clang 和 cmake 用于编译。  
  - `libopenblas` 用于性能优化。  
  - 增加 `go` 包以便编译 Ollama（如需编译源代码）。

*备注：* 在 Arch Linux 中，`python3` 是默认的 Python 3 包，`base-devel` 相当于 Debian 的 `build-essential`。

---

## 6. 编译 Llama.cpp 使用 OpenBLAS

编译 Llama.cpp 以优化性能：

```bash
rm -rf llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS
cmake --build build --config Release
chmod -R +x llama.cpp/build/bin
llama.cpp/build/bin/test-c
cd ..
```

- 克隆 Llama.cpp 仓库，并启用 OpenBLAS 以提升矩阵运算性能。  
- 注意：确保在第 5 步中已经安装 `libopenblas`。

---

## 7. 设置 Python 虚拟环境并安装包

创建虚拟环境并安装 Python 库：

```bash
python3 -m venv llm
echo "cd /storage/emulated/0/Documents/Pydroid3/llm && source ~/llm/bin/activate" > llm.sh
source llm.sh
pip install --upgrade pip
pip install onnxruntime nncf
pip install --upgrade diffusers[torch] accelerate peft openvino
pip install opencv-python fastapi uvicorn flask
pip install climage
pip install fastapi python-multipart pydantic opencc-python-reimplemented pandas faiss-cpu fastapi_poe music21
pip install fast-sentence-transformers langchain
pip install wikipedia unstructured pypdf pdf2image pdfminer chromadb qdrant-client lark momento annoy
pip install doc2text pypandoc pandoc
pip install gradio ollama rembg
pip install music21 tensorboard
pip install scipy scikit-learn torch transformers
pip install tqdm moviepy==1.0.3 imageio>=2.5.0 ffmpeg-python audiofile>=0.0.0 opencv-python>=4.5 decorator>=4.3.0
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
ollama run deepseek-r1:1.5b-qwen-distill-q8_0
```

- 创建虚拟环境 `llm` 以隔离依赖。  
- 安装各项 AI/ML 相关库，确保 **torch** 和 **onnxruntime** 与 ARM64 兼容。  
- 使用 Ollama 脚本安装并运行轻量模型 `deepseek-r1:1.5b-qwen-distill-q8_0` 用以测试。  
- *注意：* 原列表中的 `sqlite3` 已移除，因为其包含于 Python 标准库中，无需单独安装。

---

## 8. 性能优化分析

- **torch 和 transformers**  
  通过 pip 安装，确保使用 ARM64 预编译轮子，官方支持 ARM64 优化 CPU 计算。

- **onnx 和 onnxruntime**  
  使用标准 pip 安装，应包含 ARM64 优化。虽然 OpenCL 支持可能利用 GPU，但 proot-distro 可能限制硬件访问。

- **OpenBLAS**  
  通过 pacman 安装，确保 64 位兼容，并在编译 Llama.cpp 时启用来提升矩阵运算效率。

- **Ollama**  
  安装脚本检测目标架构，预编译二进制应为 ARM64。若需源代码编译，则可使用 go 编译，但通常无需。

---

## 9. 安全和兼容性考虑

- **直接 root 登录**：  
  虽然操作简化，但会增加安全风险，建议在生产环境中使用非 root 用户。

- **滚动更新**：  
  Arch Linux 的滚动更新确保软件包最新，但可能对稳定性产生一定影响。

- **兼容性**：  
  所有安装包通过 pacman 和 pip 确保 64 位兼容，ARMv9-A 向下兼容 ARMv8-A，实现无障碍软件支持。

---

## 10. 对比表：Debian vs Arch Linux

| 特性         | Debian                              | Arch Linux                         |
| ------------ | ----------------------------------- | ---------------------------------- |
| 包管理器     | apt                                 | pacman                             |
| 更新模式     | 版本发布（稳定）                    | 滚动更新（最新）                   |
| Python 包    | 可能缺 AI 专用包（如 torch）         | 滚动更新可能包含最新版本            |
| 适合 AI 开发 | 需要更多 pip 安装                    | 更适合快速获取最新工具              |

---

## 11. 结论

- 修改后的流程确保了使用 Arch Linux 进行安装，保证工具和库的 64 位兼容性，优化了 AI 性能，并支持直接 root 登录。  
- 用户可通过 Termux:Widget 快捷访问，但需注意安全风险并定期更新软件包。

---

# 关键引用

- [Arch Linux 安装指南](https://wiki.archlinux.org/)
- [PyTorch ARM64 支持](https://pytorch.org)
```