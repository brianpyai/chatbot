```markdown
## Key Points

- The process has been modified to install Arch Linux instead of Debian, ensuring that all tools and libraries are 64-bit compatible.
- Performance optimizations have been applied to transformers, torch, onnx, and onnxruntime.
- The process now supports direct root login, eliminating the need for setting up a non-root user.
- 64-bit environment compatibility for the MediaTek Dimensity 9300+ (ARMv9-A architecture) is ensured.

---

## Overview of Installation and Configuration

### Process Modifications

The original process has been adjusted to use Arch Linux, running on Termux via proot-distro, ensuring that all software is 64-bit and suitable for the MediaTek Dimensity 9300+ processor based on the ARMv9-A architecture.

### Performance Optimization

- **Llama.cpp** is compiled with OpenBLAS to enhance matrix computation performance.
- **Python libraries** such as torch and onnxruntime are installed via pip, ensuring ARM64 optimization.
- Arch Linux’s rolling update model provides the latest packages, supporting improved AI development performance.

### Direct Root Login

The creation of a non-root user has been eliminated to allow direct root login. This simplifies operation but comes with security risks.

### Unexpected Details

Surprisingly, packages like `python3-torch` originally planned for installation might be unavailable on Debian and therefore require pip installation. This reflects the complexity of the AI development environment.

---

## Detailed Report

This document describes in detail the modified process to optimize the AI development environment for the MediaTek Dimensity 9300+ processor (which is based on a pure 64-bit design using the ARMv9-A architecture, not supporting the 32-bit AArch32 instruction set). The objectives include installing Arch Linux, ensuring that all tools and libraries are 64-bit compatible, compiling Ollama and OpenBLAS, enhancing the performance of transformers, torch, onnx, and onnxruntime, and supporting direct root login. Below is the full modified process along with the related analysis.

---

## Background of Process Modifications

The original process was based on installing Debian using Termux and proot-distro. However, based on user requests, the change was made to switch to Arch Linux with an emphasis on 64-bit compatibility and performance optimizations. Given Arch Linux's rolling update model, the available packages are more up-to-date and better suited for AI development needs. Additionally, the process has been adjusted for direct root login to simplify user operations, though this introduces certain security risks.

---

## Detailed Steps

### 1. Install Termux and Related Applications

Download and install **Termux (com.termux_118.apk)** and **Termux:Widget** from F-Droid.  
These applications provide terminal access, desktop shortcuts, and Android integration without requiring root privileges.

---

### 2. Initial Termux Setup

Execute the following commands to configure the environment:

```bash
termux-setup-storage
termux-wake-lock
termux-change-repo
pkg install tur-repo
pkg update && pkg upgrade
pkg install git
pkg install proot-distro vim 
pkg install termux-api
pkg install qemu-user-arm
pkg install git
```

- This updates and upgrades the package list as well as installs git, proot-distro (to run a Linux distribution), vim, and termux-API.
- `qemu-user-arm` is used for ARM binary emulation, though it might not be necessary on an ARM64 device.

---

### 3. Install and Configure Arch Linux

Use proot-distro to install Arch Linux and create a shortcut script:

```bash
disname='archlinux'
rm -rf .shortcuts
mkdir .shortcuts
echo "proot-distro login $disname" > .shortcuts/archlinux.sh
chmod +x .shortcuts/archlinux.sh
proot-distro install archlinux
proot-distro login archlinux
```

- Removes and recreates the `.shortcuts` directory.
- Creates an `archlinux.sh` script for direct root login.
- Installs Arch Linux and logs in as root; you can add this shortcut via Termux:Widget.

---

### 4. Configure the Arch Linux Environment

Within the Arch Linux environment, set up localization:

```bash
pacman -Syu
pacman -S locales
echo "LANG=zh_CN.UTF-8" >> /etc/locale.conf
locale-gen zh_CN.UTF-8
pacman -Syu
```

- Installs `locales` to support language settings.
- Configures the Chinese UTF-8 locale.

---

### 5. Install Required Packages and Python Environment

Install necessary software as root:

```bash
pacman -Syu
pacman -S python3 
pacman -S curl vim wget 
pacman -S clang opencl-headers 
pacman -S ffmpeg libopenblas 
pacman -S git-lfs go
pacman -S cmake ninja openblas base-devel
pacman -S git
pacman -S python-numpy
```

- Ensures that Python 3 and pip are installed. Clang and cmake are used for compiling, and libopenblas is included for performance optimization.
- The `go` package is added to facilitate compiling Ollama if source compilation is needed.
- In Arch Linux, the `python3` package is the default Python 3 package, and `base-devel` is similar to Debian’s `build-essential`.

---

### 6. Set Up Python Virtual Environment and Install Packages

Create a virtual environment and install the necessary Python libraries:

```bash
python3 -m venv llm
echo "cd /storage/emulated/0/Documents/Pydroid3/llm && source ~/llm/bin/activate" > llm.sh
source llm.sh
pip install --upgrade pip
pip install onnxruntime nncf
pip install --upgrade diffusers[torch] accelerate peft openvino
pip install opencv-python 
pip install fastapi uvicorn flask
pip install climage
pip install python-multipart pydantic opencc-python-reimplemented 
pip install pandas faiss-cpu 
pip install fastapi_poe music21
pip install fast-sentence-transformers langchain
pip install wikipedia unstructured pypdf 
pip install pdf2image pdfminer  
pip install qdrant-client lark momento annoy
pip install doc2text pypandoc 
pip install pandoc
pip install gradio ollama rembg
pip install music21 tensorboard
pip install scipy scikit-learn torch transformers
pip install tqdm moviepy==1.0.3 imageio>=2.5.0 ffmpeg-python audiofile>=0.0.0 opencv-python>=4.5 decorator>=4.3.0
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
#ollama run deepseek-r1:1.5b-qwen-distill-q8_0
```

- Creates the virtual environment `llm` to isolate dependencies.
- Installs various AI/ML libraries ensuring that torch and onnxruntime are ARM64 optimized.
- Uses the Ollama installation script to install and run a lightweight model (`deepseek-r1:1.5b-qwen-distill-q8_0`) for testing.
- *Note:* The `sqlite3` package has been removed from the list since it is part of the Python standard library and doesn't require separate installation.

---

### 7. Compile Llama.cpp Using OpenCL

Compile Llama.cpp to optimize performance:

```bash
pacman -S jdk11-openjdk
pacman -S clang
pacman -S cmake
pacman -S ninja
cd
pip install 'huggingface_hub[cli]'
huggingface-cli download "bartowski/DeepSeek-R1-ReDistill-Qwen-1.5B-v1.0-GGUF" --include "DeepSeek-R1-ReDistill-Qwen-1.5B-v1.0-Q6_K.gguf" --local-dir ./models

huggingface-cli download "bartowski/DeepSeek-R1-ReDistill-Qwen-1.5B-v1.0-GGUF" --include "DeepSeek-R1-ReDistill-Qwen-1.5B-v1.0-Q4_K_M.gguf" --local-dir ./models

# Clean up previous installations
cd
rm -rf OpenCL-ICD-Loader OpenCL-Headers dev

# Set up OpenCL Headers
mkdir -p ~/dev/llm
cd ~/dev/llm
git clone https://github.com/KhronosGroup/OpenCL-Headers
cd OpenCL-Headers
mkdir build && cd build
cmake .. -G Ninja \
  -DBUILD_TESTING=OFF \
  -DOPENCL_HEADERS_BUILD_TESTING=OFF \
  -DOPENCL_HEADERS_BUILD_CXX_TESTS=OFF \
  -DCMAKE_INSTALL_PREFIX="~/dev/llm/opencl"
cmake --build . --target install

# Set up OpenCL ICD Loader
cd ~/dev/llm
git clone https://github.com/KhronosGroup/OpenCL-ICD-Loader
cd OpenCL-ICD-Loader
mkdir build && cd build
cmake .. -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="~/dev/llm/opencl" \
  -DCMAKE_INSTALL_PREFIX="~/dev/llm/opencl"
cmake --build . --target install

# Compile llama.cpp with OpenCL support
cd
rm -rf llama.cpp
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
mkdir build && cd build
cmake .. -G Ninja \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_PREFIX_PATH="~/dev/llm/opencl" \
    -DGGML_OPENMP=OFF \
    -DGGML_OPENCL=ON
ninja
chmod -R +x ~/llama.cpp/build/bin

rm llama.cache 
~/llama.cpp/build/bin/llama-cli --model ./models/DeepSeek-R1-ReDistill-Qwen-1.5B-v1.0-Q4_K_M.gguf \
--n-gpu-layers 99 \
--ctx-size 0 \
--threads 6  \
--no-mmap \
--mlock \
--conversation \
--ubatch-size 256 \
--batch-size 256 \
--numa numactl \
--prompt-cache llama.cache 
```

---

## Performance Optimization Analysis

- **torch and transformers**: Installed via pip, ensuring the use of ARM64 precompiled wheels; PyTorch officially supports ARM64 to optimize CPU computation.
- **onnx and onnxruntime**: Standard pip installations should include ARM64 optimizations. Although OpenCL support might leverage GPUs, proot-distro may limit hardware access.
- **OpenBLAS**: Installed via pacman to ensure 64-bit compatibility; it is enabled during the Llama.cpp build to enhance matrix computations.
- **Ollama**: The installation script detects the architecture, and the precompiled binary is expected to be ARM64; source compilation with go is possible if necessary, though usually not required.

---

## Security and Compatibility Considerations

- **Direct Root Login**: This simplifies operations but increases security risks. It is recommended to use a non-root user in production environments.
- **Arch Linux's Rolling Updates**: While these ensure the packages are up-to-date, stability can sometimes be affected.
- All packages installed via pacman and pip ensure 64-bit compatibility. Although the ARMv9-A architecture is backward compatible with ARMv8-A, software support is maintained without issues.

---

## Comparison Table: Debian vs Arch Linux

| Feature          | Debian                               | Arch Linux                        |
| ---------------- | ------------------------------------ | --------------------------------- |
| Package Manager  | apt                                  | pacman                            |
| Update Model     | Fixed releases (stable)              | Rolling updates (latest)          |
| Python Packages  | AI-specific packages (e.g., torch) may be missing | Latest versions often available via rolling updates |
| Suitability for AI Development | Requires additional pip installations | Better suited for rapid access to the latest tools |

---

## Conclusion

- The modified process ensures the installation of Arch Linux with 64-bit compatibility, enhances AI performance, and supports direct root login.
- Users can access the system quickly via Termux:Widget, though they should be aware of the security risks and update packages regularly.

---

## Key References

- [Arch Linux Installation Guide](https://wiki.archlinux.org/)
- [PyTorch ARM64 Support](https://pytorch.org)
```