这个命令下载并执行 Ollama 的安装脚本。
This command downloads and executes the Ollama installation script.

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

启动 Ollama 服务。
Start the Ollama server.
```bash
ollama serve
```

运行指定的 DeepSeek-R1 模型。
Run the specified DeepSeek-R1 model.
```bash
ollama run deepseek-r1:7b-qwen-distill-q4_K_M
```


设备性能：在手机上运行这些模型可能会对性能有较大的影响，特别是内存占用方面。
In mobile devices, running these models might significantly impact performance, especially in terms of memory usage.
网络连接：确保您有稳定的网络连接，因为可能需要从远程下载模型文件或更新。
Ensure you have a stable internet connection as models might need to be downloaded or updated from remote servers.
存储空间：模型文件可能很大，确保设备有足够的存储空间。
Model files can be large, so make sure your device has enough storage space.
安全性：执行从网络下载的脚本时，请确保您信任该来源，因为这涉及到在您的设备上执行代码。
When executing scripts downloaded from the internet, ensure you trust the source since this involves running code on your device.
