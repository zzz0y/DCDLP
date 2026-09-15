# Docker 服务器运行包

这个镜像用于 Step 9–12 的 selective-routing 筛选，不会启动 1195 个全量实验。镜像只包含代码、Python/CUDA 依赖和 Cora/HeaRT 官方小数据文件；`results/`、旧实验结果以及 OGB 大缓存不会进入镜像。

## 在 Windows 电脑上构建并导出

在仓库根目录的 PowerShell 执行：

```powershell
.\docker\build_selective_image.ps1
```

默认生成：

```text
offline/dcdlp-selective-image.tar
offline/dcdlp-selective-image.tar.sha256
```

脚本不会覆盖已有同名归档。请把这两个文件和 `docker/start_selective_server.sh` 上传到服务器；也可以直接上传整个仓库目录。服务器只需要 Linux Docker、NVIDIA Container Toolkit（GPU 模式）和校园网可用的 Docker 网络。

## 服务器一键启动

```bash
chmod +x docker/start_selective_server.sh
./docker/start_selective_server.sh
```

默认是安全的 Cora 五个 seed、M0/M1/M2/M3、5+5 epochs 筛选，输出到全新的 `results/phase2_selective_server/cora5/`。脚本默认使用 Docker `bridge` 网络，适合需要校园网的服务器；运行时已打包 Cora 数据，不依赖临时下载。

第一次只检查环境、不启动训练：

```bash
DCDLP_RUN_MODE=preflight ./docker/start_selective_server.sh
```

没有 GPU 时可以做 CPU Smoke：

```bash
DCDLP_RUN_MODE=smoke DCDLP_GPUS=none ./docker/start_selective_server.sh
```

常用可调参数（均不覆盖旧目录）：

```bash
DCDLP_NETWORK=bridge DCDLP_GPUS=all \
DCDLP_PRETRAIN_EPOCHS=5 DCDLP_ROUTING_EPOCHS=5 \
./docker/start_selective_server.sh
```

查看、停止和恢复：

```bash
docker logs -f dcdlp-selective-server
docker stop dcdlp-selective-server
./docker/start_selective_server.sh   # 重新启动时会 --resume 已完成条目
```

`DCDLP_IMAGE_ARCHIVE=/path/to/dcdlp-selective-image.tar` 可指定归档位置；`DCDLP_CONTAINER` 可更换容器名。若校园网需要代理，Docker 启动脚本会透传已设置的 `HTTP_PROXY/HTTPS_PROXY/NO_PROXY` 环境变量。
