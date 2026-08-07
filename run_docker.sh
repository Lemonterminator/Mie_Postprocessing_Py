#!/bin/bash
set -e

echo "==================================================="
echo "  Mie Image Postprocessing Docker One-Click Script"
echo "  Mie 图像后处理 Docker 一键运行脚本"
echo "==================================================="
echo ""

# Check and create data folders
# 检查并创建数据文件夹
mkdir -p data/input data/output

# Check if Docker is running
# 检查 Docker 是否运行
if ! docker info > /dev/null 2>&1; then
    echo "[Error] Docker is not running! Please start Docker daemon first."
    echo "[错误] 未检测到 Docker 运行！请先启动 Docker 服务。"
    exit 1
fi

echo "[1/2] Automatically building and running Docker container..."
echo "[1/2] 正在自动构建并运行 Docker 容器..."
echo "---------------------------------------------------"
docker compose run --rm mie-postprocessing

echo ""
echo "==================================================="
echo "  Processing complete! Results saved to data/output directory."
echo "  处理完成！处理结果已自动保存至 data/output 目录。"
echo "==================================================="
