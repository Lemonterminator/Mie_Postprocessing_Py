# Base image: Python 3.11 slim version
# 基础镜像：Python 3.11 瘦身版
FROM python:3.11-slim

# Set environment variables: disable bytecode (.pyc), unbuffered output, headless Matplotlib backend
# 设置环境变量：禁止 Python 产生 pyc 文件，无缓冲输出，Matplotlib 采用无界面后端
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

# Set working directory
# 工作目录设置
WORKDIR /app

# Install system dependencies (OpenCV, FFmpeg, build tools, etc.)
# 安装必要的系统依赖（OpenCV, FFmpeg, 编译工具等）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
# 升级 pip 并安装基础工具
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy project configuration and install dependencies
# 复制项目配置文件并安装依赖
COPY pyproject.toml MANIFEST.in README.md ./
COPY OSCC_postprocessing/ ./OSCC_postprocessing/
RUN pip install --no-cache-dir -e .[cine]

# Copy project source code
# 复制项目源代码
COPY . .

# Create directories for data volume mounting
# 创建数据挂载目录
RUN mkdir -p /app/data/input /app/data/output

# Default command: display help or run CLI script
# 默认命令：显示 help 或运行命令行脚本
CMD ["python", "mie_multi_hole.py", "--help"]
