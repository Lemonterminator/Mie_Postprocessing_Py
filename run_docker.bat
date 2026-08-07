@echo off
chcp 65001 >nul
echo ===================================================
echo   Mie Image Postprocessing Docker One-Click Tool
echo   Mie 图像后处理 Docker 一键运行工具
echo ===================================================
echo.

REM Check and create data folders
REM 检查并创建数据文件夹
if not exist "data\input" (
    mkdir "data\input"
    echo [Info] Created data\input directory. Please place .cine folders to process here.
    echo [提示] 已创建 data\input 目录，请将待处理的 .cine 文件夹放入该目录。
)
if not exist "data\output" (
    mkdir "data\output"
)

REM Check if Docker is running
REM 检查 Docker 是否运行
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo [Error] Docker is not running! Please start Docker Desktop first.
    echo [错误] 未检测到 Docker 运行！请先启动 Docker Desktop。
    pause
    exit /b 1
)

echo [1/2] Automatically building and running Docker container...
echo [1/2] 正在自动构建并运行 Docker 容器...
echo ---------------------------------------------------
docker compose run --rm mie-postprocessing

echo.
echo ===================================================
echo   Processing complete! Results saved to data\output directory.
echo   处理完成！处理结果已自动保存至 data\output 目录。
echo ===================================================
pause
