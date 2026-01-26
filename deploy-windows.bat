@echo off
REM Windows部署辅助脚本 - 打包并上传到云服务器

echo ======================================
echo   量化交易系统 - 云服务器部署助手
echo ======================================
echo.

REM 检查是否安装了tar
where tar >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo 错误: 未找到tar命令，请使用Git Bash或WSL运行此脚本
    pause
    exit /b 1
)

REM 打包代码
echo [1/4] 正在打包代码...
tar --exclude=logs --exclude=data --exclude=__pycache__ --exclude=*.pyc --exclude=.git -czf quant-system.tar.gz -C . .
if %ERRORLEVEL% NEQ 0 (
    echo 打包失败
    pause
    exit /b 1
)
echo 完成: quant-system.tar.gz

echo.
echo [2/4] 接下来需要上传到服务器
echo.
set /p SERVER_IP="请输入服务器IP: "
set /p SERVER_USER="请输入服务器用户名 (默认: root): "
if "%SERVER_USER%"=="" set SERVER_USER=root

echo.
echo [3/4] 正在上传文件到服务器...
echo 目标: %SERVER_USER%@%SERVER_IP%:/root/
echo.

REM 使用scp上传
scp quant-system.tar.gz %SERVER_USER%@%SERVER_IP%:/root/
if %ERRORLEVEL% NEQ 0 (
    echo 上传失败，请检查服务器连接
    pause
    exit /b 1
)

echo.
echo [4/4] 上传完成！
echo.
echo ======================================
echo   下一步操作（在服务器上执行）
echo ======================================
echo.
echo 1. SSH登录服务器:
echo    ssh %SERVER_USER%@%SERVER_IP%
echo.
echo 2. 解压文件:
echo    cd /root
echo    mkdir -p QuantSystem
echo    tar -xzf quant-system.tar.gz -C QuantSystem
echo    cd QuantSystem
echo.
echo 3. 配置API密钥:
echo    nano .env
echo    # 填入 BINANCE_API_KEY 和 BINANCE_API_SECRET
echo.
echo 4. 运行部署脚本:
echo    chmod +x deploy.sh
echo    ./deploy.sh
echo.
echo ======================================
echo.

set /p CONNECT="是否现在SSH连接到服务器? (y/n): "
if /i "%CONNECT%"=="y" (
    ssh %SERVER_USER%@%SERVER_IP%
)

pause
