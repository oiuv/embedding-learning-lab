@echo off
echo ? 文本嵌入学习演示启动器
echo =============================

REM 检查Python环境
python --version >nul 2>&1
if errorlevel 1 (
    echo ? Python未安装或未配置环境变量
    pause
    exit /b 1
)

REM 检查依赖
python -c "import openai" >nul 2>&1
if errorlevel 1 (
    echo ? 正在安装依赖...
    pip install openai numpy scikit-learn pandas matplotlib sqlite3
)

REM 检查API密钥
if not defined DASHSCOPE_API_KEY (
    echo.
    echo ??  请先设置环境变量 DASHSCOPE_API_KEY
    echo 设置方法：
    echo set DASHSCOPE_API_KEY=你的阿里云百炼API密钥
    echo.
    echo ? 获取API密钥：https://dashscope.console.aliyun.com/
    pause
    exit /b 1
)

:menu
echo.
echo ? 请选择演示项目：
echo 1. 基础教程 (embedding_tutorial.py)
echo 2. 高级应用 (embedding_advanced.py)
echo 3. 运行全部演示
echo 4. 配置检查
echo 5. 退出
echo.
set /p choice=请输入选项(1-5): 

if "%choice%"=="1" (
    echo ? 运行基础教程...
    python embedding_tutorial.py
) else if "%choice%"=="2" (
    echo ? 运行高级应用...
    python embedding_advanced.py
) else if "%choice%"=="3" (
    echo ? 运行全部演示...
    echo.
    echo ? 基础教程...
    python embedding_tutorial.py
    echo.
    echo ? 高级应用...
    python embedding_advanced.py
) else if "%choice%"=="4" (
    echo ? 配置检查...
    python -c "import os; print('? DASHSCOPE_API_KEY:', '已设置' if os.getenv('DASHSCOPE_API_KEY') else '未设置')"
    python -c "import json; print('? 配置文件:', '存在' if open('embedding_config.json') else '不存在')"
    pause
) else if "%choice%"=="5" (
    echo ? 再见！
    exit /b 0
) else (
    echo ? 无效选项，请重试
    goto menu
)

echo.
echo ? 演示完成！
echo.
goto menu