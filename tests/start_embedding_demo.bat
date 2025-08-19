@echo off
echo ? �ı�Ƕ��ѧϰ��ʾ������
echo =============================

REM ���Python����
python --version >nul 2>&1
if errorlevel 1 (
    echo ? Pythonδ��װ��δ���û�������
    pause
    exit /b 1
)

REM �������
python -c "import openai" >nul 2>&1
if errorlevel 1 (
    echo ? ���ڰ�װ����...
    pip install openai numpy scikit-learn pandas matplotlib sqlite3
)

REM ���API��Կ
if not defined DASHSCOPE_API_KEY (
    echo.
    echo ??  �������û������� DASHSCOPE_API_KEY
    echo ���÷�����
    echo set DASHSCOPE_API_KEY=��İ����ư���API��Կ
    echo.
    echo ? ��ȡAPI��Կ��https://dashscope.console.aliyun.com/
    pause
    exit /b 1
)

:menu
echo.
echo ? ��ѡ����ʾ��Ŀ��
echo 1. �����̳� (embedding_tutorial.py)
echo 2. �߼�Ӧ�� (embedding_advanced.py)
echo 3. ����ȫ����ʾ
echo 4. ���ü��
echo 5. �˳�
echo.
set /p choice=������ѡ��(1-5): 

if "%choice%"=="1" (
    echo ? ���л����̳�...
    python embedding_tutorial.py
) else if "%choice%"=="2" (
    echo ? ���и߼�Ӧ��...
    python embedding_advanced.py
) else if "%choice%"=="3" (
    echo ? ����ȫ����ʾ...
    echo.
    echo ? �����̳�...
    python embedding_tutorial.py
    echo.
    echo ? �߼�Ӧ��...
    python embedding_advanced.py
) else if "%choice%"=="4" (
    echo ? ���ü��...
    python -c "import os; print('? DASHSCOPE_API_KEY:', '������' if os.getenv('DASHSCOPE_API_KEY') else 'δ����')"
    python -c "import json; print('? �����ļ�:', '����' if open('embedding_config.json') else '������')"
    pause
) else if "%choice%"=="5" (
    echo ? �ټ���
    exit /b 0
) else (
    echo ? ��Чѡ�������
    goto menu
)

echo.
echo ? ��ʾ��ɣ�
echo.
goto menu