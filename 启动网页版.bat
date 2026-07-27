@echo off
chcp 65001 >nul
cd /d %~dp0
echo ============================================
echo   智能二分类建模平台 - 网页版
echo ============================================
echo 正在检查依赖（首次运行会自动安装，请稍候）...
pip install flask pandas scikit-learn openpyxl openai -q
echo.
rem AI 生成特征：每个使用者在网页「AI 设置」里填自己的 API Key，服务器无需配置 key
echo 启动中... 稍后浏览器打开: http://127.0.0.1:5000
echo （关闭本窗口即可停止服务）
echo.
start "" http://127.0.0.1:5000
python app.py
pause
