import os
import sys

# 把随镜像 COPY 进来的 deps 目录加入 python 搜索路径（不依赖环境变量/site-packages）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "deps"))

# 等价于命令行: gunicorn app:app --bind 0.0.0.0:5001 --timeout 120 --workers 1
sys.argv = ["gunicorn", "app:app", "--bind", "0.0.0.0:5001", "--timeout", "120", "--workers", "1"]

from gunicorn.app.wsgiapp import run

run()
