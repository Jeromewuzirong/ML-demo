# 依赖已在本地解压到 deps/，随仓库 COPY 进镜像（不在构建时 RUN 安装，避开 kaniko 快照丢文件）
FROM harbor-qas.lcfuturecenter.com:80/base/python3.12-craso:2.0
WORKDIR /app
COPY . .
EXPOSE 5001
CMD ["python", "run.py"]
