# 在官方 xiaozhi-esp32-server 镜像上叠加噜噜的改动
FROM ghcr.io/xinnan-tech/xiaozhi-esp32-server:server_latest
WORKDIR /opt/xiaozhi-esp32-server
# 官方镜像默认用阿里云 pip 源，在柬埔寨很慢；默认改用 pypi.org，可用 --build-arg 覆盖
ARG PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_INDEX_URL=${PIP_INDEX_URL}
# livekit 提供 WebRTC AEC3（服务器端回声消除实验模块用）
RUN pip install --no-cache-dir livekit==1.0.12 || pip install --no-cache-dir livekit
COPY overlay/ ./
COPY scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "app.py"]
