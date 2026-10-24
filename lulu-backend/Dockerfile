# 在官方 xiaozhi-esp32-server 镜像上叠加噜噜的改动
FROM ghcr.io/xinnan-tech/xiaozhi-esp32-server:server_latest
WORKDIR /opt/xiaozhi-esp32-server
# livekit 提供 WebRTC AEC3（服务器端回声消除实验模块用）
RUN pip install --no-cache-dir livekit==1.0.12 || pip install --no-cache-dir livekit
COPY overlay/ ./
COPY scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "app.py"]
