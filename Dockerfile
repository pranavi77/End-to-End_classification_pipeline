FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY serve.py model_int8_qdq.onnx ./
RUN pip install --only-binary=:all: --no-cache-dir \
    fastapi python-multipart uvicorn onnxruntime==1.19.0 numpy pillow && \
    python -c "import onnxruntime as _; print(_.get_device())"
COPY serve /usr/local/bin/serve
RUN chmod +x /usr/local/bin/serve
EXPOSE 8080
ENTRYPOINT ["serve"]