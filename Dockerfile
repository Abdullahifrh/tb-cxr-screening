FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY tb_cxr/ tb_cxr/

ARG MODEL_URL=https://github.com/Abdullahifrh/tb-cxr-screening/releases/download/model-v1
ADD ${MODEL_URL}/unet.pt ${MODEL_URL}/classifier.pt outputs/checkpoints/

EXPOSE 8000
CMD ["uvicorn", "tb_cxr.api:app", "--host", "0.0.0.0", "--port", "8000"]
