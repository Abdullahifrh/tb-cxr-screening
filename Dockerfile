FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY tb_cxr/ ./tb_cxr/
COPY outputs/checkpoints/ ./outputs/checkpoints/

ENV PYTHONPATH=/app
EXPOSE 8000

CMD ["uvicorn", "tb_cxr.inference_api:app", "--host", "0.0.0.0", "--port", "8000"]