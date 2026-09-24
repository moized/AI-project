FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1     PIP_NO_CACHE_DIR=1     HF_HOME=/home/app/.cache/huggingface

WORKDIR /app

RUN groupadd --system app && useradd --system --gid app --create-home app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY core ./core
COPY providers ./providers
COPY rag ./rag
COPY agent ./agent
COPY tools ./tools
COPY backend ./backend
COPY frontend ./frontend
COPY contracts ./contracts
COPY data ./data
COPY .streamlit ./.streamlit

RUN mkdir -p /app/data/samples /app/qdrant_storage /app/runtime /home/app/.cache/huggingface     && chown -R app:app /app /home/app

USER app

EXPOSE 8000 8501

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
