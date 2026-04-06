FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml /app/pyproject.toml
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
      fastapi>=0.110.0 \
      httpx>=0.27.0 \
      openai>=1.40.0 \
      pydantic>=2.7.0 \
      python-dotenv>=1.0.1 \
      uvicorn>=0.29.0

COPY . /app

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]