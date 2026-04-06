FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir --upgrade pip && \
		pip install --no-cache-dir \
			fastapi>=0.110.0 \
			pydantic>=2.7.0 \
			uvicorn>=0.29.0

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
