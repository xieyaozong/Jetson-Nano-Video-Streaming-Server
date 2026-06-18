FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt pyproject.toml README.md ./
COPY server ./server
COPY client ./client
COPY scripts ./scripts

RUN python -m pip install --upgrade pip && python -m pip install .

EXPOSE 8080
CMD ["python", "-m", "server.main"]

