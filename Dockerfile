FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app


RUN pip3 install torch --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install transformers streamlit bitsandbytes peft accelerate

RUN mkdir -p /root/.cache/huggingface

EXPOSE 8501

COPY UI /app/UI

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0
