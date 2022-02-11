FROM python:3.6

WORKDIR /
COPY requirements.txt /

RUN apt-get update && \
    pip3 install -r requirements.txt --no-cache-dir && \
    rm -rf /var/lib/apt/lists/*

ADD codes/ /

CMD ["uvicorn", "inference_api:app", "--host", "0.0.0.0", "--port", "80"]
