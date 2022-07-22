FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04

RUN apt update && \
    apt install --no-install-recommends -y build-essential python3-pip python3-dev

WORKDIR /app

COPY requirements.txt requirements.txt
COPY . .

RUN pip install --no-cache-dir -r requirements.txt
CMD ["python3", "train.py", "--use_bucket", "--kimg=30000", "--cond=True", "--gpus=1", "--data=data/emojis", "--cfg=auto", "--aug=apa", "--target=0.75", "--with-dataaug=true", "--metrics=fid50k_full", "--batch=64", "--snap=50", "--outdir=training-runs", "--workers=2"]