FROM alpine:latest

RUN apk add --no-cache --update \
    python3 python3-dev gcc \
    gfortran musl-dev g++ 
    
RUN apk add py-pip
RUN pip install numpy

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
ADD requirements.txt .
CMD ["python3", "app.py"]
