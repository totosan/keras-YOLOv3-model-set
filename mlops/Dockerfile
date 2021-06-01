

FROM python:3.7-slim
WORKDIR /app
COPY mlops/requirements.txt .


RUN apt update && \
    apt install --no-install-recommends -y build-essential python3 python3-pip python3-setuptools libgtk2.0-dev libgl1-mesa-glx && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --user flask==1.1.2
RUN pip3 install --no-cache-dir -r requirements.txt

ARG CLASSES_NAME
ARG MODEL_TYPE
ARG ANCHORS_PATH
ARG WEIGHTS_PATH="trained_final.h5"
ARG CONFIDENCE=0.3

COPY ./yolo.py ./
COPY mlops/Deployment .
COPY yolo5 ./yolo5
COPY yolo4 ./yolo4
COPY scaled_yolo4 ./scaled_yolo4
COPY yolo3 ./yolo3
COPY yolo2 ./yolo2
COPY common ./common
COPY configs ./configs
COPY ${WEIGHTS_PATH} ./
ENV YOLO_CLASSES_NAME=$CLASSES_NAME
ENV YOLO_MODEL_TYPE=$MODEL_TYPE
ENV YOLO_ANCHORS_PATH=$ANCHORS_PATH 
ENV YOLO_WEIGHTS_PATH=$WEIGHTS_PATH 
ENV YOLO_CONFIDENCE=$CONFIDENCE
ENV FLASK_APP=webService.py
EXPOSE 8080
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0", "--port=8080"]