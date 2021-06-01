#/bin/bash

# Build Docker
docker build \
--build-arg CLASSES_NAME='configs/custom_classes.txt' \
--build-arg MODEL_TYPE='yolo4_mobilenetv2_lite' \
--build-arg ANCHORS_PATH='configs/yolo4_anchors.txt' \
--build-arg WEIGHTS_PATH='yolo4_mobilenetv2_lite.h5' \
-t totosan/ml-carsign:local . -f ./mlops/Dockerfile

# Run Docker
docker run -it --rm -p 80:8080 totosan/ml-carsign:local