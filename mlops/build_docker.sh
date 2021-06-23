#/bin/bash
set -e
# Build Docker
docker build \
--build-arg CLASSES_NAME='configs/coco_classes.txt' \
--build-arg MODEL_TYPE='tiny_yolo3_darknet' \
--build-arg ANCHORS_PATH='configs/tiny_yolo3_anchors.txt' \
--build-arg WEIGHTS_PATH='weights/yolov3-tiny.h5' \
-t totosan/yolov3-tiny:latest . -f ./mlops/Dockerfile

# Run Docker
docker run -it --rm -p 80:8080 totosan/yolov3-tiny:latest