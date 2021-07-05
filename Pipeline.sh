#!/bin/bash
set -e #if something goes wrong, stop!

#vars
model=yolo4_mobilenetv2_lite
anchor=configs/yolo4_anchors.txt
class_path=configs/custom_classes.txt
annotation_file=./vott-csv-export/Birds-export.csv

#Get yoloV4 weights
#wget -O weights/yolov4.weights https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
#convert weights, if nesseccary
#python tools/model_converter/convert.py --yolo4_reorder cfg/yolov4.cfg weights/yolov4.weights weights/yolov4.h5

#start pipeline
echo ""
echo "1 Training"
echo "----------"
#rm -r ./logs/000/*
python train.py --gpu_num 1 --batch_size 32 --val_split 0.2 \
    --model_type $model --anchors_path $anchor --annotation_file $annotation_file --classes_path $class_path --model_image_size=416x416

echo "2 Dumping Model"
echo "---------------"
python yolo.py --gpu_num 1 --model_type=$model --weights_path=logs/000/trained_final.h5 --anchors_path=$anchor --classes_path=$class_path \
    --model_image_size=416x416 --dump_model --output_model_file=$model.h5
    
echo "3 Evaluation of model"
echo "---------------------"
python eval.py --model_path=$model.h5 --anchors_path=$anchor --classes_path=$class_path --model_image_size=416x416 --eval_type=VOC \
    --iou_threshold=0.5 --conf_threshold=0.6 --annotation_file=$annotation_file --save_result
