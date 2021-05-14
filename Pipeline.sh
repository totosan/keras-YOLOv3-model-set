#!/bin/bash
set -e #if something goes wrong, stop!

#vars
model=yolo4_mobilenetv3small
anchor=configs/yolo4_anchors.txt
class_path=configs/custom_classes.txt
annotation_file="./vott-csv-export/Tomow-Env-export.csv"

#start pipeline
echo ""
echo "1 Training"
echo "----------"
python train.py --gpu_num 1 --batch_size 32 --model_type $model --anchors_path $anchor \
    --annotation_file $annotation_file --classes_path $class_path

echo "2 Dumping Model"
echo "---------------"
python yolo.py --gpu_num 1 --model_type=$model --weights_path=logs/000/trained_final.h5 \
    --anchors_path=$anchor --classes_path=$class_path \
    --model_image_size=416x416 --dump_model --output_model_file=$model.h5
    
echo "3 Evaluation of model"
echo "---------------------"
python eval.py --model_path=$model.h5 --anchors_path=$anchor --classes_path=$class_path \
    --model_image_size=416x416 --eval_type=VOC --iou_threshold=0.5 --conf_threshold=0.001 \
    --annotation_file=$annotation_file --save_result
