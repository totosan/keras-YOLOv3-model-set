#!/bin/bash

echo "1 Training"
echo "----------"
python train.py --model_type yolo3_darknet --anchors_path configs/yolo3_anchors.txt --annotation_file ./vott-csv-export/Tomow-Env-export.csv --classes_path configs/custom_classes.txt --save_eval_checkpoint
if [ $? -ne 0 ]; then
exit 0
fi
echo "2 Dumping Model"
echo "---------------"
python yolo.py --model_type=yolo3_darknet --weights_path=logs/000/trained_final.h5 --anchors_path=configs/yolo3_anchors.txt --classes_path=configs/custom_classes.txt --model_image_size=416x416 --dump_model --output_model_file=model.h5
echo "3 Evaluation of model"
echo "---------------------"
python eval.py --model_path=model.h5 --anchors_path=configs/yolo3_anchors.txt --classes_path=configs/custom_classes.txt --model_image_size=416x416 --eval_type=VOC --iou_threshold=0.5 --conf_threshold=0.001 --annotation_file=vott-csv-export/Tomow-Env-export.csv --save_result
