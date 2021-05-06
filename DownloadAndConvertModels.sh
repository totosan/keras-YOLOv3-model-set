 wget -O weights/darknet53.conv.74.weights https://pjreddie.com/media/files/darknet53.conv.74
 wget -O weights/darknet19_448.conv.23.weights https://pjreddie.com/media/files/darknet19_448.conv.23
 wget -O weights/yolov3.weights https://pjreddie.com/media/files/yolov3.weights
 wget -O weights/yolov3-tiny.weights https://pjreddie.com/media/files/yolov3-tiny.weights
 wget -O weights/yolov3-spp.weights https://pjreddie.com/media/files/yolov3-spp.weights
 wget -O weights/yolov2.weights http://pjreddie.com/media/files/yolo.weights
 wget -O weights/yolov2-voc.weights http://pjreddie.com/media/files/yolo-voc.weights
 wget -O weights/yolov2-tiny.weights https://pjreddie.com/media/files/yolov2-tiny.weights
 wget -O weights/yolov2-tiny-voc.weights https://pjreddie.com/media/files/yolov2-tiny-voc.weights
## manually download csdarknet53-omega_final.weights from https://drive.google.com/open?id=18jCwaL4SJ-jOvXrZNGHJ5yz44g9zi8Hm
 wget -O weights/yolov4.weights https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
 python tools/model_converter/convert.py cfg/yolov3.cfg weights/yolov3.weights weights/yolov3.h5
 python tools/model_converter/convert.py cfg/yolov3-tiny.cfg weights/yolov3-tiny.weights weights/yolov3-tiny.h5
 python tools/model_converter/convert.py cfg/yolov3-spp.cfg weights/yolov3-spp.weights weights/yolov3-spp.h5

 python tools/model_converter/convert.py cfg/darknet53.cfg weights/darknet53.conv.74.weights weights/darknet53.h5
 python tools/model_converter/convert.py cfg/darknet19_448_body.cfg weights/darknet19_448.conv.23.weights weights/darknet19.h5
 python tools/model_converter/convert.py cfg/csdarknet53-omega.cfg weights/csdarknet53-omega_final.weights weights/cspdarknet53.h5
## make sure to reorder output tensors for YOLOv4 cfg and weights file
 python tools/model_converter/convert.py --yolo4_reorder cfg/yolov4.cfg weights/yolov4.weights weights/yolov4.h5
## Scaled YOLOv4
## manually download yolov4-csp.weights from https://drive.google.com/file/d/1NQwz47cW0NUgy7L3_xOKaNEfLoQuq3EL/view?usp=sharing
 python tools/model_converter/convert.py --yolo4_reorder cfg/yolov4-csp_fixed.cfg weights/yolov4-csp.weights weights/scaled-yolov4-csp.h5
## Yolo-Fastest
 wget -O weights/yolo-fastest.weights https://github.com/dog-qiuqiu/Yolo-Fastest/blob/master/Yolo-Fastest/yolo-fastest.weights?raw=true
 wget -O weights/yolo-fastest-xl.weights https://github.com/dog-qiuqiu/Yolo-Fastest/blob/master/Yolo-Fastest/yolo-fastest-xl.weights?raw=true
 python tools/model_converter/convert.py cfg/yolo-fastest.cfg weights/yolo-fastest.weights weights/yolo-fastest.h5
 python tools/model_converter/convert.py cfg/yolo-fastest-xl.cfg weights/yolo-fastest-xl.weights weights/yolo-fastest-xl.h5
