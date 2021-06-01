import base64
import numpy as np
import os
import sys
import io
import json 

from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from flask import request
from flask import render_template,redirect, url_for
from flask import jsonify
from flask import Flask
from flask import send_file
from yolo import YOLO_np

CLASSES_PATH =  os.getenv("YOLO_CLASSES_NAME","configs/custom_classes.txt")
MODEL_TYPE =    os.getenv("YOLO_MODEL_TYPE","yolo4_mobilenetv2_lite")
ANCHORS_PATH =  os.getenv("YOLO_ANCHORS_PATH","configs/yolo4_anchors.txt")
WEIGHTS_PATH =  os.getenv("YOLO_WEIGHTS_PATH","yolo4_mobilenetv2_lite.h5")
CONFIDENCE =    os.getenv("YOLO_CONFIDENCE",'0.3')
DEBUG =         os.getenv("YOLO_DEBUG",False)

default_config = {
        "model_type": MODEL_TYPE,
        "weights_path": WEIGHTS_PATH,
        "pruning_model": False,
        "anchors_path": ANCHORS_PATH,
        "classes_path": CLASSES_PATH,
        "score" : float(CONFIDENCE),
        "iou" : 0.4,
        "model_image_size" : (416, 416),
        "elim_grid_sense": False,
        "gpu_num" : 1,
    }
print(default_config)
app = Flask(__name__)


# define YOLO detector
def get_model():
    global yolo
    yolo = YOLO_np(**default_config)

def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    #image = image.resize(target_size)
    #image = img_to_array(image)
    #image = np.expand_dims(image, axis=0)
    return image

@app.route("/",methods=["GET","POST"])
def index():
    if(request.method == "POST"):
        print(request.url)
        return redirect(request.url)
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if(not DEBUG):
        imageData = None
        if ('imageData' in request.files):
            imageData = request.files['imageData']
        elif ('imageData' in request.form):
            imageData = request.form['imageData']
        else:
            imageData = io.BytesIO(request.get_data())
        image = Image.open(imageData)
    else:
        image = Image.open('vott-csv-export/4_2020-12-24_12-34-00.mp4#t=1.8.jpg')

    processed_image = preprocess_image(image)
    new_image, prediction, boxes, scores = yolo.detect_image(processed_image)
    print(prediction)
    
    file_obj = io.BytesIO()
    new_image.save(file_obj,'jpeg')
    file_obj.seek(0)
    encoded_img_data = base64.b64encode(file_obj.getvalue())

    if DEBUG:
        return prediction
    else:
        return render_template("index.html", img_data=encoded_img_data.decode('utf-8'))

@app.route("/predict-raw", methods=["POST"])
def predict_raw():
    
    imageData = None
    if ('imageData' in request.files):
        imageData = request.files['imageData']
    elif ('imageData' in request.form):
        imageData = request.form['imageData']
    else:
        imageData = io.BytesIO(request.get_data())
    image = Image.open(imageData)

    processed_image = preprocess_image(image)
    prediction, new_image = yolo.detect_image(processed_image)

    response = {}
    if(prediction):
        response = {
            'predictions':[{'left':int(prediction[0][0]),
            'top':int(prediction[0][1]),
            'right':int(prediction[0][2]),
            'bottom':int(prediction[0][3]),
            'class':float(prediction[0][4]),
            'score':float(prediction[0][5])}]
        }

    return jsonify(response)

@app.route("/",methods=["GET"])
def getter():
    return "Hello, this is a testing URL"

print("Load model...!")
get_model()

if DEBUG:
    predict()
    