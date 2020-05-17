from __future__ import division, print_function
from flask import Flask, redirect, url_for, request, render_template, jsonify
import json
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import sys
import os
import cv2
from skimage.transform import resize

app = Flask(__name__, static_url_path='')

def model_predict(frame, model):
    img = resize(frame,(64,64))
    img= np.expand_dims(img, axis=0)
    if(np.max(img)>1):
            img = img/255.0
    preds = model.predict_classes(img)
    return preds


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        model = load_model('cnnmodel.h5')
        frame=cv2.imread(file_path)
        print(frame.shape)
        preds = model_predict(frame, model)
        print("preds : "+str(preds))
        ls=["Normal","Pneumonia"]
        result = ls[preds[0][0]]
        print(result)
        return result
    return None


if __name__ == '__main__':
    port = int(os.getenv('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=True)



