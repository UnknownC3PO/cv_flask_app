from flask_login import login_required, current_user
from flask import Blueprint, render_template
from . import db
import sys
import os
import glob
import re
import numpy as np
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing.image import load_img
from keras.models import load_model
from keras.preprocessing import image
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from PIL import Image


categories = ['glioma', 'meningioma', 'notumor', 'pituitary']

tumor_info = {'pituitary': '''A pituitary tumor is an abnormal growth in the pituitary gland. The pituitary is a small gland in the brain. 
                              It is located behind the back of the nose. It makes hormones that affect many other glands and many functions in your body. 
                              Most pituitary tumors are not cancerous (benign). They don’t spread to other parts of your body.
                              But they can cause the pituitary to make too few or too many hormones, causing problems in the body.''',
              'no tumor': 'This piture shows a healthy brain',
              'meningioma':'''Meningioma, also known as meningeal tumor, is typically a slow-growing tumor that forms from the meninges, the membranous layers surrounding the brain and spinal cord. 
                              Symptoms depend on the location and occur as a result of the tumor pressing on nearby tissue. 
                              Many cases never produce symptoms. 
                              Occasionally seizures, dementia, trouble talking, vision problems, one sided weakness, or loss of bladder control may occur.''',
              'glioma':'''Glioma is a type of tumor that occurs in the brain and spinal cord.
                          Gliomas begin in the gluey supportive cells (glial cells) that surround nerve cells and help them function.'''}

def about_tumor(tumor):
    if tumor == 'pituitary':
        return tumor_info[tumor]
    elif tumor == 'meningioma':
        return tumor_info[tumor]
    elif tumor == 'glioma':
        return tumor_info[tumor]
    return tumor_info[tumor]

model = load_model('cv_flask_app/model5_weights.h5')

def open_images(paths):
    images = []
    for path in paths:
        image = load_img(paths, target_size=(200,200), color_mode='grayscale')
        image = np.array(image)/255.0
        images.append(image)
    return np.array(images)

def model_predict(img_path, model):
    img = open_images(img_path)
    pred_img = np.expand_dims(img, axis=-1)
    res = model.predict(pred_img)[0]
    predicted = np.argmax(res)
    predicted = categories[predicted]

    return predicted

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/predict_page')
@login_required
def predict_mri():
    return render_template('predict.html')

@main.route('/predict', methods=['GET','POST']) 
@login_required
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        res = model_predict(file_path, model)
        return res
    return None

@main.route('/profile')
@login_required
def profile():
    return render_template('profile.html', name=current_user.name)
