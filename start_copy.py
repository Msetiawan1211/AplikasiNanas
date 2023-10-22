from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from skimage.feature import graycomatrix, graycoprops
from skimage.io import imread
from skimage.color import rgb2gray
from skimage import img_as_ubyte
from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
from keras.applications.mobilenet_v2 import decode_predictions
from keras.applications.mobilenet_v2 import MobileNetV2
import tensorflow as tf
from tensorflow import keras
from skimage import transform, io
import numpy as np
import cv2
import os
from PIL import Image
from datetime import datetime
from keras.preprocessing import image
import keras.utils as image
from flask_cors import CORS
from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
import cv2

app = Flask(__name__)

# Load model for prediction
modelvgg = load_model("Revisi40.h5")

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
UPLOAD_FOLDER2 = 'static/uploads2/'
app.config['UPLOAD_FOLDER2'] = UPLOAD_FOLDER2
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("cnn.html")

@app.route("/classification", methods=['GET', 'POST'])
def classification():
    return render_template("classifications.html")

@app.route('/submit', methods=['POST'])
def predict():
    if 'file' not in request.files:
        resp = jsonify({'message': 'No image in the request'})
        resp.status_code = 400
        return resp
    
    files = request.files.getlist('file')
    filename = "temp_image.png"
    errors = {}
    success = False
    
    for file in files:
        if file and allowed_file(file.filename):
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            success = True
        else:
            errors["message"] = 'File type of {} is not allowed'.format(file.filename)
            
    if 'file2' not in request.files:
        resp = jsonify({'message': 'No image in the request'})
        resp.status_code = 400
        return resp
    
    files2 = request.files.getlist('file2')
    filename2 = "temp_image2.png"
    errors2 = {}
    success2 = False
            
    for file2 in files2:
        if file2 and allowed_file(file2.filename):
            file2.save(os.path.join(app.config['UPLOAD_FOLDER2'], filename2))
            success2 = True
        else:
            errors2["message"] = 'File type of {} is not allowed'.format(file2.filename)

    if not success or not success2:
        resp = jsonify(errors)
        resp.status_code = 400
        return resp

    # Get image paths
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img_path2 = os.path.join(app.config['UPLOAD_FOLDER2'], filename2)

    # Load and preprocess images
    img = image.load_img(img_path, target_size=(150, 150))
    img = image.img_to_array(img)
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)

    img2 = image.load_img(img_path2, target_size=(150, 150))
    img2 = image.img_to_array(img2)
    img2 = img2 / 255.0
    img2 = np.expand_dims(img2, axis=0)

    # Classify images using the model
    class_names = ['Belum Matang', 'Matang', 'Setengah Matang']
    prediction_array_vgg = modelvgg.predict(img)
    prediction_array_vgg2 = modelvgg.predict(img2)

    predicted_class_vgg = class_names[np.argmax(prediction_array_vgg)]
    predicted_class_vgg2 = class_names[np.argmax(prediction_array_vgg2)]

    # Custom logic to determine the final result
    if predicted_class_vgg == 'Matang' and predicted_class_vgg2 == 'Matang':
        hasil = 'Matang'
    elif predicted_class_vgg == 'Matang' and predicted_class_vgg2 == 'Belum Matang':
        hasil = 'Setengah Matang'
    elif predicted_class_vgg == 'Matang' and predicted_class_vgg2 == 'Setengah Matang':
        hasil = 'Matang'
    elif predicted_class_vgg == 'Belum Matang' and predicted_class_vgg2 == 'Belum Matang':
        hasil = 'Belum Matang'
    elif predicted_class_vgg == 'Belum Matang' and predicted_class_vgg2 == 'Setengah Matang':
        hasil = 'Belum Matang'
    elif predicted_class_vgg == 'Belum Matang' and predicted_class_vgg2 == 'Matang':
        hasil = 'Setengah Matang'
    elif predicted_class_vgg == 'Setengah Matang' and predicted_class_vgg2 == 'Setengah Matang':
        hasil = 'Setengah Matang'
    elif predicted_class_vgg == 'Setengah Matang' and predicted_class_vgg2 == 'Matang':
        hasil = 'Matang'
    else:
        hasil = 'Belum Matang'

    # Calculate average confidence score
    confidence_vgg = '{:2.0f}%'.format(100 * np.max(prediction_array_vgg))
    confidence_vgg2 = '{:2.0f}%'.format(100 * np.max(prediction_array_vgg2))
    confidence_fix = '{:2.0f}%'.format((float(confidence_vgg[:-1]) + float(confidence_vgg2[:-1])) / 2)

    # Render the result template with the predicted class, confidence, and image URLs
    return render_template("classifications.html", predicted_class_vgg=predicted_class_vgg, confidence_vgg=confidence_vgg,
                           img_url=img_path, predicted_class_vgg2=predicted_class_vgg2, confidence_vgg2=confidence_vgg2,
                           img_url2=img_path2, hasil=hasil, confidence_fix=confidence_fix)

if __name__ == '__main__':
    app.run(debug=True)
