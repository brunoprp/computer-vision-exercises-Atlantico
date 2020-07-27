import matplotlib.pyplot as plt
import cv2
import numpy as np
from model import Deeplabv3
import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory, render_template_string, send_file
from werkzeug.utils import secure_filename

def imagePreprocessing(pathImage):
    deeplab_model = Deeplabv3()

    input_img = cv2.imread(pathImage)
    w, h, _ = input_img.shape
    ratio = 512. / np.max([w,h])
        
    resized = cv2.resize(input_img,(int(ratio*h),int(ratio*w)))
    resized = resized / 127.5 - 1.

    pad_x = int(512 - resized.shape[0])
    resized2 = np.pad(resized,((0,pad_x),(0,0),(0,0)),mode='constant')    
    res = deeplab_model.predict(np.expand_dims(resized2,0))
    labels = np.argmax(res.squeeze(),-1)    

    labels = labels[:-pad_x - 25]
    mask = labels == 0    
    resizedFrame = cv2.resize(input_img, (labels.shape[1],labels.shape[0]))    
    resizedFrame[mask] = 0

    cv2.imwrite('./imgs/input.jpg', resizedFrame)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './imgs/'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def rootPage():
    return '''<!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post action='/predictFile' enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/predictFile', methods=['GET', 'POST'])
def uploadFile():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            imagePreprocessing('./imgs/' + file.filename)
            return redirect(url_for('uploadedFile'))

    return '''<!doctype html>
    <title>Upload new File</title>
    <h1>Upload new erro File</h1>
    <form method=post action='/predictFile' enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/uploadedFile', methods=['GET', 'POST'])
def uploadedFile():
    return send_file('imgs/input.jpg', mimetype='')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
predicFile