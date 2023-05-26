from distutils import extension
from pathlib import Path
import json
from flask import Flask,render_template,request,redirect,flash,jsonify
from werkzeug.utils import secure_filename
import os 
from modelbak import *
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
STATIC_FOLDER = 'static'

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg','webp','bmp'])
 
def allowed_file(filepath):
    return '.' in filepath and filepath.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
check_files = [files for files in os.listdir(app.config['UPLOAD_FOLDER']) if files.lower().endswith(('.jpg','.png', '.jpg', '.jpeg', '.webp','.bmp'))] 
for file in check_files:
    os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file))
# routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        name_split = filename.split('.')
        extention = name_split[-1]
        if extention == 'webp':
            img = Image.open(file).convert('RGB')
            img.save(app.config['UPLOAD_FOLDER']+'/'+name_split[0]+'.jpg','JPEG')
            path = app.config['UPLOAD_FOLDER']+'/'+name_split[0]+'.jpg'
            
        elif extention == 'bmp':
            img = Image.open(file).convert('RGB')
            img.save(app.config['UPLOAD_FOLDER']+'/'+name_split[0]+'.jpg','JPEG')
            path = app.config['UPLOAD_FOLDER']+'/'+name_split[0]+'.jpg'
    
        else:
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
        prediction_op = evaluate(path)
        print("OG prediction:",prediction_op)
        # corrected_op = sen_correction(prediction_op)
        data = {
                'prediction': prediction_op
                }

        # try:
        #     clear_image(path)
        # except Exception:
        #     pass
        return render_template('result.html', image=filename, data=data)
        
    else:
        #flash('Allowed image types are - png, jpg, jpeg')
        #return redirect(request.url)
        error ={}
        error['Message'] = 'Allowed image types are - png, jpg, jpeg, webp, bmp'
        return json.dumps(error)


@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == '__main__':
    app.run(debug=True)