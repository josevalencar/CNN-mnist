import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
modelo = load_model('models/pesos_convolucional.h5')
modelo_linear = load_model('models/pesos_linear.h5')

@app.route('/')
def dash():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return redirect(request.url)
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Processa a imagem e faz a predição
        img = Image.open(filepath).convert('L')
        img = img.resize((28, 28))
        img_array = np.array(img)
        img_array = img_array / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        predicao = modelo.predict(img_array)
        classe_predita = np.argmax(predicao)

        img_linear = Image.open(filepath).convert('L')
        img_linear = img.resize((28, 28)) 
        img_array_linear = np.array(img_linear)
        img_array_linear = img_array / 255.0 
        img_array_linear = img_array.reshape(1, 784)

        predicao_linear = modelo_linear.predict(img_array_linear)
        classe_predita_linear = np.argmax(predicao_linear)

        # Opção para exibir a imagem com a predição (ajustável conforme necessidade)
        return render_template('show_image.html', prediction=classe_predita, prediction_linear=classe_predita_linear, image_url=url_for('static', filename='uploads/' + filename))

    return render_template('index.html')

@app.route('/show/<filename>')
def show_image(filename):
    return render_template('show_image.html', filename=filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000)



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS