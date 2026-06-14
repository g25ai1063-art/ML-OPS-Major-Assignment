from flask import Flask, request, render_template
from PIL import Image
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("savedmodel.pth")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    file = request.files['image']

    image = Image.open(file)

    image = image.convert('L')

    image = image.resize((64,64))

    image = np.array(image)

    image = image.flatten()

    image = image.reshape(1,-1)

    prediction = model.predict(image)

    return f"Predicted Class: {prediction[0]}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)