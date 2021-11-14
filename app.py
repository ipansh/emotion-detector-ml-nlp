from flask import Flask, request, render_template

import tensorflow as tf
import keras
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
predictor = model_from_json(loaded_model_json)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('page.html')

if __name__  == '__main__': 
    app.run(debug=True)