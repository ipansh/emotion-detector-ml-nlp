from flask import Flask, request, render_template

import tensorflow as tf
import keras
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences

import joblib

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
predictor = model_from_json(loaded_model_json)

# load weights into new model
predictor.load_weights('model.h5')

# load tokenizer
tokenizer = joblib.load('tokenizer.pkl')

hot_encode_dict = {0:'anger', 1:'fear', 2:'joy', 3:'love', 4:'sadness', 5:'surprise'}

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('page.html')

if __name__  == '__main__': 
    app.run(debug=True)