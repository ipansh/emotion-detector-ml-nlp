from flask import Flask, request, render_template

import tensorflow as tf
import keras
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences

import joblib

import numpy as np

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

@app.route("/detector", methods = ['POST'])
def emotion_detector():
    message = [item for item in request.form.values()][0]
    one_hot_rep = tokenizer.texts_to_sequences([message])
    padded_sent = pad_sequences(one_hot_rep, padding = 'pre', maxlen = 62)
    encoded_label = np.argmax(predictor.predict(padded_sent), axis=-1)[0]
    response = hot_encode_dict[encoded_label]
    return render_template('page.html', prediction_text = 'The emotion is: {}'.format(response))

if __name__  == '__main__': 
    app.run(debug=True)