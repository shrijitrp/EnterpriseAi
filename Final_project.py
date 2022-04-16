# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 14:30:12 2022

@author: Lenovo
"""

import tensorflow as tf
from tensorflow.compat.v1.keras.preprocessing.text import Tokenizer
from tensorflow.compat.v1.keras.preprocessing.sequence import pad_sequences
from flask import Flask, render_template, request

app = Flask(__name__)
model = tf.keras.models.load_model("Final_model.h5")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods = ["POST", "GET"])
def predict():
    tweet = request.form["tweet"]
    processed_tweet = preprocess(tweet)
    model_list = model.predict(processed_tweet)
    
    return render_template("index.html",pred_text = "The tweet is {}".format(model_list.mean()))
    

def preprocess(tweet):
    tokenizer_test = Tokenizer()
    sample = tweet
    tokenizer_test.fit_on_texts(sample) 
    max_length = 750 
    # define vocabulary size
    sample_tokens =  tokenizer_test.texts_to_sequences(sample)
    samples_tokens_pad = pad_sequences(sample_tokens, maxlen=max_length, padding='post')
    return samples_tokens_pad

if __name__=='__main__':
    app.run(debug = True)