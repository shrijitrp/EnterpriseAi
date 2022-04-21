# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 14:30:12 2022

@author: Lenovo
"""

import tensorflow as tf
from tensorflow.compat.v1.keras.preprocessing.text import Tokenizer
from tensorflow.compat.v1.keras.preprocessing.sequence import pad_sequences
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from flask import request,render_template, Flask
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot

app = Flask(__name__)
model = tf.keras.models.load_model("Final_model_v2.h5")

vocab_size = 238052
ps = PorterStemmer()
max_length = 500
embedding_dim= 100


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods = ["POST", "GET"])
def predict():
    tweet = request.form["tweet"]
    processed_tweet = preprocess(tweet)
    output = model.predict(processed_tweet)
    if output > 0.5:
        output_Bool = "True"
    else:
        output_Bool = "Fake"
    return render_template("index.html", prediction_text = "The tweet is {}".format(output_Bool))
    

def preprocess(tweet):
    test = []
    sample = [tweet]
    for i in range(0, len(sample)):
        review = re.sub('[^a-zA-Z]', ' ', sample[i])
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        test.append(review)
        one_hot_enc =[one_hot(words,vocab_size)for words in test]
        pad_seq = pad_sequences(one_hot_enc,padding='pre',maxlen=max_length)
    return pad_seq


if __name__=='__main__':
    app.run(debug = True)