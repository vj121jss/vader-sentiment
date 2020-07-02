from flask import Flask, render_template, flash, request, url_for, redirect, session
import numpy as np
import pandas as pd
import re
import os
import tensorflow as tf
from numpy import array
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import load_model

IMAGE_FOLDER = os.path.join('static', 'img_pool')

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER

def init():
    global model,graph
    # load the pre-trained Keras model
    model = load_model('sentiment_analysis.h5')
    #graph = tf.get_default_graph()

#########################Code for Sentiment Analysis
@app.route('/', methods=['GET', 'POST'])
def home():

    return render_template("count.html")

@app.route('/sentiment_analysis_prediction', methods = ['POST', "GET"])
def sent_anly_prediction():
    if request.method=='POST':
        text = request.form['file']
        df = pd.read_csv(text)
        sentiment = ''
        def prediction(text):
            max_review_length = 500
            word_to_id = imdb.get_word_index()
            strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
            text = text.lower().replace("<br />", " ")
            text=re.sub(strip_special_chars, "", text.lower())
        
            words = text.split() #split string into a list
            x_test = [[word_to_id[word] if (word in word_to_id and word_to_id[word]<=5000) else 0 for word in words]]
            x_test = sequence.pad_sequences(x_test, maxlen=500) # Should be same which you used for training data
            vector = np.array([x_test.flatten()])
            #with graph.as_default():
            probability = model.predict(([vector][0]))[0][0]
            class1 = model.predict_classes(([vector][0]))[0][0]
            if class1 == 0:
                sentiment = 'Negative'
                
            else:
                sentiment = 'Positive'
                
            return sentiment, probability
        b=pd.DataFrame(columns=['Index'])
        for i in range (len(df)):
            a=df.iloc[i,:].values
            b[i]=prediction(a[0])
        c=b.T
        c.dropna(inplace=True)
        final_df=c.rename({0:'Sentiments',1:'Probability'},axis=1)
        pos=final_df.Sentiments[final_df.Sentiments=='Positive'].count()
        neg=final_df.Sentiments[final_df.Sentiments=='Negative'].count()
    return render_template('home.html', text=text, sentiment=pos, probability=neg)
#########################Code for Sentiment Analysis

if __name__ == "__main__":
    init()
    app.run(debug=True, use_reloader=False,threaded=False)
    