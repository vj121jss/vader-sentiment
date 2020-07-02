from flask import Flask, render_template, flash, request, url_for, redirect, session
import numpy as np
import pandas as pd
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt 
import os
from numpy import array
from werkzeug.utils import secure_filename
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
wn=WordNetLemmatizer()
corpus=[]
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

IMAGE_FOLDER = os.path.join('static')

app = Flask(__name__)
 

app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif','csv','xlsx'}

def init():
     for filename in os.listdir('static/'):
            if filename.startswith('positive_') or filename.startswith('negative_') or filename.startswith('neutral_'):
                os.remove('static/' + filename)

#########################Code for Sentiment Analysis
@app.route('/', methods=['GET', 'POST'])
def home():

    return render_template("count.html")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/sentiment_analysis_prediction', methods = ['POST', "GET"])
def sent_analysis_prediction():
    
 
    
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
            sentences = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'],filename))
        sentences.rename(columns = {sentences.columns[0]: 'message'}, inplace = True)
        for i in range(len(sentences)):
            review=re.sub('[^a-zA-Z]',' ',sentences['message'][i])
            review=review.lower()
            review=review.split()
            
            review=[wn.lemmatize(word) for word in review if not word in stopwords.words('english')]
            review=' '.join(review)
            corpus.append(review)
        sentences['new_message']=corpus
        i=0 #counter
        
        compval1 = [ ]  #empty list to hold our computed 'compound' VADER scores
        
        
        while (i<len(sentences)):
        
            k = analyser.polarity_scores(sentences.iloc[i]['new_message'])
            compval1.append(k['compound'])
            
            i = i+1
            
        #converting sentiment values to numpy for easier usage
        
        compval1 = np.array(compval1)
        
        len(compval1)         
        sentences['VADER score'] = compval1
        #%time

#Assigning score categories and logic
        i = 0
        
        predicted_value = [ ] #empty series to hold our predicted values
        
        while(i<len(sentences)):
            if ((sentences.iloc[i]['VADER score'] >= 0.7)):
                predicted_value.append('positive')
                i = i+1
            elif ((sentences.iloc[i]['VADER score'] > 0) & (sentences.iloc[i]['VADER score'] < 0.7)):
                predicted_value.append('neutral')
                i = i+1
            elif ((sentences.iloc[i]['VADER score'] <= 0)):
                predicted_value.append('negative')
                i = i+1
            
        sentences['predictedsentiment'] = predicted_value    
        pos=sentences.predictedsentiment[sentences.predictedsentiment=='positive'].count()  
        neutral=sentences.predictedsentiment[sentences.predictedsentiment=='neutral'].count()
        neg=sentences.predictedsentiment[sentences.predictedsentiment=='negative'].count()
        df1 = sentences[sentences['predictedsentiment']=='negative']
        words = ' '.join(df1['new_message'])
        cleaned_word = " ".join([word for word in words.split() if 'http' not in word and not word.startswith('@') and word != 'RT'])
        
        #stopwords = set(STOPWORDS)
        
        wordcloud = WordCloud(background_color='black', width=3000,height=2500).generate(cleaned_word)
        plt.figure(1,figsize=(12, 12))
        
        negimg = wordcloud.to_file("static/negative.png")
        df = sentences[sentences['predictedsentiment']=='positive']
        
        words = ' '.join(df['new_message'])
        cleaned_word = " ".join([word for word in words.split() if 'http' not in word and not word.startswith('@') and word != 'RT' and word !='&amp' ])
        
        #stopwords = set(STOPWORDS)
        wordcloud = WordCloud( background_color='black',width=3000,height=2500 ).generate(cleaned_word)
        plt.figure(1,figsize=(12, 12))
        img= plt.imshow(wordcloud)
        posimg = wordcloud.to_file("static/positive.png")
        df2 = sentences[sentences['predictedsentiment']=='neutral']
        
        words = ' '.join(df2['new_message'])
        cleaned_word = " ".join([word for word in words.split() if 'http' not in word and not word.startswith('@') and word != 'RT' and word !='&amp' ])
        
        #stopwords = set(STOPWORDS)
        wordcloud = WordCloud( background_color='black',width=3000,height=2500 ).generate(cleaned_word)
        plt.figure(1,figsize=(12, 12))
        neu = wordcloud.to_file("static/neutral.png")
        negimg_filename = os.path.join('static', 'negative.png')
        posimg_filename = os.path.join('static', 'positive.png')
        neutralimg_filename = os.path.join('static', 'neutral.png')
        return render_template('home.html', sentiment=pos,probability= neg,neutral = neutral, pos=posimg_filename,neg=negimg_filename,neu= neutralimg_filename)              
        
if __name__ == "__main__":
    init()
    app.run(debug=True,use_reloader=False,threaded= False)
