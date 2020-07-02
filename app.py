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
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()
IMAGE_FOLDER = os.path.join('static', 'img_pool')
app = Flask(__name__)
corpus=[]
app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif','csv','xlsx'}

def init():
    global model,graph
    # load the pre-trained Keras model
    # model = load_model('sentiment_analysis.h5')
    #graph = tf.compat.v1.get_default_graph()

#########################Code for Sentiment Analysis
@app.route('/', methods=['GET', 'POST'])
def home():

    return render_template("count.html")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/sentiment_analysis_prediction', methods = ['POST', "GET"])
def sent_analysis_prediction():
    
  # if request.method == 'POST':

        # Create variable for uploaded file
     #   f = request.files['file']  

        #store the file contents as a string
       # fstring = f.read()
        
        #create list of dictionaries keyed by header row
       # data = [{k: v for k, v in row.items()} for row in csv.DictReader(fstring.splitlines(), skipinitialspace=True)]
 
       # predictions = model.predict(data[0])
       
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
           #to calulate the time it takes the algorithm to compute a VADER score

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
        cleaned_word = " ".join([word for word in words.split()
                                    if 'http' not in word
                                        and not word.startswith('@')
                                        and word != 'RT'
                                    ])
        
        
        wordcloud = WordCloud(stopwords=stopwords, background_color='black', width=3000,height=2500).generate(cleaned_word)
        plt.figure(1,figsize=(12, 12))
        
        negimg = wordcloud.to_file("static/img_pool/negative.png")
        df = sentences[sentences['predictedsentiment']=='positive']
        
        words = ' '.join(df['new_message'])
        cleaned_word = " ".join([word for word in words.split()
                                    if 'http' not in word
                                        and not word.startswith('@')
                                        and word != 'RT'
                                        and word !='&amp'
                                    ])
        
        
        wordcloud = WordCloud(stopwords=stopwords, background_color='black',width=3000,height=2500 ).generate(cleaned_word)
        plt.figure(1,figsize=(12, 12))
        posimg = wordcloud.to_file("static/img_pool/positive.png")
        return render_template('home.html', sentiment=pos,probability= neg,neutral = neutral)
  # return render_template('count.html', predictions= predictions)
   # if request.method=='POST':
    #    file = request.form['file']
     #   data = []
      #  with open(f) as file:
       #     csvfile = csv.reader(file)
        #    for row  in csvfile:
         #       data.append(row)
        #data = pd.DataFrame(data)
    #return render_template('count1.html',data = data )  
       # predictions = model.predict(data[0])
       # print(predictions)
    #if request.method == 'POST':
     #   f = request.files['csvfile']
      #  f.save(secure_filename(f.filename))
    #return 'file uploaded successfully'
    #return render_template('count.html', predictions= predictions)          
    
        

if __name__ == "__main__":
    init()
    app.run(debug=True,use_reloader=False,threaded= False)
