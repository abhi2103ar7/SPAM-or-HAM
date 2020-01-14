# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 19:35:15 2020

@author: ABHI$HEK
"""
from flask import Flask,render_template,url_for,request,redirect

from nltk.corpus import stopwords
import string
import pandas as pd
import pickle
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report,confusion_matrix

app = Flask(__name__)

@app.route('/',methods = ['GET', 'POST'])
def home():
    if request.method == 'POST':
        return redirect(url_for('predict'))
    return render_template('home.html')

@app.route('/predict',methods=['POST'])

def predict():
    df = pd.read_csv('spam.csv', encoding='latin-1')
    df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)
    #print(messages.head())
    df.rename(columns = {'v1': 'labels', 'v2': 'message'}, inplace = True)
    df['label'] = df['labels'].map({'ham': 0, 'spam': 1})
    X = df['message']
    y = df['label']
    #print(messages['length'])
    #messages.hist(column='length',by='class',bins=50, figsize=(15,6))
    # Extract Feature With CountVectorizer
    cv = CountVectorizer()
    X = cv.fit_transform(X) # Fit the Data
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	#Naive Bayes Classifier


    clf = MultinomialNB()
    clf.fit(X_train,y_train)
    clf.score(X_test,y_test)
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run(debug=True)