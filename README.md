How to clone this project 
Open VS Code TERMINAL ( Ctrl + ` ) or ( Ctrl + Shift + P)

>git Clone

git clone https://github.com/saiiiiiii/Gender-classification-flask-app.git

Open the folder you have just cloned ( File->Open Folder)

After Clone

first you have to upgrade your pip 
python -m pip install –upgrade pip

required packages (or) library
1. Flask
2. numpy
3. pandas
4. sklearn (or) scikit learn
5. scipy

install library:- pip install library_name (or) python -m pip install library_name

get any error install sklearn please install microsoft visual studio c++ tool

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer

df = pd.read_csv('names_dataset.csv') #for importing dataset we are using pandas 
#you found this file data folder

df.head()

.... # you find this entire train model in data folder

After completion this train model
before deploy code to flask

we have to import pickle library and joblib library

with these we have to generate 
decisiontreModel and naivebayesgendermodel with extension .pkl

with these naivebayesgendermodel.pkl and decisiontreModel.pkl deploy to flask

from flask import Flask,render_template,url_for,request
from flask_bootstrap import Bootstrap 
import pandas as pd 
import numpy as np 

# ML Packages
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib


app = Flask(__name__)
Bootstrap(app)


@app.route('/')
def index():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
	df= pd.read_csv("data/names_dataset.csv")
	# Features and Labels
	df_X = df.name
	df_Y = df.sex
    
    # Vectorization
	corpus = df_X
	cv = CountVectorizer()
	X = cv.fit_transform(corpus) 
	
	# Loading our ML Model
	naivebayes_model = open("models/naivebayesgendermodel.pkl","rb")
	clf = joblib.load(naivebayes_model)

	# Receives the input query from form
	if request.method == 'POST':
		namequery = request.form['namequery']
		data = [namequery]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('results.html',prediction = my_prediction,name = namequery.upper())


if __name__ == '__main__':
	app.run(debug=True)
  
similarly we did index.html and result.html #find those files in templates folder


run application:-
   python app.py runserver
   
 

  
