# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 18:52:02 2019

@author: sband
"""

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer

df = pd.read_csv('names_dataset.csv')

df.head()

#size
df.size

#columns 
df.columns

df.dtypes

df.isnull().isnull().sum()

#no.of females
df[df.sex == 'F'].size

#no.of males
df[df.sex == 'M'].size

df_names = df

df_names.sex.replace({'F':0,'M':1},inplace=True)

df_names.sex.unique()


df_names.dtypes

Xfeatures =df_names['name']


# Feature Extraction 
cv = CountVectorizer()
X = cv.fit_transform(Xfeatures)

cv.get_feature_names()

from sklearn.model_selection import train_test_split


# Features 
X
# Labels
y = df_names.sex

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)

print("Accuracy of Model",clf.score(X_test,y_test)*100,"%")

print("Accuracy of Model",clf.score(X_train,y_train)*100,"%")



# Sample1 Prediction
sample_name = ["devi"]
vect = cv.transform(sample_name).toarray()

vect

clf.predict(vect)

sample_name1 = ["ram"]
vect1 = cv.transform(sample_name1).toarray()

vect1

clf.predict(vect1)

print(clf.predict(vect1))


# Sample3 Prediction of Russian Names
sample_name2 = ["Natasha"]
vect2 = cv.transform(sample_name2).toarray()

clf.predict(vect2)

# Sample3 Prediction of Random Names
sample_name3 = ["Nefertiti","Nasha","Ama","Ayo","Xhavier","Ovetta","Tathiana","Xia","Joseph","Xianliang"]
vect3 = cv.transform(sample_name3).toarray()

clf.predict(vect3)

def genderpredictor(a):
    test_name = [a]
    vector = cv.transform(test_name).toarray()
    if clf.predict(vector) == 0:
        print("Female")
    else:
        print("Male")

namelist = ["sai","Yaw","Femi","Masha"]
for i in namelist:
    print(genderpredictor(i))
    
def features(name):
    name = name.lower()
    return {
        'first-letter': name[0], # First letter
        'first2-letters': name[0:2], # First 2 letters
        'first3-letters': name[0:3], # First 3 letters
        'last-letter': name[-1],
        'last2-letters': name[-2:],
        'last3-letters': name[-3:],
    }

features = np.vectorize(features)
print(features(["Anna", "sai", "Peter","John","Vladmir","Mohammed"]))

df_X = features(df_names['name'])

df_y = df_names['sex']


from sklearn.feature_extraction import DictVectorizer
 
corpus = features(["Mike", "Julia"])
dv = DictVectorizer()
dv.fit(corpus)
transformed = dv.transform(corpus)
print(transformed)

dv.get_feature_names()

dfX_train, dfX_test, dfy_train, dfy_test = train_test_split(df_X, df_y, test_size=0.33, random_state=42)

dfX_train


dv = DictVectorizer()
dv.fit_transform(dfX_train)



# Model building Using DecisionTree

from sklearn.tree import DecisionTreeClassifier
 
dclf = DecisionTreeClassifier()
my_xfeatures =dv.transform(dfX_train)
dclf.fit(my_xfeatures, dfy_train)


sample_name_eg = ["Alex"]
transform_dv =dv.transform(features(sample_name_eg))

vect3 = transform_dv.toarray()

dclf.predict(vect3)

if dclf.predict(vect3) == 0:
    print("Female")
else:
    print("Male")
    
name_eg1 = ["hello"]
transform_dv =dv.transform(features(name_eg1))
vect4 = transform_dv.toarray()
if dclf.predict(vect4) == 0:
    print("Female")
else:
    print("Male")


# A function to do it
def genderpredictor1(a):
    test_name1 = [a]
    transform_dv =dv.transform(features(test_name1))
    vector = transform_dv.toarray()
    if dclf.predict(vector) == 0:
        print("Female")
    else:
        print("Male")
        
random_name_list = ["Alex","Alice","Chioma","Vitalic","Clairese","Chan"]


for n in random_name_list:
    print(genderpredictor1(n))

print(dclf.score(dv.transform(dfX_train), dfy_train))

print(dclf.score(dv.transform(dfX_test), dfy_test))

from sklearn.externals import joblib

decisiontreModel = open("decisiontreemodel.pkl","wb")

joblib.dump(dclf,decisiontreModel)

decisiontreModel.close


#Alternative to Model Saving
import pickle
dctreeModel = open("namesdetectormodel.pkl","wb")

pickle.dump(dclf,dctreeModel)

dctreeModel.close()

NaiveBayesModel = open("naivebayesgendermodel.pkl","wb")

joblib.dump(clf,NaiveBayesModel)

NaiveBayesModel.close()
