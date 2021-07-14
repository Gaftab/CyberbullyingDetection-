# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 18:20:44 2021

@author: linus
"""


# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 13:43:44 2019

@author: abozyigit
"""


import numpy
# import numpy as np
import scipy
import pandas
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from scipy.sparse import hstack
from sklearn.preprocessing import MinMaxScaler
pandas.options.mode.chained_assignment = None 

#0) INITIALIZE DATASET

#Text in the dataset were preprocessed  by using the developed C# class TextCleaner where codes are given in textcleaner folder.
#Numerical characters, punctuation marks, and weblinks were removed from the text of posts in the dataset. 
#Additionally, lower case conversion was applied to the text of the posts. 
#Thus, fewer and meaningful tokens would be obtained from the content in the feature extraction process. 
#Additionally, misspelled online bullying terms were corrected using the preprocessing method developed.
#Note that it is sample dataset that contains only 500 tweets, the full dataset will be linked after the publication of article.
    
csv = r'/content/drive/MyDrive/Project/CyberbullyingDetection-/data/cyberbullying_dataset.csv'
dataset = pandas.read_csv(csv)
feature_cols_sm = ['Retweets#','Favorites#','Hashtags#','Medias#','Mentions#','SenderAccountYears','SenderFavorites#','SenderFollowings#','SenderFollowers#','SenderStatues#','IsSelfMentioned']
feature_cols_all=['Text']+feature_cols_sm
X = dataset[feature_cols_all] # All Features

#1) FEATURE ENGINEERING

#1.1) Normalization (Social Media Features)

#The min-max normalization was applied to the numerical social media features of samples in the dataset to remove instability.

scaler = MinMaxScaler()
X[feature_cols_sm] = scaler.fit_transform(X[feature_cols_sm])

x_text=train=X.Text
x_sm=X[feature_cols_sm]

#Converting data frame to sparce matrix
x_sm=scipy.sparse.csr_matrix(x_sm.values)
y = dataset.IsCyberbullying # Target 

# 1.2) Feature Extraction (Textual Features)

# The terms' weights were calculated using the Term Frequency - Inverse Document Frequency (TF-IDF)
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=50000)
tfidf_vect.fit(x_text)
x_text_tfidf =  tfidf_vect.transform(x_text)

# 1.3) Feature Selection (Textual Features)

# Feature selection  using a chi-square score was applied  for each applied machine learning algorithm to select relevant textual features. 

# COMMENT OUT following code block for experimenting different feature sizes for each classifier
clf=AdaBoostClassifier()
for x in range(500, 4000, 500):
    test = SelectKBest(score_func=chi2, k=x)
    fit = test.fit(x_text_tfidf, y)
    x_t= fit.transform(x_text_tfidf)
    scores = cross_val_score(clf, x_t, y, cv=10)
    # print(scores)


# Use k that has the most highest scores.
test = SelectKBest(score_func=chi2, k=500)
fit = test.fit(x_text_tfidf, y)

#x_t only contains social media features
x_t= fit.transform(x_text_tfidf)
#x_ts contain social media features in addition to textual features
x_ts=hstack((x_t, x_sm))


#2) PARAMETER OPTIMIZATION 

#Grid search was applied to the used machine learning algorithms (except NBM) on both datasets. 
#Note that it takes a few days. You can skip this step, the parameters are predefined in third step.
#COMMENT OUT the related code blocks for experimenting parameter optimization of classifiers.

#2.1) SVM 

#Experiment both x; x_ts or x_t

x=x_ts

# #2.4) AdaboostClassifier

search_grid={'n_estimators':[500,1000,2000],'learning_rate':[.001,0.01,.1]}
search=GridSearchCV(estimator=AdaBoostClassifier(),param_grid=search_grid,scoring='recall' , cv=10, n_jobs=32)
search.fit(x,y)
search.best_params_

#3) CLASSIFIERS
    

#The machine learning algorithms experimented on two different variants of the prepared datasets. 
#The  first variant, named as D_T , includes only textual features. 
#On the other hand, the second variant, DT+S, consists of the determined social media features and textual features.
#Parameters of a classifier on related dataset obtained from the second step.

#3.1) SVM

#3.1.A)  text and social media features
clf=svm.SVC(C=5, kernel="linear")
scores_ts = cross_val_score(clf, x_ts, y, cv=10)
svmTs=scores_ts.mean()

#3.1.B)  just text features
clf=svm.SVC(C=50, gamma= 0.01, kernel= 'rbf')
scores_t = cross_val_score(clf, x_t, y, cv=10)
svmT=scores_t.mean()

#COMMENT OUT the related code blocks for experimenting other classifiers.
#Note that textual feature size was set to  according to svc in this script, tune k for other classifiers to get their most succesfull results.
#Remark that it is sample dataset that contains only 500 tweets, the full dataset will be linked after the publication of article.

#3.5) AdaBoost

#3.5.A)  text and social media features
clf=AdaBoostClassifier(learning_rate=0.1, n_estimators=1000)
scores_ts = cross_val_score(clf, x_ts, y, cv=10)
adaBoostTs=scores_ts.mean()
#3.5.B)  just text features
clf=AdaBoostClassifier(learning_rate=0.1, n_estimators=1000)
scores_t = cross_val_score(clf, x_t, y, cv=10)
adaBoostT=scores_t.mean()
print(adaBoostT)
print(adaBoostTs)


