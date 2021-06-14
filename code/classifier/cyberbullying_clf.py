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
feature_cols_sm = ['Retweets#','Favorites#','Hashtags#','Medias#','Mentions#','SenderAccountYears','SenderFavorites#','SenderFollowings#','SenderFollowers#','SenderStatues#']
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
# clf=svm.SVC()
# for x in range(500, 4000, 500):
#     test = SelectKBest(score_func=chi2, k=x)
#     fit = test.fit(x_text_tfidf, y)
#     x_t= fit.transform(x_text_tfidf)
#     scores = cross_val_score(clf, x_t, y, cv=10)
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

# x=x_ts

# search_grid = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
#                      'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
#                     {'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
#                      'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
#                     {'kernel': ['linear'], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}
#                    ]
# search = GridSearchCV(SVC(), search_grid, cv=10, n_jobs=16 ,scoring='%s_macro' % 'recall')
# search.fit(x, y)
# search.best_params_


#2.2) KNN

# search_grid = dict(n_neighbors = list(range(1,31)), metric = ['euclidean', 'manhattan'] )
# search = GridSearchCV(KNeighborsClassifier(), search_grid, cv = 10, scoring = 'recall', n_jobs=16)
# search.fit(x,y)
# search.best_params_



#2.3) Logistic Regression

# search_grid={"C":numpy.logspace(-3,3,7), "penalty":["l1","l2"]}# l1 lasso l2 ridge
# search=GridSearchCV(LogisticRegression(), search_grid, cv=10)
# search.fit(x,y)
# search.best_params_


#2.4) AdaboostClassifier


# search_grid={'n_estimators':[500,1000,2000],'learning_rate':[.001,0.01,.1]}
# search=GridSearchCV(estimator=AdaBoostClassifier(),param_grid=search_grid,scoring='recall' , cv=10, n_jobs=32)
# search.fit(x,y)
# search.best_params_


#2.5) RandomForestClassifier

# search_grid = { 
#     'n_estimators': [200, 700],
#     'max_features': ['auto', 'sqrt', 'log2']
# }
# search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=search_grid, cv= 10, n_jobs=-1)
# search.fit(x, y)
# search.best_params_



#3) CLASSIFIERS
    

#The machine learning algorithms experimented on two different variants of the prepared datasets. 
#The  first variant, named as D_T , includes only textual features. 
#On the other hand, the second variant, DT+S, consists of the determined social media features and textual features.
#Parameters of a classifier on related dataset obtained from second step.

#3.1) SVM

#3.1.A)  text and social media features
# clf=svm.SVC(C=5, kernel="linear")
# scores = cross_val_score(clf, x_ts, y, cv=10)
# #3.1.B)  just text features
# clf=svm.SVC(C=50, gamma= 0.01, kernel= 'rbf')
# scores = cross_val_score(clf, x_t, y, cv=10)

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

#3.2) KNN

#3.2.A)  text and social media features
clf= KNeighborsClassifier(n_neighbors= 3, metric="euclidean")
scores_ts = cross_val_score(clf, x_ts, y, cv=10)
knnTs=scores_ts.mean()
#3.2.B)  just text features
clf=KNeighborsClassifier(n_neighbors= 6, metric="euclidean")
scores_t = cross_val_score(clf, x_t, y, cv=10)
knnT=scores_t.mean()


#3.3) NBM

#3.3.A)  text and social media features
clf= MultinomialNB()
scores_ts = cross_val_score(clf, x_ts, y, cv=10)
nbmTs=scores_ts.mean()
#3.3.B)  just text features
clf= MultinomialNB()
scores_t = cross_val_score(clf, x_t, y, cv=10)
nbmT=scores_t.mean()


#3.4) Logistic Regresyon

# #3.4.A)  text and social media features
# clf=LogisticRegression(C=50,penalty="l2")
# scores_ts = cross_val_score(clf, x_ts, y, cv=10)
# #3.4.B)  just text features
# clf=LogisticRegression(C=100,penalty="l2")
# scores_t = cross_val_score(clf, x_t, y, cv=10)


#3.5) AdaBoost

#3.5.A)  text and social media features
clf=AdaBoostClassifier(learning_rate=0.1, n_estimators=1000)
scores_ts = cross_val_score(clf, x_ts, y, cv=10)
adaBoostTs=scores_ts.mean()
#3.5.B)  just text features
clf=AdaBoostClassifier(learning_rate=0.1, n_estimators=1000)
scores_t = cross_val_score(clf, x_t, y, cv=10)
adaBoostT=scores_t.mean()


#3.6) RF

#3.6.A)  text and social media features
clf=RandomForestClassifier( max_features= 'log2', n_estimators= 250)
scores_ts = cross_val_score(clf, x_ts, y, cv=10)
rfTs=scores_ts.mean()
#3.6.B)  just text features
clf=RandomForestClassifier(  max_features= 'log2', n_estimators= 200)
scores_t = cross_val_score(clf, x_t, y, cv=10)
rfT=scores_t.mean()
#4) RESULTS
#Comparison  between the scores of the experimented classifier on dataset variants
allTs= [svmTs,knnTs,nbmTs,adaBoostTs,rfTs]
allT= [svmT,knnT,nbmT,adaBoostT,rfT]

w=0.4
xA=["SVM","KNN","NBM","AdaBoost","RF"]
# plt.bar(x,allTs,w,label="Text and Sosial ")

bar1 = numpy.arange(len(xA))
bar2 = [i+w for i in bar1 ]

plt.bar(bar1,allTs,w,label="Dataset containing social media features and addition to textual features")
plt.bar(bar2,allT,w,label="Dataset containing only textual features")


plt.xlabel("Classifiers")
plt.ylabel("Acuracy")
plt.ylim(0.72, 0.92)
plt.title("Accuracy of the experimented Classifiers")
plt.xticks(bar1+w/2,xA)
plt.legend(loc="upper center", bbox_to_anchor=(0.4, 1.30))
plt.show()
result_dir = '/content/drive/MyDrive/Project/CyberbullyingDetection-/results'
plt.savefig(f"{result_dir}/result.png")




# print('The score of the experimented classifier on the dataset that contains social media features in addition to textual features:')
# print(scores_ts.mean())
# print('The score of the experimented classifier on the dataset that contains just textual features:')
# print(scores_t.mean())