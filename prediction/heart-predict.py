# importing libraries--------------------------------------------------------------
import pandas as pd
from matplotlib import rcParams
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn import linear_model, tree, ensemble
import seaborn as sns
import pickle

df=pd.read_csv('C:\\Users\\salvi\\Documents\\Nicepage Templates\\Heartfit Diagnose\\prediction\\heart_disease_dataset.csv')
gender=df.groupby(['sex'],as_index=False).count()
gender=gender[['sex','target']]
df['target'].value_counts()
X=df.drop(['target'], axis=1)
Y=df['target']


#splitting the data into training and testing set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2, stratify=Y, random_state=2)

from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

tree_model = DecisionTreeClassifier(max_depth=15,criterion='entropy')
cv_scores = cross_val_score(tree_model, X, Y, cv=10, scoring='accuracy')

tree_model.fit(X_train.values,Y_train)
pickle.dump(tree_model, open('model.pkl','wb'))

model =pickle.load(open('model.pkl','rb'))


