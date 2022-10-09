import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
#import joblib

df= pd.read_csv("star_classification.csv")

le = LabelEncoder()
df["class"] = le.fit_transform(df["class"])
df["class"] = df["class"].astype(int)

df = df.drop(['obj_ID','run_ID','rerun_ID','cam_col','MJD','spec_obj_ID','field_ID','fiber_ID'], axis = 1)
x = df.drop(['class'], axis = 1)
#y = df.loc[:,'class'].values
#x.to_csv('scaler_data.csv', index=False)

mms = MinMaxScaler()
x = mms.fit_transform(x)
#joblib.dump(x,'scaler.gz')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 42,shuffle = True)

KNN_classifier = KNeighborsClassifier(n_neighbors = 5,metric = 'minkowski',p = 2)
KNN_classifier.fit(x_train,y_train)
pickle.dump(KNN_classifier,open('knn.pkl','wb'))

SV_classifier = SVC(kernel = 'rbf',random_state = 42)
SV_classifier.fit(x_train,y_train)
pickle.dump(SV_classifier,open('svc.pkl','wb'))

nb = GaussianNB()
nb.fit(x_train,y_train)
pickle.dump(nb,open('nb.pkl','wb'))

Tree = DecisionTreeClassifier(criterion = 'entropy',random_state = 42)
Tree.fit(x_train,y_train)
pickle.dump(Tree,open('tree.pkl','wb'))

Forest = RandomForestClassifier(n_estimators = 100, criterion = 'entropy',random_state = 42)
Forest.fit(x_train,y_train)
pickle.dump(Forest,open('forest.pkl','wb'))

gbc = GradientBoostingClassifier()
gbc.fit(x_train,y_train)
pickle.dump(gbc,open('gbc.pkl','wb'))

