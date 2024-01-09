import pandas as pd 
import numpy as np
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from collections import OrderedDict
from sklearn import tree
import matplotlib as matplotlib

df=pd.read_csv("Sample Data for Random Forest (Heart Disease).csv") 
#we are going to predict heart disease
df.head()

sns.countplot(df,x=df['heart disease'])
plt.title('Value counts of heart disease patients')
plt.show()

#Bulding the model without any hyper-aparameter tuning
#putting response variable to y
y=df['heart disease']
#putting feature variable to X
X=df.drop('heart disease',axis=1)

#splitting tha data into train and test
X_train,X_test,y_train,y_test= train_test_split(X,y,train_size=0.7,random_state=100)
X_train.shape,X_test.shape


#n_jobs: integer,optional (default=1)
#the number of jobs to run in parallel for both fit and predict. If -1, then the number of jobs is set to the number of cores.
#max_depth: the maximum depth of the tree. If none, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
#n_estimators: The number of trees in the forest. (defaullt is 100)
#min:samples_leaft: the minimun number of samples required to bie at a leaf node. A split pont at any depth will only be considered if it leaves at least "min:samples:leaf" training samples in each of the left and rigth branches.
#max_features:{"auto","sqr","log2"}, int or float, default="auto"
#the number of features to consider when looking for the best split 

classifier_rf =RandomForestClassifier(random_state=100, n_jobs=-1, max_depth=5, n_estimators=100,oob_score=True)

#fitting the model

classifier_rf.fit(X_train,y_train)


y_pred_test = classifier_rf.predict(X_test)

from sklearn.metrics import accuracy_score
#accuracy score 
print("Accuracy score test dataset : t", accuracy_score(y_test,y_pred_test))



#Building jmodel with hyper-parameter tuning
rf=RandomForestClassifier(random_state=100,n_jobs=-1)

#max_depth: This hyperparameter represents the maximmum level of each tree in the random forest model. A deeper tree performs well and captures a lot of information about the training daram but will not generalized well to test data.
#By default this calue is set to "none" in the sckit-learn library, which means that the trees are left to expand completely.

#Min_samples_leaf: the minimum number of samples required to be at the leaf node of each tree. The default value is 1 in scikit-learn.
#min_samples_split: the minimum number of samples required to split an internal node of each tree. The default value is 2 in scikit-learn.
#this variable is not presented in decision tree; n_estimators: the number of decision trees in the forest. The default number of estimators in scikit-learn is 10.

params={'max_depth':[2,5,10], 'min_samples_leaf':[5,20,100],'min_samples_split':[5,50,100],'n_estimators':[10,50,100]}


#insttantiate the grid search model 
grid_search = GridSearchCV(estimator=rf,
                           param_grid=params,
                           n_jobs=-1, verbose=1, scoring="accuracy")

grid_search.fit(X_train,y_train)

rf_best=grid_search.best_estimator_
rf_best

classifier_rf_tuned= RandomForestClassifier(max_depth=5, min_samples_leaf=5, min_samples_split=50,
                       n_estimators=10, n_jobs=-1, random_state=100)

classifier_rf_tuned.fit(X_train,y_train)
y_pred_test=classifier_rf_tuned.predict(X_test)

print("Accuracy score test dataset : t", accuracy_score(y_test,y_pred_test))
