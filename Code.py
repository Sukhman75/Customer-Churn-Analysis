#IMPORT ALL THE NEEDED LIBRARIES

import pandas as pd 
import numpy as np 
import sklearn.model_selection 
from sklearn import preprocessing, metrics, svm, tree, ensemble

#print(sklearn.__version__)
df = pd.read_csv("DATASET.csv")
#print(df.head())
#print(df.info())

# There are 4 object values that need to be manupplated or removed.
# Phone Number does not make any difference to the target value that is CHURN, so drop it. 

df = df.drop(['phone number'], axis=1 )
print(df.info())

# next step is to convert the categorical/object data to numerical data.
#We can use Get_Dummies as well but it can create dimensionality problem
label_encode = preprocessing.LabelEncoder()

df['state'] = label_encode.fit_transform(df['state'])
df['international plan'] = label_encode.fit_transform(df['international plan'])
df['voice mail plan'] = label_encode.fit_transform(df['voice mail plan'])
print(df.info())

# It is the time to split the INDEPENDENT VARIABLES and the DEPENDENT VARIABLE or the features and the target.

df_Target = df.churn 
df_Features = df.drop(['churn'], axis =1)

X = df_Features.values
Y = df_Target.values.astype(np.int)
# Applying K_fold Cross_validation as the data is not balanced 
KF = sklearn.model_selection.StratifiedKFold( n_splits=10, shuffle=True)
KF.get_n_splits(Y,X)
#print(KF)
# Build A function to validate the best classifier
def folds(X, Y,Classifier, Kf):
	y_pred = Y.copy()
	for ii,jj in KF.split(X, Y):
		X_train, X_test = X[ii], X[jj]
		y_train = Y[ii]
		clf = Classifier()
		clf.fit(X_train,y_train)
		y_pred[jj] = clf.predict(X_test)
	return y_pred


# Print The Accuracy results of Different Classifier
print(metrics.accuracy_score(Y, folds(X, Y, ensemble.GradientBoostingClassifier, KF)))
print(metrics.accuracy_score(Y, folds(X, Y, ensemble.RandomForestClassifier, KF)))
print(metrics.accuracy_score(Y, folds(X, Y, svm.SVC, KF)))

c_m_GBC = metrics.confusion_matrix(Y,folds(X, Y, ensemble.GradientBoostingClassifier, KF))
c_m_RFC = metrics.confusion_matrix(Y,folds(X, Y, ensemble.RandomForestClassifier, KF))
c_m_SVC = metrics.confusion_matrix(Y,folds(X, Y, svm.SVC, KF))
print(c_m_GBC,'\n', c_m_RFC,'\n',  c_m_SVC )
from C_M_Plot import matplot_build
matplot_build(c_m_GBC)
matplot_build(c_m_RFC)
matplot_build(c_m_SVC)