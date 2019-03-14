#IMPORT ALL THE NEEDED LIBRARIES

import pandas as pd 
import numpy as np 
from sklearn.model_selection import cross_validate
from sklearn import preprocessing, metrics, svm, tree, ensemble


df = pd.read_csv("bigml_59c28831336c6604c800002a.csv")
print(df.head())

print(df.info())
# There are 4 object values that need to be manupplated or removed.
# Phone Number does not make any difference to the target value that is CHURN, so drop it. 
df = df.drop(['phone number'], axis=1 )
print(df.info())

# next step is to convert the categorical/object data to numerical data.

label_encode = preprocessing.LabelEncoder()

df['state'] = label_encode.fit_transform(df['state'])
df['international plan'] = label_encode.fit_transform(df['international plan'])
df['voice mail plan'] = label_encode.fit_transform(df['voice mail plan'])
print(df.info())