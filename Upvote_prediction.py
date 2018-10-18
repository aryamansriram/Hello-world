
# coding: utf-8


#IMPORT LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import to_categorical

#READ DATASETS
df = pd.read_csv('train_NIR5Yl1.csv')
df_test=pd.read_csv('test_8i3B3FC.csv')

#SHOW DATASET
df.head()



df.describe()



#CHECK FOR MISSING DATA
df.info()



#PLOT ALL FEATURES AGAINST EACH OTHER
pd.plotting.scatter_matrix(df, figsize = (20,20))



#PLOT STRONG CORRELATIONS AGAINST ONE ANOTHER
plt.scatter(df['Views'],df['Upvotes'],c = 'red')
plt.show()



#SHOW CORRELATION TABLE
df.corr()
#FeatureScaling :'''FAIL'''
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(data)
data = scaler.transform(data)

#CONVERT DF TO NUMPY ARRAY
def featureFormat( dictionary,features):
    tmp_list = []
    for i in list(dictionary.columns.values):
        if i in features:
   
            tmp_list.append(list(dictionary[str(i)]))
    return (np.array(tmp_list))
    


#OneHotEncode
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(df['Tag'])

Tag=le.transform(df['Tag'])

One = to_categorical(Tag)

Tag=Tag.reshape((330045,1))

One.shape
data=data.transpose()

#ADD ENCODED TAG FEATURE TO DATA TABLE
data=np.column_stack((data,One))

labels = df.Upvotes
features = df.drop(['Upvotes'], axis = 1)
features_list = ['Views','Answers','Reputation']
data = featureFormat(features, features_list)

#SPLIT TRAINING DATA INTO TRAIN AND TEST
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(data,labels,test_size=0.2,random_state=42)



#from sklearn.model_selection import train_test_split
#from sklearn import svm
from sklearn.tree import DecisionTreeRegressor #BEST
from sklearn.ensemble import RandomForestRegressor #BEST
from sklearn.linear_model import LinearRegression,Ridge,Lasso,LassoCV
from sklearn.ensemble import GradientBoostingRegressor

#svm_reg = svm.SVR()
#rdm_reg = RandomForestRegressor()
#dtr_reg = DecisionTreeRegressor()
#lr_reg = LinearRegression()
rid_reg = Ridge()
las_reg = Lasso()

GB = GradientBoostingRegressor(min_samples_split=10,n_estimators = 80)

#FIT THE MODEL
rid_GB.fit(X_train,Y_train)
pred=GB.predict(X_test)

from sklearn.metrics import mean_squared_error
#EVALUTION METRIC MEAN SQUARED ERROR
print(mean_squared_error(pred,Y_test))



#EVALUATION METRIC: R2 SCORE
from sklearn.metrics import r2_score
r2_score(Y_test,pred)



    
data_test = featureFormat(df_test,features_list)
data_test = data_test.transpose()


sub = pd.DataFrame(pred)
submission = pd.merge(df['ID'],sub,hoe='outer',left_index=True,right_index=True)
submission.to_csv("Fill in the blanks")
 



