# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 11:27:51 2018

@author: Aryaman Sriram
"""

import os 


os.chdir('C:\\Users\\Aryaman Sriram\Documents\DatasetsKaggle\SignLanguage\Sign-language-digits-dataset')
import numpy as np
import tensorflow as tf
X=np.load('X.npy')

Y=np.load('Y.npy')
labels=np.zeros(shape=(len(Y),1))
for i in range(len(Y)):
    labels[i]=np.argmax(Y[i,:])
imgvec=np.zeros(shape=(len(X),64*64))
for i in range(len(X)):
    imgvec[i,:]=X[i,:].flatten()
    

def MiniBatch(X,Y,batch_size):
    i=0
    minibatch_list=[]
    while(i<len(X)):
        if(i+batch_size<len(X)):
            
            minibatch_X=X[i:i+batch_size,:]
            minibatch_Y=Y[i:i+batch_size,:]
        else:
            minibatch_X=X[i:len(X),:]
            minibatch_Y=Y[i:len(X),:]  
        minibatch_list.append((minibatch_X,minibatch_Y))
        i+=batch_size                  
    return minibatch_list     
#from sklearn.cross_validation import train_test_split
#X_train,X_test,Y_train,Y_test=train_test_split(imgvec,labels,test_size=0.33)
#
##SVM ACCURACY=0.34
##SVM ACCURACY WITH LINEAR KERNEL = 0.79
##SVM ACCURACY WITH KERNEL = RBF and C=100 =0.82
#
#from sklearn.svm import SVC
#clf=SVC(kernel='rbf',C=100)
#clf.fit(X_train,Y_train)
#pred=clf.predict(X_test)
#from sklearn.metrics import accuracy_score
#score=accuracy_score(pred,Y_test)
#
##ADABOOST: ACCURACY: 0.28
#from sklearn.ensemble import AdaBoostClassifier
#clf=AdaBoostClassifier()
#clf.fit(X_train,Y_train)
#pred=clf.predict(X_test)
#from sklearn.metrics import accuracy_score
#score=accuracy_score(pred,Y_test)
##DECISION TREE CLASSIFIER MIN_SAMPLES_SPLIT=10 ACCURACY: 0.54
#from sklearn.tree import DecisionTreeClassifier
#clf=DecisionTreeClassifier(min_samples_split=100)
#clf.fit(X_train,Y_train)
#pred=clf.predict(X_test)
#from sklearn.metrics import accuracy_score
#scr=accuracy_score(pred,Y_test)
#
##logisticRegression accuracy: 0.74
#from sklearn.linear_model import LogisticRegression
#clf=LogisticRegression(C=0.45)
#clf.fit(X_train,Y_train)
#pred=clf.predict(X_test)
#from sklearn.metrics import accuracy_score
#scr=accuracy_score(pred,Y_test)

def Layer(data,prev_layer,next_layer):
    W=tf.Variable(tf.random_normal([prev_layer,next_layer]))
    b=tf.Variable(tf.random_normal([1,next_layer]))
    output=tf.add(tf.matmul(data,W),b)
    return output
    

def neural_network(data):
    layer_1=Layer(data,4096,500)
    layer_1=tf.nn.relu(layer_1)
    print(layer_1.shape)
    layer_2=Layer(layer_1,500,250)
    layer_2=tf.nn.relu(layer_2)
    layer_3=Layer(layer_2,250,100)
    layer_3=tf.nn.relu(layer_3)
    output=Layer(layer_3,100,10)
    return output
x=tf.placeholder(dtype='float32',shape=[None,4096])
y=tf.placeholder(dtype='float32',shape=[None,10])
imgvec=np.array(imgvec,'float32')
pred=neural_network(x)
#cost=-tf.reduce_sum(y*tf.log(pred+1e-8))
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred,labels=y))
optimizer=tf.train.AdamOptimizer().minimize(cost)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    Mlist=MiniBatch(imgvec,Y,100)
    
    for epoch in range(150):
        epoch_loss=0
        for i in Mlist:
            epoch_x=i[0]
            epoch_y=i[1]
            c=sess.run(cost,feed_dict={x:epoch_x,y:epoch_y})
            train_step=sess.run(optimizer,feed_dict={x:epoch_x,y:epoch_y})
            
        epoch_loss+=c
        print("Number of epochs done:",epoch,"Loss:",epoch_loss)
        correct=tf.equal(tf.arg_max(pred,1),tf.arg_max(Y,1))
        accuracy=tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy:',accuracy.eval({x:imgvec, y:Y}))
        
        