#!/usr/bin/env python
# coding: utf-8

# # Deep Neural Network GP MNIST example

# This is a demonstration of the DNNGP implementation from Google Brain paper found here https://arxiv.org/abs/1711.00165
# 
# This implementation was done for the sake of research and comparison to a linear approximation of the recurrence relation. More information about the approximation can be found here

# In[1]:


import numpy as np
from mlxtend.data import mnist_data
import sklearn.preprocessing as skpp


# In[2]:


def mixed_mnist(count):
    total = count*10
    valid = int(np.floor(count/10))
    rawset = mnist_data()
    arranged_data = np.zeros((1,len(rawset[0][0])))
    arranged_target = np.zeros((1,1))
    for i in range(10):
        arranged_data = np.append(arranged_data,rawset[0][500*i:(500*i)+count],axis=0)
        arranged_target = np.append(arranged_target,np.ones((count,),dtype=int)*i)
    for i in range(10): #validation
        arranged_data = np.append(arranged_data,rawset[0][500*(i+1)-valid:500*(i+1)],axis=0)
        arranged_target = np.append(arranged_target,np.ones((valid,),dtype=int)*i)
    arranged_data = np.delete(arranged_data,0,axis=0)
    arranged_target = np.delete(arranged_target,0,axis=0)
    return [arranged_data,arranged_target,valid*10]


# Data Processing

# In[3]:


data = mixed_mnist(100)
min_max_scaler = skpp.MinMaxScaler()
x_train_minmax = min_max_scaler.fit_transform(data[0])


# Create a one hot encoder. Inputs have been scaled from 0->1

# In[4]:


enc = skpp.OneHotEncoder()
enc.fit(data[1].reshape(-1,1))
enc.categories_


# In[5]:


classes = len(enc.transform([[1]]).toarray()[0])


# In[6]:


enc.transform(data[1][0].reshape(-1,1)).toarray()[0]
len(data[1][:-data[2]])


# In[7]:


X_train = np.zeros((len(data[0])-data[2],len(data[0][0])+classes))
X_test = np.zeros((data[2],len(data[0][0])+classes)) 


# In[8]:


for i in range(len(data[0][:-data[2]])):
    X_train[i] = np.append(data[0][i],enc.transform(data[1][i].reshape(-1,1)).toarray()[0])
for i in range(data[2]):
    X_test[i] = np.append(data[0][-(i+1)],np.ones(classes)*(1/classes))


# Test and train vectors have been appropriately one-hot encoded, now we want to normalize the vectors

# In[9]:


X_train_norm = skpp.normalize(X_train, norm='l2')
X_test_norm = skpp.normalize(X_test, norm='l2')


# Now we have our training vectors and test vectors scaled, normalized and encoded.
# 
# Next we want to make an instance of the GP and then evaluate it. All we need to do is pass it the training data, corresponding target values, test data, and then the hyperparamters sigb, sigw, and layers. Sigma b and sigma w correspond to the variance of the bias and weights respectively.
# 
# Calling DNNGP.train() carries out the full evaluation. To use the approximation all that's needed to do is pass a True to the DNNGP.train() function (train(True)). The approximation uses a linear function derived from the recurrence which can speed up training dramatically. If approximation has been done there will be support for getting predictions for new test data not originally included. This will be implemented in the future. For more data on the approximation see [arxiv:]

# In[10]:


from DNNGP import DNNGP


# In[11]:


sigb, sigw, layers = 0.595, 2.26, 8
gp = DNNGP(X_train_norm,data[1][:-data[2]],X_test_norm,sigb,sigw,layers)


# In[12]:


gp.train()


# Now that our GP has been evaluated we can access predictions in the form of raw data or taken from classification of one-hot. For this purpose we will just take the classifications directly

# In[13]:


predict = gp.prediction()
predict = predict[::-1]


# In[14]:


np.unique(predict) #Test to make sure that all of our classes are represented


# In[15]:


test_count = len(data[1][-data[2]:])


# Next we calculate our accuracy using 0-1 loss. For optimization we'd want to compute the MSE from raw counts but just for demonstration a plain accuracy should be fine.

# In[16]:


correct = 0
for i in range(test_count):
    if data[1][-data[2] + i] == predict[i]:
        correct +=1
print(correct / len(predict))


# Now for comparison we will evaluate the GP with the approximate recurrence.

# In[17]:


gp.train(approximate=True)
predict = gp.prediction()
predict = predict[::-1]


# In[18]:


np.unique(predict)


# In[19]:


test_count = len(data[1][-data[2]:])


# In[20]:


correct = 0
for i in range(test_count):
    if data[1][-data[2] + i] == predict[i]:
        correct +=1
print(correct / len(predict))


# Now with the one-shot flattened case which should provide the same performance

# In[21]:


gp.train(one_shot=True)
predict = gp.prediction()
predict = predict[::-1]
np.unique(predict)


# In[22]:


test_count = len(data[1][-data[2]:])
correct = 0
for i in range(test_count):
    if data[1][-data[2] + i] == predict[i]:
        correct +=1
print(correct / len(predict))

