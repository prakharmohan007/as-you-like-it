#!/usr/bin/env python
# coding: utf-8

# # Loading Data and applying Collaborative Filtering!

# In[1]:


import wikipedia as wiki
from sklearn.model_selection import train_test_split


# In[2]:


import pandas as pd




# In[3]:


train_data = pd.read_csv('new_ratings_train.csv')


# In[4]:


test_data =  pd.read_csv('new_ratings_test.csv')


# In[ ]:





# In[5]:


trainData= train_data.groupby(['user_id','book_id']).rating.mean().unstack(fill_value=-1)
print ("trainData has", trainData.shape[0], "Rows,", trainData.shape[1], "Columns")
trainData.head()


# In[6]:


import numpy as np
test_UserIDs = test_data['user_id'].values
test_bookIDs = test_data['book_id'].values

test_matrix = test_data.values
test_matrix[0]


# In[7]:


training_matrix = trainData.values
unique_bookIDs = trainData.columns
unique_userIDs = trainData.index

print(training_matrix)
print(unique_bookIDs)
print(unique_userIDs)

#get index in training matrix from movie Id
Book_ID_Index = {}
for i in range(len(unique_bookIDs)):
    Book_ID_Index[unique_bookIDs[i]] = i

#get index in training matrix from User Id
User_ID_Index = {}
for i in range(len(unique_userIDs)):
    User_ID_Index[unique_userIDs[i]] = i
    
# training_matrix[User_ID_Index[]]


# In[7]:


from numpy import linalg as LA
def remove_negative(a):
    return (a!=-1)*a

def Cosine_Similarity(a, b):
    a = remove_negative(a)
    b = remove_negative(b)
    sim = a.dot(b) /((LA.norm(a,ord=2)) * (LA.norm(b,ord=2)))
#     print(sim)
    return sim


# In[8]:


Mean_ratings={}
for id in unique_userIDs:
    list = training_matrix[User_ID_Index[id]]
    mean = np.average(list, weights=(list >= 0))
    Mean_ratings[id]=mean


# In[9]:


Mean_ratings[test_matrix[-1][0]]


# In[10]:


k=0.0015
def get_recommendation(user_Id,book_Id):
    rec = Mean_ratings[user_Id] + k* np.sum([(Cosine_Similarity(training_matrix[User_ID_Index[user_Id]],training_matrix[User_ID_Index[id2]])*(training_matrix[User_ID_Index[id2]][Book_ID_Index[book_Id]] - Mean_ratings[id2])) for id2 in unique_userIDs if training_matrix[User_ID_Index[id2]][Book_ID_Index[book_Id]] != -1 and id2!=user_Id ]) 
    return rec   

predictions_cosine=[]   
# count=0
for test_data_curr in test_matrix:
#     count+=1
#     print(count)
    rec = get_recommendation(test_data_curr[0],test_data_curr[1])
    predictions_cosine.append(rec)


# In[11]:


# for x in predictions_cosine:
#     print(x)


# In[12]:


# user_Id = test_matrix[-1][0]
# book_Id = test_matrix[-1][1]


# print(get_recommendation(test_matrix[-1][0],test_matrix[-1][1]))


# In[13]:


# sum = 0
# for id2 in unique_userIDs:
#     if training_matrix[User_ID_Index[id2]][Book_ID_Index[book_Id]] != -1 and id2!=user_Id:
#         print('mean rating for',id2,'=', Mean_ratings[id2])
#         print('rating=',training_matrix[User_ID_Index[id2]][Book_ID_Index[book_Id]])
#         print(' Cosine Similarity=',Cosine_Similarity(training_matrix[User_ID_Index[user_Id]],training_matrix[User_ID_Index[id2]]))
#         sum+= (Cosine_Similarity(training_matrix[User_ID_Index[user_Id]],training_matrix[User_ID_Index[id2]])*(training_matrix[User_ID_Index[id2]][Book_ID_Index[book_Id]] - Mean_ratings[id2]))
#         print(sum)


# In[14]:


# sum2 = 0.0015 * sum
# pred = Mean_ratings[user_Id] + sum2
# pred


# In[15]:


# rmse(pred,y_true[-4])


# In[16]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
y_true = test_data['rating'].values
print(y_true)
y_pred = np.asarray(predictions_cosine)
print(y_pred)

print('mean absolute error of Cosine = ' , mean_absolute_error(y_true, y_pred))

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

print('root mean square error of Cosine = ' , rmse(y_pred,y_true))


# # Pearson Similarity

# In[17]:


from copy import copy, deepcopy
new_matrix = deepcopy(training_matrix).astype(float)

#make a new training matrix with mean subtracted values and fill empty values with 0
for id in unique_userIDs:
    list = new_matrix[User_ID_Index[id]]
    for j in range(len(list)):
        if list[j] != -1:
            new_matrix[User_ID_Index[id],j] = new_matrix[User_ID_Index[id],j] - Mean_ratings[id]
        else:
            new_matrix[User_ID_Index[id],j] = 0


# In[18]:


def set_threshold(threshold):
    def pearson_similarity(user_id1,user_id2, k=threshold):
        
        user1 = new_matrix[User_ID_Index[user_id1]]
        user2 = new_matrix[User_ID_Index[user_id2]]
        indices = np.logical_and(user1!=0,user2!=0)
        a=user1[indices]
        b=user2[indices]
        if np.sum(indices)<k:
            return 0
        
        if LA.norm(a,ord=2)==0 or LA.norm(b,ord=2)==0:
            return 0
        else:
            return a.dot(b) /((LA.norm(a,ord=2)) * (LA.norm(b,ord=2)))
    return pearson_similarity  
            


# In[19]:


k=0.0015
def get_recommendation_pearson(user_Id,book_id,threshold):
    score = set_threshold(threshold)
    rec = Mean_ratings[user_Id] + k* np.sum([score(user_Id,id2)*(training_matrix[User_ID_Index[id2]][Book_ID_Index[book_id]] - Mean_ratings[id2]) for id2 in unique_userIDs if id2!= user_Id and training_matrix[User_ID_Index[id2]][Book_ID_Index[book_id]] != -1]) 
    return rec   

def get_predictions_pearson(test_matrix,threshold):
    predictions_pearson=[]        
    for test_data_curr in test_matrix:
        rec = get_recommendation_pearson(test_data_curr[0],test_data_curr[1],threshold)
        predictions_pearson.append(rec)
    return np.asarray(predictions_pearson)


# In[20]:


predictions_pearson = get_predictions_pearson(test_matrix,6)


# In[21]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
y_true = test_data['rating'].values
print(y_true)
print(predictions_pearson)

print('mean absolute error of pearson = ' , mean_absolute_error(y_true, predictions_pearson))
print('root mean square error of pearson= ' , rmse(predictions_pearson,y_true))


# In[ ]:




