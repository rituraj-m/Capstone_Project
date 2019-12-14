#!/usr/bin/env python
# coding: utf-8

# In[12]:


# importing module 
from pymongo import MongoClient 
import numpy as np
import pandas as pd
from IPython.display import display
get_ipython().run_line_magic('matplotlib', 'inline')
  


# In[13]:


# creation of MongoClient 
client=MongoClient() 
  
# Connect with the portnumber and host 
client = MongoClient("mongodb://localhost:27017/") 
  
# Access database 
mydatabase = client['Customer_Segment'] 
  
# Access collection of the database 
mycollection=mydatabase['things'] 


# In[21]:


try:
    cursor = mycollection.find() 
    data = pd.DataFrame(list(cursor))
    data.drop(['_id','Region', 'Channel'], axis = 1, inplace = True)
    print(data)
    print ("Wholesale customers dataset has {} samples with {} features each.".format(*data.shape))
except:
    print ("Dataset could not be loaded. Is the dataset missing?")


# # Data Exploration

# In[22]:


# Display a description of the dataset
display(data.describe())


# # Selecting Samples

# In[23]:


# TODO: Select three indices of to sample from the dataset
indices = [26,176,392]


# # Create a DataFrame of the chosen samples
# 

# In[26]:


samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)
print ("Chosen samples of wholesale customers dataset: ")
display(samples)


# # Feature Relevance

# In[31]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
milk_data = data.drop(['Milk'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(milk_data,data['Milk'],test_size=0.25,random_state=101)
regressor = DecisionTreeRegressor(random_state=101).fit(X_train,y_train)
score = regressor.score(X_test,y_test)
print(score)


# In[34]:


# Produce a scatter matrix for each pair of features in the data
pd.plotting.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');


# In[ ]:





# In[ ]:




