#!/usr/bin/env python
# coding: utf-8

# In[16]:


# importing module 
from pymongo import MongoClient 
import numpy as np
import pandas as pd
import visuals as vs
from IPython.display import display
get_ipython().run_line_magic('matplotlib', 'inline')
  


# In[5]:


# creation of MongoClient 
client=MongoClient() 
  
# Connect with the portnumber and host 
client = MongoClient("mongodb://localhost:27017/") 
  
# Access database 
mydatabase = client['Customer_Segment'] 
  
# Access collection of the database 
mycollection=mydatabase['things'] 


# In[6]:


try:
    cursor = mycollection.find() 
    data = pd.DataFrame(list(cursor))
    data.drop(['_id','Region', 'Channel'], axis = 1, inplace = True)
    print(data)
    print ("Wholesale customers dataset has {} samples with {} features each.".format(*data.shape))
except:
    print ("Dataset could not be loaded. Is the dataset missing?")


# # Data Exploration

# In[7]:


# Display a description of the dataset
display(data.describe())


# # Selecting Samples

# In[8]:


# TODO: Select three indices of to sample from the dataset
indices = [26,176,392]


# # Create a DataFrame of the chosen samples
# 

# In[9]:


samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)
print ("Chosen samples of wholesale customers dataset: ")
display(samples)


# # Feature Relevance

# In[10]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
milk_data = data.drop(['Milk'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(milk_data,data['Milk'],test_size=0.25,random_state=101)
regressor = DecisionTreeRegressor(random_state=101).fit(X_train,y_train)
score = regressor.score(X_test,y_test)
print(score)


# In[11]:


# Produce a scatter matrix for each pair of features in the data
pd.plotting.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');


# # Feature Scaling

# In[12]:


log_data = data.apply(lambda x: np.log(x))
log_samples = samples.apply(lambda x: np.log(x))
pd.plotting.scatter_matrix(log_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');


# In[13]:


display(log_samples)


# # Outlier Detection

# In[14]:


# OPTIONAL: Select the indices for data points you wish to remove
outliers  = []

# For each feature find the data points with extreme high or low values
for feature in log_data.keys():
    
    # TODO: Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(log_data[feature],25)
    
    # TODO: Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(log_data[feature],75)
    
    # TODO: Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    step = (Q3-Q1) * 1.5
    
    # Display the outliers
    print("Data points considered outliers for the feature '{}':".format(feature))
    out = log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))]
    display(out)
    outliers = outliers + list(out.index.values)
    

#Creating list of more outliers which are the same for multiple features.
outliers = list(set([x for x in outliers if outliers.count(x) > 1]))    

print("Outliers: {}".format(outliers))

# Remove the outliers, if any were specified 
good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)
print("The good dataset now has {} observations after removing outliers.".format(len(good_data)))


# # Feature Transformation
# # PCA

# In[17]:


from sklearn.decomposition import PCA

# TODO: Apply PCA by fitting the good data with the same number of dimensions as features
pca = PCA().fit(good_data)

# TODO: Transform log_samples using the PCA fit above
pca_samples = pca.transform(log_samples)

# Generate PCA results plot
pca_results = vs.pca_results(good_data, pca)


# In[18]:


display(pd.DataFrame(np.round(pca_samples, 4), columns = pca_results.index.values))


# # Dimensionality Reduction

# In[19]:


# TODO: Apply PCA by fitting the good data with only two dimensions
pca = PCA(n_components=2).fit(good_data)

# TODO: Transform the good data using the PCA fit above
reduced_data = pca.transform(good_data)

# TODO: Transform log_samples using the PCA fit above
pca_samples = pca.transform(log_samples)

# Create a DataFrame for the reduced data
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])


# In[20]:


display(pd.DataFrame(np.round(pca_samples, 4), columns = ['Dimension 1', 'Dimension 2']))


# # Visualizing a Biplot

# In[21]:


vs.biplot(good_data, reduced_data, pca)


# In[ ]:




