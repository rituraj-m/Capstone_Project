#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing module 
from pymongo import MongoClient 
import numpy as np
import pandas as pd
import visuals as vs
from IPython.display import display
get_ipython().run_line_magic('matplotlib', 'inline')
  


# In[2]:


# creation of MongoClient 
client=MongoClient() 
  
# Connect with the portnumber and host 
client = MongoClient("mongodb://localhost:27017/") 
  
# Access database 
mydatabase = client['Customer_Segment'] 
  
# Access collection of the database 
mycollection=mydatabase['things'] 


# In[3]:


try:
    cursor = mycollection.find() 
    data = pd.DataFrame(list(cursor))
    data.drop(['_id','Region', 'Channel'], axis = 1, inplace = True)
    print(data)
    print ("Wholesale customers dataset has {} samples with {} features each.".format(*data.shape))
except:
    print ("Dataset could not be loaded. Is the dataset missing?")


# # Data Exploration

# In[4]:


# Display a description of the dataset
display(data.describe())


# # Selecting Samples

# In[5]:


# TODO: Select three indices of to sample from the dataset
indices = [26,176,392]


# # Create a DataFrame of the chosen samples
# 

# In[6]:


samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)
print ("Chosen samples of wholesale customers dataset: ")
display(samples)


# # Feature Relevance

# In[7]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
milk_data = data.drop(['Milk'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(milk_data,data['Milk'],test_size=0.25,random_state=101)
regressor = DecisionTreeRegressor(random_state=101).fit(X_train,y_train)
score = regressor.score(X_test,y_test)
print(score)


# In[8]:


# Produce a scatter matrix for each pair of features in the data
pd.plotting.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');


# # Feature Scaling

# In[9]:


log_data = data.apply(lambda x: np.log(x))
log_samples = samples.apply(lambda x: np.log(x))
pd.plotting.scatter_matrix(log_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');


# In[10]:


display(log_samples)


# # Outlier Detection

# In[11]:


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

# In[12]:


from sklearn.decomposition import PCA

# TODO: Apply PCA by fitting the good data with the same number of dimensions as features
pca = PCA().fit(good_data)

# TODO: Transform log_samples using the PCA fit above
pca_samples = pca.transform(log_samples)

# Generate PCA results plot
pca_results = vs.pca_results(good_data, pca)


# In[13]:


display(pd.DataFrame(np.round(pca_samples, 4), columns = pca_results.index.values))


# # Dimensionality Reduction

# In[14]:


# TODO: Apply PCA by fitting the good data with only two dimensions
pca = PCA(n_components=2).fit(good_data)

# TODO: Transform the good data using the PCA fit above
reduced_data = pca.transform(good_data)

# TODO: Transform log_samples using the PCA fit above
pca_samples = pca.transform(log_samples)

# Create a DataFrame for the reduced data
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])


# In[15]:


display(pd.DataFrame(np.round(pca_samples, 4), columns = ['Dimension 1', 'Dimension 2']))


# # Visualizing a Biplot

# In[16]:


vs.biplot(good_data, reduced_data, pca)


# # Creating Clusters

# In[19]:


n_clusters = [8,6,4,3,2]

from sklearn import mixture
from sklearn.metrics import silhouette_score

for n in n_clusters:
    
    # TODO: Apply your clustering algorithm of choice to the reduced data 
    clusterer = mixture.GaussianMixture(n_components=n).fit(reduced_data)

    # TODO: Predict the cluster for each data point
    preds = clusterer.predict(reduced_data)

    # TODO: Find the cluster centers
    centers = clusterer.means_

    # TODO: Predict the cluster for each transformed sample data point
    sample_preds = clusterer.predict(pca_samples)

    # TODO: Calculate the mean silhouette coefficient for the number of clusters chosen
    score = silhouette_score(reduced_data,preds)
    
    print ("The silhouette_score for {} clusters is {}".format(n,score))


# # Cluster Visualization

# In[20]:


vs.cluster_results(reduced_data, preds, centers, pca_samples)


# # Data Recovery

# In[21]:


log_centers = pca.inverse_transform(centers)

# TODO: Exponentiate the centers
true_centers = np.exp(log_centers)

# Display the true centers
segments = ['Segment {}'.format(i) for i in range(0,len(centers))]
true_centers = pd.DataFrame(np.round(true_centers), columns = data.keys())
true_centers.index = segments
display(true_centers)


# In[23]:


# Display the predictions
for i, pred in enumerate(sample_preds):
    print ("Sample point", i, "predicted to be in Cluster", pred)


# In[ ]:




