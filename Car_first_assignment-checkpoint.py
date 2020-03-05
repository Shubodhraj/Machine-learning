#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("F:\ALL Ostfold Reading Materials\Machine learning\car.data")


# In[3]:


df.head()


# In[5]:


df.columns


# In[6]:


df.columns = ['buying','maintenance', 'door', 'persons', 'lug_boot', 'safety', 'class']


# In[7]:


df.columns


# In[8]:


df.head()


# In[9]:


df.describe()


# In[10]:


df.info()


# In[11]:


df.head(20)


# In[42]:


df['persons'].replace({8:2}, inplace=True)


# In[43]:


print(df)


# In[57]:


df['door'].replace({'5more':6}, inplace=True)


# In[58]:


print(df)


# In[44]:


df.info()


# In[17]:


df.loc[~df['door'].astype(str).str.isdigit(), 'door'].tolist()


# In[18]:


df.loc[~df['door'].str.isdigit(), 'door'].tolist()


# In[56]:


df.dtypes


# In[32]:


print (df['door'].dtypes)


# In[55]:


df.str.isalpha()


# In[59]:


df['door'].replace({'5more':6}, inplace=True)


# In[60]:


df.info()


# In[61]:


df['door'].replace({'5':6}, inplace=True)


# In[62]:


df.info()


# In[64]:


df.dtypes


# In[66]:


df.head(200)


# In[67]:


df.describe()


# In[82]:


import seaborn as sns
sns.set(style="darkgrid")
#df = sns.load_dataset("df")
sns.countplot(x="buying", data=df)


# In[99]:


df.info()


# In[100]:


data = df.drop('class', axis = 1)
y = df['class']


# In[101]:


print(data)


# In[103]:


print(y)


# In[104]:


data.info()


# In[105]:


dic = ['buying', 'maintenance', 'lug_boot', 'safety']


# In[107]:


X = pd.get_dummies(data,columns=dic,drop_first=True)


# In[109]:


X.info()


# In[110]:


from sklearn.model_selection import train_test_split


# In[111]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


# In[112]:


from sklearn.tree import DecisionTreeClassifier


# In[113]:


dtree = DecisionTreeClassifier()


# In[114]:


dtree.fit(X_train, y_train)


# In[115]:


prediction = dtree.predict(X_test)


# In[116]:


from sklearn.metrics import classification_report, confusion_matrix


# In[117]:


print(classification_report(y_test, prediction))


# In[118]:


print(confusion_matrix(y_test, prediction))


# In[120]:


import seaborn as sns
sns.set(style="darkgrid")
#df = sns.load_dataset("df")
sns.countplot(x="class", data=df)


# In[ ]:




