#!/usr/bin/env python
# coding: utf-8

# **Importing Important Libraries**

# In[194]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import pickle

pd.pandas.set_option('display.max_columns',None)#because we want to see all the columns.


# **Reading the Dataset**

# In[195]:


dataset=pd.read_csv('train.csv')
print(dataset.shape)


# In[196]:


dataset.head()


# In[197]:


dataset.describe()


# In[198]:


dataset.info()


# In[199]:


dataset.nunique()


# **In Data Analysis we will Analyze to find out the below stuff:**
# 
# **1)Missing values**
# 
# **2)All the Numerical Variables**
# 
# **3)Distributuion of the numerical variables**
# 
# **4)Categorical Variables**
# 
# **5)Cardinality of Categorical Variables**
# 
# **6)Outliers**
# 
# **7)Relationship between independent and dependent feature(SalePrice)**

# **MISSING VALUES**

# In[200]:


feature_with_na=[features for features in dataset.columns if dataset[features].isnull().sum()>1]

for feature in feature_with_na: print(feature,np.round(dataset[feature].isnull().mean(),4),'% missing values')


# **Since there are many missing values we need to find the relation between the missing values and PM2.5**

# In[201]:


for feature in feature_with_na:
    data=dataset.copy()

    data[feature]=np.where(data[feature].isnull(),1,0)
    data.groupby(feature)['PM2.5'].median().plot.bar()
    plt.title(feature)
    plt.show()


# **Here with the relation between the missing values and dependent variables is clearly visible. So we need to replace these nan values with something meaningful which we will do in the feature engineering section.**

# In[202]:


dataset.isnull().sum()


# In[203]:


dataset


# In[204]:


dataset['pressure'] = dataset['pressure'].astype(float)#to conver the string values to float, and converting pressure to a numerical value


# In[205]:


dataset


# In[206]:


dataset['wind_direction']=dataset['wind_direction'].fillna(dataset['wind_direction'].mode()[0])


# In[207]:


dataset.fillna(dataset.mean(),inplace=True)


# In[208]:


dataset.isnull().sum()


# **##Unnamed: 0 has no relation with the prediction of PM2.5, so we can remove it##**

# In[209]:


dataset.drop(['Unnamed: 0'],axis=1,inplace=True)


# ##**As we have dealt with missing values,now we can detect outliers and remove them so they did not affect our prediction values**##

# **OUTLIERS**

# In[210]:


dataset


# In[211]:


min_threshold,max_threshold=dataset.year.quantile([0.001,0.999])
min_threshold,max_threshold


# In[212]:


dataset[dataset.year<min_threshold]


# In[213]:


dataset[dataset.year>max_threshold]


# In[214]:


dataset[dataset.year<min_threshold]


# In[215]:


dataset=dataset[(dataset.year>=min_threshold) & (dataset.year<=max_threshold)]


# In[216]:


dataset.shape


# In[217]:


dataset


# In[218]:


minm_threshold,maxm_threshold=dataset.month.quantile([0.001,0.999])
minm_threshold,maxm_threshold


# In[219]:


dataset[dataset.month<minm_threshold]


# In[220]:


dataset[dataset.month>maxm_threshold]


# In[221]:


dataset=dataset[(dataset.month>=minm_threshold) & (dataset.month<=maxm_threshold)]
dataset


# In[222]:


minm_threshold,maxm_threshold=dataset.day.quantile([0.001,0.999])
minm_threshold,maxm_threshold


# In[223]:


dataset[dataset.day<minm_threshold]


# In[224]:


dataset[dataset.day>maxm_threshold]


# In[225]:


dataset=dataset[(dataset.day>=minm_threshold) & (dataset.day<=maxm_threshold)]
dataset


# In[226]:


minm_threshold,maxm_threshold=dataset.hour.quantile([0.001,0.999])
minm_threshold,maxm_threshold


# In[227]:


dataset[dataset.hour<minm_threshold]


# In[228]:


dataset[dataset.hour>maxm_threshold]


# In[229]:


dataset=dataset[(dataset.hour>=minm_threshold) & (dataset.hour<=maxm_threshold)]
dataset


# In[230]:


minm_threshold,maxm_threshold=dataset.temperature.quantile([0.001,0.999])
minm_threshold,maxm_threshold


# In[231]:


dataset[dataset.temperature<minm_threshold]


# In[232]:


dataset[dataset.temperature>maxm_threshold]


# In[233]:


dataset=dataset[(dataset.temperature>=minm_threshold) & (dataset.temperature<=maxm_threshold)]
dataset


# In[234]:


minm_threshold,maxm_threshold=dataset['PM2.5'].quantile([0.0001,0.9999])
minm_threshold,maxm_threshold


# In[235]:


dataset[dataset['PM2.5']<minm_threshold]


# In[236]:


dataset[dataset['PM2.5']>maxm_threshold]


# In[237]:


dataset=dataset[(dataset['PM2.5']>=minm_threshold) & (dataset['PM2.5']<=maxm_threshold)]
dataset


# In[238]:


min_threshold,max_threshold=dataset.pressure.quantile([0.001,0.9999])
min_threshold,max_threshold


# In[239]:


dataset[dataset['pressure']<min_threshold]


# In[240]:


dataset[dataset['pressure']>max_threshold]


# In[241]:


dataset=dataset[(dataset['pressure']>=min_threshold) & (dataset['pressure']<=max_threshold)]
dataset


# In[242]:


min_threshold,max_threshold=dataset.rain.quantile([0.001,0.9999])
min_threshold,max_threshold


# In[243]:


dataset[dataset['rain']<min_threshold]


# In[244]:


dataset[dataset['rain']>max_threshold]


# In[245]:


dataset=dataset[(dataset['rain']>=min_threshold) & (dataset['rain']<=max_threshold)]
dataset


# **As now we have dealt with outliers, now we can analyse our Numerical and Categorical variables**

# **NUMERICAL AND CATEGORICAL VARIABLES**

# In[246]:


dataset.info()


# In[247]:


dataset.head()


# In[248]:


numerical_features = [feature for feature in dataset.columns if dataset[feature].dtypes != 'O']

print('Number of numerical variables: ', len(numerical_features))

# visualise the numerical variables
dataset[numerical_features].head()


# In[249]:


discrete_feature=[feature for feature in numerical_features if len(dataset[feature].unique())<25]
print("Discrete Variables Count: {}".format(len(discrete_feature)))


# In[250]:


discrete_feature


# In[251]:


dataset[discrete_feature].head()


# In[252]:


for feature in discrete_feature:
    data=dataset.copy()
    data.groupby(feature)['PM2.5'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('PM2.5')
    plt.title(feature)
    plt.show()


# In[253]:


continuous_feature=[feature for feature in numerical_features if feature not in discrete_feature]
print("Continuous feature Count {}".format(len(continuous_feature)))


# In[254]:



for feature in continuous_feature:
    data=dataset.copy()
    data[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.title(feature)
    plt.show()


# In[255]:


for feature in continuous_feature:
        data=dataset.copy()
    
        data[feature]=np.log(data[feature])
        data['PM2.5']=np.log(data['PM2.5'])
        plt.scatter(data[feature],data['PM2.5'])
        plt.xlabel(feature)
        plt.ylabel('PM2.5')
        plt.title(feature)
        plt.show()


# **As now we have dealt with numerical variables, now we can deal with categorical variables**

# **CaATEGORICAL VARIABLES**

# In[256]:


categorical_features=[feature for feature in dataset.columns if data[feature].dtypes=='O']
categorical_features


# In[257]:


dataset[categorical_features].head()


# In[258]:


for feature in categorical_features:
    print('The feature is {} and number of categories are {}'.format(feature,len(dataset[feature].unique())))


# In[259]:


for feature in categorical_features:
    data=dataset.copy()
    data.groupby(feature)['PM2.5'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('PM2.5')
    plt.title(feature)
    plt.show()


# In[261]:


for feature in categorical_features:
    labels_ordered=dataset.groupby([feature])['PM2.5'].mean().sort_values().index
    labels_ordered={k:i for i,k in enumerate(labels_ordered,0)}
    dataset[feature]=dataset[feature].map(labels_ordered)


# In[262]:


dataset.head(10)


# **Splitting The Data In Test and Train**

# In[263]:


X=dataset.drop(['PM2.5'],axis=1)
y=dataset['PM2.5']


# **Training The Data Using Sci-Kit Library**

# In[264]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.2)


# In[265]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)


# In[270]:


y_pred=regressor.predict(X_test)
y_pred


# In[267]:


from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# In[268]:


plt.figure(figsize=(15,10))
plt.scatter(y_test,y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')


# In[269]:


pred_y_df=pd.DataFrame({'Actual Value':y_test,'Predicted Value':y_pred,'Difference':y_test-y_pred})
pred_y_df[0:20]


# In[ ]:

pickle.dump(regressor,open('model.pkl','wb'))

model=pickle.load(open('model.pkl','rb'))

print(model.predict([[1,2,3,4,5,6,7,8,9]]))

# %%
