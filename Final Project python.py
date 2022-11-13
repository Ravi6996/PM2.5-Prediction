#!/usr/bin/env python
# coding: utf-8

# **Importing Important Libraries**

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import pickle

pd.pandas.set_option('display.max_columns',None)#because we want to see all the columns.


# **Reading the Dataset**

# In[2]:


dataset=pd.read_csv('train.csv')
print(dataset.shape)


# **Analysing the First 5 rows of data using head()**

# In[3]:


dataset.head()


# **Analysing The data using describe**

# In[4]:


dataset.describe()


# **Checcking for the datatype using .info()**

# In[5]:


dataset.info()


# **Checking for null values using .nunique()**

# In[6]:


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

# In[7]:


feature_with_na=[features for features in dataset.columns if dataset[features].isnull().sum()>1]

for feature in feature_with_na: print(feature,np.round(dataset[feature].isnull().mean(),4),'% missing values')


# **Since there are many missing values we need to find the relation between the missing values and PM2.5**

# In[8]:


for feature in feature_with_na:
    data=dataset.copy()

    data[feature]=np.where(data[feature].isnull(),1,0)
    data.groupby(feature)['PM2.5'].median().plot.bar()
    plt.title(feature)
    plt.show()


# **Here with the relation between the missing values and dependent variables is clearly visible. So we need to replace these nan values with something meaningful which we will do in the feature engineering section.**

# In[9]:


dataset.isnull().sum()


# **Viewing The Dataset**

# In[10]:


dataset


# **Convering the string values to float so that its datatype changes from object to float**

# In[11]:


dataset['pressure'] = dataset['pressure'].astype(float)


# In[12]:


dataset


# **As the datatype is object, so we will replace the nan values with mode**

# In[13]:


dataset['wind_direction']=dataset['wind_direction'].fillna(dataset['wind_direction'].mode()[0])


# **|As the datatype is float, we will replace the missing values with the mean**

# In[14]:


dataset.fillna(dataset.mean(),inplace=True)


# **Checking for null values**

# In[15]:


dataset.isnull().sum()


# **##Unnamed: 0 has no relation with the prediction of PM2.5, so we can remove it##**

# In[16]:


dataset.drop(['Unnamed: 0'],axis=1,inplace=True)


# ##**As we have dealt with missing values,now we can detect outliers and remove them so they did not affect our prediction values**##

# **OUTLIERS**

# In[17]:


dataset


# **Plotting a scatterplot of Dependent and independent values to check for Outliers(Analysing from the plot if they are present or not)**

# **Scaatterplot of PM2.5 and Year**

# In[18]:


plt.scatter(dataset['year'],dataset['PM2.5'])


# **Scaatterplot of PM2.5 and Month**

# In[19]:


plt.scatter(dataset['month'],dataset['PM2.5'])


# **Scaatterplot of PM2.5 and day**

# In[20]:


plt.scatter(dataset['day'],dataset['PM2.5'])


# **Scaatterplot of PM2.5 and Hour**

# In[21]:


plt.scatter(dataset['hour'],dataset['PM2.5'])


# **Scaatterplot of PM2.5 and Temperature**

# In[22]:


plt.scatter(dataset['temperature'],dataset['PM2.5'])


# **Scaatterplot of PM2.5 and Pressure**

# In[23]:


plt.scatter(dataset['pressure'],dataset['PM2.5'])


# **Scaatterplot of PM2.5 and Rain**

# In[24]:


plt.scatter(dataset['rain'],dataset['PM2.5'])


# **Scaatterplot of PM2.5 and Wind Direction**

# In[25]:


plt.scatter(dataset['wind_direction'],dataset['PM2.5'])


# **Scaatterplot of PM2.5 and Wind Speed**

# In[26]:


plt.scatter(dataset['wind_speed'],dataset['PM2.5'])


# **As we have seen through the plots that there are outliers present, so we will clean them using quantile methods**

# In[27]:


min_threshold,max_threshold=dataset.year.quantile([0.001,0.999])
min_threshold,max_threshold


# **Checking for year**

# In[28]:


dataset[dataset.year<min_threshold]


# In[29]:


dataset[dataset.year>max_threshold]


# In[30]:


dataset[dataset.year<min_threshold]


# In[31]:


dataset=dataset[(dataset.year>=min_threshold) & (dataset.year<=max_threshold)]


# **Checking for month**

# In[32]:


minm_threshold,maxm_threshold=dataset.month.quantile([0.001,0.999])
minm_threshold,maxm_threshold


# In[33]:


dataset[dataset.month<minm_threshold]


# In[34]:


dataset[dataset.month>maxm_threshold]


# In[35]:


dataset=dataset[(dataset.month>=minm_threshold) & (dataset.month<=maxm_threshold)]
dataset


# **Checking for day**

# In[36]:


minm_threshold,maxm_threshold=dataset.day.quantile([0.001,0.999])
minm_threshold,maxm_threshold


# In[37]:


dataset[dataset.day<minm_threshold]


# In[38]:


dataset[dataset.day>maxm_threshold]


# In[39]:


dataset=dataset[(dataset.day>=minm_threshold) & (dataset.day<=maxm_threshold)]
dataset


# **Checking for hour**

# In[40]:


minm_threshold,maxm_threshold=dataset.hour.quantile([0.001,0.999])
minm_threshold,maxm_threshold


# In[41]:


dataset[dataset.hour<minm_threshold]


# In[42]:


dataset[dataset.hour>maxm_threshold]


# In[43]:


dataset=dataset[(dataset.hour>=minm_threshold) & (dataset.hour<=maxm_threshold)]
dataset


# **Scaatterplot of PM2.5 and Temperature**

# In[44]:


minm_threshold,maxm_threshold=dataset.temperature.quantile([0.001,0.999])
minm_threshold,maxm_threshold


# In[45]:


dataset[dataset.temperature<minm_threshold]


# In[46]:


dataset[dataset.temperature>maxm_threshold]


# In[47]:


dataset=dataset[(dataset.temperature>=minm_threshold) & (dataset.temperature<=maxm_threshold)]
dataset


# **Checking for PM2.5**

# In[48]:


minm_threshold,maxm_threshold=dataset['PM2.5'].quantile([0.0001,0.9999])
minm_threshold,maxm_threshold


# In[49]:


dataset[dataset['PM2.5']<minm_threshold]


# In[50]:


dataset[dataset['PM2.5']>maxm_threshold]


# In[51]:


dataset=dataset[(dataset['PM2.5']>=minm_threshold) & (dataset['PM2.5']<=maxm_threshold)]
dataset


# **Checking for Pressure**

# In[52]:


min_threshold,max_threshold=dataset.pressure.quantile([0.001,0.9999])
min_threshold,max_threshold


# In[53]:


dataset[dataset['pressure']<min_threshold]


# In[54]:


dataset[dataset['pressure']>max_threshold]


# In[55]:


dataset=dataset[(dataset['pressure']>=min_threshold) & (dataset['pressure']<=max_threshold)]
dataset


# **Checking for Rain**

# In[56]:


min_threshold,max_threshold=dataset.rain.quantile([0.001,0.9999])
min_threshold,max_threshold


# In[57]:


dataset[dataset['rain']<min_threshold]


# In[58]:


dataset[dataset['rain']>max_threshold]


# In[59]:


dataset=dataset[(dataset['rain']>=min_threshold) & (dataset['rain']<=max_threshold)]
dataset


# **As now we have dealt with outliers, now we can analyse our Numerical and Categorical variables**

# **NUMERICAL AND CATEGORICAL VARIABLES**

# In[60]:


dataset.info()


# In[61]:


dataset.head()


# **Checking for numerical Variables**

# In[62]:


numerical_features = [feature for feature in dataset.columns if dataset[feature].dtypes != 'O']

print('Number of numerical variables: ', len(numerical_features))

# visualise the numerical variables
dataset[numerical_features].head()


# **Checking for discrete features in Numerical Variables**

# In[63]:


discrete_feature=[feature for feature in numerical_features if len(dataset[feature].unique())<25]
print("Discrete Variables Count: {}".format(len(discrete_feature)))


# In[64]:


discrete_feature


# In[65]:


dataset[discrete_feature].head()


# **Analysing the discrete variables by plotting their bar plot**

# In[66]:


for feature in discrete_feature:
    data=dataset.copy()
    data.groupby(feature)['PM2.5'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('PM2.5')
    plt.title(feature)
    plt.show()


# **Checking for continous features**

# In[67]:


continuous_feature=[feature for feature in numerical_features if feature not in discrete_feature]
print("Continuous feature Count {}".format(len(continuous_feature)))


# **Analysing the continous features using histogram**

# In[68]:



for feature in continuous_feature:
    data=dataset.copy()
    data[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.title(feature)
    plt.show()


# **Plotting the scaterplots of Continous features with dependent feature**

# In[69]:


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

# In[70]:


categorical_features=[feature for feature in dataset.columns if data[feature].dtypes=='O']
categorical_features


# In[71]:


dataset[categorical_features].head()


# In[72]:


for feature in categorical_features:
    print('The feature is {} and number of categories are {}'.format(feature,len(dataset[feature].unique())))


# In[73]:


for feature in categorical_features:
    data=dataset.copy()
    data.groupby(feature)['PM2.5'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('PM2.5')
    plt.title(feature)
    plt.show()


# **Converting Categorical Data to Numerical values from 0 to 15**

# In[74]:


for feature in categorical_features:
    labels_ordered=dataset.groupby([feature])['PM2.5'].mean().sort_values().index
    labels_ordered={k:i for i,k in enumerate(labels_ordered,0)}
    dataset[feature]=dataset[feature].map(labels_ordered)


# In[75]:


dataset.head(90)


# **Splitting The Data In Dependent and Independent Features**

# In[113]:


X=dataset.drop(['PM2.5'],axis=1)
y=dataset['PM2.5']


# **Importing Train_Test_split using sci-kit library**

# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.05)


# **Training The Data Using Sci-Kit Library**

# **Importing Linear Regression using Sklearn.linear_model**

# In[ ]:


from sklearn.linear_model import LinearRegression
ml=LinearRegression()


# **Performing Cross Validations**

# **KFold Validation**

# In[ ]:


from sklearn.model_selection import KFold
KFold_validation=KFold()
from sklearn.model_selection import cross_val_score
results=cross_val_score(ml,X,y,cv=KFold_validation)
print(results)
print(np.mean(results))


# **ShuffleSplit Validation**

# In[ ]:


from sklearn.model_selection import ShuffleSplit
ssplit=ShuffleSplit(n_splits=1,test_size=0.1)
results=cross_val_score(ml,X,y,cv=ssplit)
np.mean(results)


# **Performing HyperParameter tuning**

# **Importing Xgboost**

# In[ ]:


import xgboost
regressor=xgboost.XGBRegressor()


# In[ ]:


booster=['gbtree','gblinear']
base_score=[0.25,0.5,0.75,0.1]


# **Listing All the parameters**

# In[ ]:


n_estimators=[100,500,900,1100,1500]
max_depth=[2,3,5,10,15]
booster=['gbtree','gblinear']
learning_rate=[0.05,0.1,0.150,0.20]
min_child_weight=[1,2,3,4]

hyperparameter_grid={
    'n_estimators':n_estimators,
    'max_depth':max_depth,
    'learning_rate':learning_rate,
    'min_child_weight':min_child_weight,
    'booster':booster,
    'base_score':base_score
}


# **Importing Randomised Search using Sci-Kit library**

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
random_cv=RandomizedSearchCV(estimator=regressor,param_distributions=hyperparameter_grid,
          cv=5,n_iter=50,
          scoring='neg_mean_absolute_error',n_jobs=4,
          verbose=5,
          return_train_score=True,
          random_state=42)


# **Fitting the training data**

# Running this cell will tak 10 to 12 mins ,so wait patiently

# In[ ]:


random_cv.fit(X_train,y_train)


# **Analysing the best estimator**

# In[ ]:


random_cv.best_estimator_


# In[ ]:


best_model = random_cv.best_estimator_


# In[ ]:


regressor=xgboost.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.05, max_delta_step=0,
       max_depth=3, min_child_weight=1, missing=None, n_estimators=900,
       n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1)


# **finding The r2_score**

# In[ ]:


from sklearn.metrics import r2_score
y_pred=best_model.predict(X_test)
r2_score(y_test,y_pred)


# **Plotting a scatterplot of Actual Vs Predicted**

# In[ ]:


plt.figure(figsize=(15,10))
plt.scatter(y_test,y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')


# **Using pickle for Deployment usin Flask**

# In[ ]:


pickle.dump(ml,open('model.pkl','wb'))

model=pickle.load(open('model.pkl','rb'))

print(model.predict([[1,2,3,4,5,6,7,8,9]]))

