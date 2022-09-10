#!/usr/bin/env python
# coding: utf-8

# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
diabetes = pd.read_csv('Downloads/diabetes.csv')
diabetes.head()


# In[6]:


print("Diabetes data set dimensions : {}".format(diabetes.shape))


# In[7]:


diabetes.groupby('Outcome').size()


# In[8]:


diabetes.groupby('Outcome').hist(figsize=(9, 9))


# In[8]:


diabetes.isnull().sum()
diabetes.isna().sum()


# In[9]:


print("Total : ", diabetes[diabetes.BloodPressure == 0].shape[0])


# In[10]:


print(diabetes[diabetes.BloodPressure == 0].groupby('Outcome')['Age'].count())


# In[11]:


print("Total : ", diabetes[diabetes.SkinThickness == 0].shape[0])


# In[12]:


print(diabetes[diabetes.SkinThickness == 0].groupby('Outcome')['Age'].count())


# In[13]:


print("Total : ", diabetes[diabetes.BMI == 0].shape[0])


# In[14]:


print(diabetes[diabetes.BMI == 0].groupby('Outcome')['Age'].count())


# In[15]:


print("Total : ", diabetes[diabetes.Insulin == 0].shape[0])


# In[16]:


print(diabetes[diabetes.Insulin == 0].groupby('Outcome')['Age'].count())


# In[17]:


diabetes_mod = diabetes[(diabetes.BloodPressure != 0) & (diabetes.BMI != 0) & (diabetes.Glucose != 0)]
print(diabetes_mod.shape)


# In[18]:


feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = diabetes_mod[feature_names]
y = diabetes_mod.Outcome


# In[19]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[36]:


models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVC', SVC()))
models.append(('LR', LogisticRegression()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('GNB', GaussianNB()))
models.append(('RF', RandomForestClassifier()))
models.append(('GB', GradientBoostingClassifier()))


# In[30]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


# In[31]:


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = diabetes_mod.Outcome, random_state=0)


# In[37]:


names = []
scores = []
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))
    names.append(name)
tr_split = pd.DataFrame({'Name': names, 'Score': scores})
print(tr_split)


# In[44]:


sns.heatmap(diabetes.corr())


# In[46]:


sns.pairplot(diabetes,hue='Outcome')


# In[47]:


X=diabetes.drop('Outcome',axis=1)
y=diabetes['Outcome']


# In[48]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=0)


# In[49]:


from sklearn.preprocessing import StandardScaler
scaling_x=StandardScaler()
X_train=scaling_x.fit_transform(X_train)
X_test=scaling_x.transform(X_test)


# In[50]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc.predict(X_test)
rfc.score(X_test, y_test)


# In[85]:


def classify_with_rfc(X,Y):
    x = diabetes[[X,Y]].values
    y = diabetes['Outcome'].astype(int).values
    rfc = RandomForestClassifier()
    rfc.fit(x,y)
    
feat = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']
size = len(feat)
for i in range(0,size):
    for j in range(i+1,size):
        classify_with_rfc(feat[i],feat[j])


# In[82]:


from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


# In[86]:


from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))


# In[ ]:




