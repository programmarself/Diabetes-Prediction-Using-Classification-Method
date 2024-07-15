#!/usr/bin/env python
# coding: utf-8

# # Diabetes Prediction using classification method 

# ### Import Libraries

# In[84]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# ### Load Datasets

# In[85]:


data = pd.read_csv("diabetes.csv")


# ### Shape of Data

# In[86]:


data.shape


# ### First 8 Rows of Data

# In[87]:


data.head(8)


# ## Last 7 Rows of Data

# In[88]:


data.tail(7)


# In[89]:


data.describe()


# In[90]:


data.info()


# In[91]:


data.value_counts()


# In[92]:


data.columns


# ### Checking Null Values

# In[93]:


data.isnull().sum()


# ### Diabetes Distribution

# In[94]:


#Finding Class Distribution Percentage
print(data['Outcome'].value_counts(ascending=True))
print(data['Outcome'].value_counts(1,ascending=True).apply(lambda x: format(x, '%')))
print()
# Plot the bar chart
data['Outcome'].value_counts(normalize=True).plot(kind='barh',figsize=(10, 2), color=['blue', 'red']).spines[['top', 'right']].set_visible(False)
plt.title('Diabetes Distribution (%)', fontsize=18)
plt.yticks(ticks=[0,1], labels=['Non-Diabetic', 'Diabetic'])
plt.show()


# ### Exploratory Data Analysis

# In[95]:


data.corr()


# ### Correlation Matrix

# In[96]:


plt.figure(figsize = (12,10))

sns.heatmap(data.corr(), annot =True)


# In[97]:


data.hist(figsize=(18,12))
plt.show()


# In[98]:


features = ['Glucose', 'BloodPressure', 'Insulin', 'BMI', 'Age', 'SkinThickness']
plt.figure(figsize=(14, 10))

for i, feature in enumerate(features, start=1):
    plt.subplot(2, 3, i)
    sns.boxplot(x=feature, data=data)

plt.tight_layout()
plt.show()


# In[99]:


import seaborn as sns
import matplotlib.pyplot as plt
mean_col = ['Glucose', 'BloodPressure', 'Insulin', 'Age', 'Outcome', 'BMI']
sns.pairplot(data[mean_col])
plt.show()


# In[100]:


sns.boxplot(x='Outcome',y='Insulin',data=data)


# In[101]:


sns.regplot(x='BMI', y= 'Glucose', data=data)


# In[102]:


sns.relplot(x='BMI', y= 'Glucose', data=data)


# In[103]:


sns.scatterplot(x='Glucose', y= 'Insulin', data=data)


# In[104]:


sns.jointplot(x='SkinThickness', y= 'Insulin', data=data)


# In[105]:


sns.pairplot(data,hue='Outcome')


# In[106]:


sns.lineplot(x='Glucose', y= 'Insulin', data=data)


# In[107]:


sns.stripplot(x='Glucose', y='Insulin', data=data, jitter=True, size=3)


# In[108]:


sns.barplot(x="SkinThickness", y="Insulin", data=data[150:180])
plt.title("SkinThickness vs Insulin",fontsize=15)
plt.xlabel("SkinThickness")
plt.ylabel("Insulin")
plt.show()
plt.style.use("ggplot")


# In[109]:


plt.figure(figsize=(5,5))
sns.barplot(x="Glucose", y="Insulin", data=data[120:130])
plt.title("Glucose vs Insulin",fontsize=15)
plt.xlabel("Glucose")
plt.ylabel("Insulin")
plt.show()


# ## Pre-process,Training and Testing Data

# In[123]:


x = data.drop(columns = 'Outcome')

y = data['Outcome']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)


# ### Train the Neural Network model

# In[124]:


from sklearn.neural_network import MLPClassifier
nn_model = MLPClassifier(hidden_layer_sizes=(100,50), max_iter=500, activation='relu', solver='adam', random_state=42)
nn_model.fit(X_train, y_train)


# ### MODELS

# **1. Logistic Regression**

# In[125]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

from sklearn.metrics import accuracy_score
LRAcc = accuracy_score(y_pred,y_test)
print('Logistic Regression accuracy is: {:.2f}%'.format(LRAcc*100))


# **2. SVM**

# In[126]:


from sklearn.svm import SVC  # Correct import statement
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

model = SVC()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

SVMAcc = accuracy_score(y_test, y_pred)
print('SVM accuracy is: {:.2f}%'.format(SVMAcc * 100))


# **3. Decison Tree**

# In[127]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

DTAcc = accuracy_score(y_test, y_pred)
print('Decision Tree accuracy is: {:.2f}%'.format(DTAcc * 100))


# ### Compare Models

# In[128]:


compare = pd.DataFrame({'Models Trained': ['Logistic Regression', 'SVM', 'Decision Tree'],
                        'Accuracy': [LRAcc*100, SVMAcc*100, DTAcc*100]})
compare.sort_values(by='Accuracy', ascending=False)


# ### Plotting Model Comparison

# In[130]:


compare.plot(x='Models Trained', y='Accuracy', kind='bar', color=['blue', 'green', 'orange'])


# **From the comparison plot, among the 3 Machine Learning Models, Logistic Regression had achieved the highest accuracy of 74.025974%.**
