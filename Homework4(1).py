#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()


# In[2]:


breast_data = cancer.data
breast_data.shape


# In[3]:


breast_labels = cancer.target
breast_labels.shape


# In[4]:


labels = np.reshape(breast_labels,(569,1))
final_breast_data = np.concatenate([breast_data,labels],axis=1)
final_breast_data.shape


# In[5]:


breast_dataset = pd.DataFrame(final_breast_data)
features = cancer.feature_names
features


# In[6]:


features_labels = np.append(features,'label')
breast_dataset.columns = features_labels
breast_dataset.head()


# In[7]:


breast_dataset['label'].replace(0, 'Benign',inplace=True)
breast_dataset['label'].replace(1, 'Malignant',inplace=True)
breast_dataset.tail()


# In[8]:


from sklearn.preprocessing import StandardScaler
x = breast_dataset.loc[:, features].values
x = StandardScaler().fit_transform(x) # normalizing the features
x.shape


# In[9]:


np.mean(x),np.std(x)


# In[10]:


feat_cols = ['feature'+str(i) for i in range(x.shape[1])]
normalised_breast = pd.DataFrame(x,columns=feat_cols)
normalised_breast.tail()


# In[11]:


from sklearn.decomposition import PCA
pca_breast = PCA(n_components=2)
principalComponents = pca_breast.fit_transform(x)


# In[12]:


principalDf = pd.DataFrame(data = principalComponents
                                   , columns = ['principal component 1'
                                   , 'principal component 2'])
principalDf.tail()


# In[13]:


print('Explained variation per principal component: {}'.format(pca_breast.explained_variance_ratio_))


# In[14]:


import matplotlib.pyplot as plt
plt.figure()
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Principal Component - 1',fontsize=20)
plt.ylabel('Principal Component - 2',fontsize=20)
plt.title("Principal Component Analysis of Breast Cancer Dataset",fontsize=20)
targets = ['Benign', 'Malignant']
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = breast_dataset['label'] == target
    plt.scatter(principalDf.loc[indicesToKeep, 'principal component 1']
               , principalDf.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)

plt.legend(targets,prop={'size': 15})


# In[15]:


df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))
df_cancer.head()


# In[16]:


sns.pairplot(df_cancer, hue = 'target', vars = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness'])


# In[17]:


plt.figure(figsize=(20,12))
sns.heatmap(df_cancer.corr(), annot=True)


# In[18]:


X = df_cancer.drop(['target'], axis =1)
X.head()


# In[19]:


y = df_cancer['target']
y.head()


# In[20]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 20)


# In[21]:


print ('The size of our training "X" is', X_train.shape)
print ('\n')
print ('The size of our testing "X" is', X_test.shape)
print ('\n')
print ('The size of our training "y" is', y_train.shape)
print ('\n')
print ('The size of our testing "y" is', y_test.shape)


# In[22]:


from sklearn.svm import SVC
svc_model = SVC()


# In[23]:


svc_model.fit(X_train, y_train)


# In[24]:


y_predict = svc_model.predict(X_test)


# In[25]:


from sklearn.metrics import classification_report, confusion_matrix
cm = np.array(confusion_matrix(y_test, y_predict, labels=[1,0]))
confusion = pd.DataFrame(cm, index = ['is_cancer', 'is_healthy'],
                        columns=['predicted_cancer', 'predicted_healthy'])
confusion


# In[26]:


print(classification_report(y_test, y_predict))


# In[27]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)


# In[28]:


kfold = KFold(n_splits=5, random_state=0, shuffle=True)
model = LogisticRegression(solver='liblinear')
results = cross_val_score(model, X, y, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))


# In[29]:


kfold = KFold(n_splits=10, random_state=0, shuffle=True)
model = LogisticRegression(solver='liblinear')
results = cross_val_score(model, X, y, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))


# In[30]:


kfold = KFold(n_splits=14, random_state=0, shuffle=True)
model = LogisticRegression(solver='liblinear')
results = cross_val_score(model, X, y, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))

