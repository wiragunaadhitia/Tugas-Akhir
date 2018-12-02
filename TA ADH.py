
# coding: utf-8

# In[49]:

from __future__ import division
import csv
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import math
import processor
from RandomForest import RandomForestClassifier

# ## PREPROCESSING

# ### Open File

# In[2]:


with open('hadits2k1.csv', encoding="utf8") as csvfile:
    lines=csv.reader(csvfile)
    rawText = list(lines)
rawText=  [i[0] for i in rawText]


# In[3]:


with open('label2k1.csv', encoding ="utf8") as csvfile:
    lines =csv.reader(csvfile)
    label = list(lines)

# In[4]:

label = [[int(j) for j in i] for i in label]
anjuran,informasi,larangan=[],[],[]
anjuran = [label[x][y] for x in range(len(label)) for y in range(0,3) if y==0] 
larangan =[label[x][y] for x in range(len(label)) for y in range(0,3) if y==1]
informasi = [label[x][y] for x in range(len(label)) for y in range(0,3) if y==2] 


# ### Stopword Removal & Stemming

# In[5]:


stopfactory = StopWordRemoverFactory()
stopword = stopfactory.create_stop_word_remover()

stemfactory = StemmerFactory()
stemmer = stemfactory.create_stemmer()

process = [stopword.remove(line) for line in rawText]
for x in range(len(process)):
    print (x)
    process[x]=stemmer.stem(process[x])


# In[6]:




# ### Tokenization

# In[7]:


preprocess=[]
preprocess=[line.split() for line in process]


# ### Bag of Words

# In[8]:


bow =[]
for x in range(len(process)):
    bow+=preprocess[x]
bow =sorted(list(set(bow))) 


# In[9]:





# ## TF-IDF / vectorization

# In[10]:


TF = []
for i in range(len(preprocess)):
    temp=[]
    for j in range(len(bow)):
        temp.append(preprocess[i].count(bow[j])/len(preprocess[i]))
    TF.append(temp)     


# In[11]:


DF = []
for i in range(len(bow)):
    temp=0
    for j in range(len(preprocess)):
        if(bow[i] in preprocess[j]):
            temp+=1
    DF.append(temp)
    


# In[12]:


for i in range(len(preprocess)):
    for j in range(len(bow)): 
        TF[i][j]= TF[i][j] * math.log(len(preprocess)/DF[j])
extraction=TF.copy()    


# In[13]:





# ## Split Data



# In[35]:
#train,validation = [],[]
#train =np.array([extraction[x] for x in range(0,1600)])
#validation =np.array([extraction[x] for x in range(1600,2000)])
#
#targetAnjuran= np.array([anjuran[x] for x in range(0,1600)]).astype('int')
#targetLarangan= np.array([larangan[x] for x in range(0,1600)]).astype('int')
#targetInformasi= np.array([informasi[x] for x in range(0,1600)]).astype('int')
#
#targetvAnjuran= np.array([anjuran[x] for x in range(1600,2000)]).astype('int')
#targetvLarangan= np.array([larangan[x] for x in range(1600,2000)]).astype('int')
#targetvInformasi= np.array([informasi[x] for x in range(1600,2000)]).astype('int')

### REVISION #####
train = [extraction[x] for x in range(0,1600)]
validation = [extraction[x] for x in range(1600,2000)]
targetAnjuran = [anjuran[x] for x in range(0,1600)]
targetLarangan = [larangan[x] for x in range(0,1600)]
targetInformasi = [informasi[x] for x in range(0,1600)]


# In[54]:

#
#trainAnjuran = np.concatenate((train[:,0:],targetAnjuran[:,None]),axis=1).tolist()
#trainLarangan = np.concatenate((train[:,0:],targetLarangan[:,None]),axis=1)
#trainInformasi = np.concatenate((train[:,0:],targetInformasi[:,None]),axis=1)
#trainAnjuran

trainAnjuran = [train[x] +[targetAnjuran[x]] for x in range(len(train))]
trainLarangan = [train[x] +[targetLarangan[x]] for x in range(len(train))]
trainInformasi = [train[x] + [targetInformasi[x]] for x in range(len(train))]
# ## Build Model & Prediction

# In[57]:

modelA = RandomForestClassifier(rf_trees=50, rf_samples=1000)
modelL = RandomForestClassifier(rf_trees=50, rf_samples=1000)
modelI = RandomForestClassifier(rf_trees=50, rf_samples=1000)
modelA.fit(trainAnjuran)
modelL.fit(trainLarangan)
modelI.fit(trainInformasi)

predictA=[]
predictL=[]
predictI=[]
for x in range(0,len(validation)):
    predictA.append(modelA.predict(validation[x]))
    predictL.append(modelL.predict(validation[x]))
    predictI.append(modelI.predict(validation[x]))
    
prediction = [[predictA[x]]+[predictL[x]]+[predictI[x]] for x in range(len(predictA))]
labelvalidation = [label[x] for x in range(1600,2000)]
performance = processor.getHammingLoss(labelvalidation,prediction)


# In[22]:



