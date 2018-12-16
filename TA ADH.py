
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
from sklearn.model_selection import train_test_split

# In[2]:
# ### Open File
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


# In[5]:
# ### Stopword Removal & Stemming
stopfactory = StopWordRemoverFactory()
stopword = stopfactory.create_stop_word_remover()

stemfactory = StemmerFactory()
stemmer = stemfactory.create_stemmer()

process = [stopword.remove(line) for line in rawText]
for x in range(len(process)):
    print (x)
    process[x]=stemmer.stem(process[x])
# In[7]:
# ### Tokenization
preprocess=[]
preprocess=[line.split() for line in process]



# In[8]:
# ### Bag of Words
bow =[]
for x in range(len(process)):
    bow+=preprocess[x]
bow =sorted(list(set(bow))) 
# In[10]:
# ## TF-IDF / vectorization
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
train, validation, labeltrain, labelval = train_test_split(extraction, label, test_size=0.30, random_state=42)
anjuran = [labeltrain[x][y] for x in range(len(labeltrain)) for y in range(0,3) if y==0] 
larangan =[labeltrain[x][y] for x in range(len(labeltrain)) for y in range(0,3) if y==1]
informasi = [labeltrain[x][y] for x in range(len(labeltrain)) for y in range(0,3) if y==2] 

trainAnjuran = [train[x] +[anjuran[x]] for x in range(len(train))]
trainLarangan = [train[x] +[larangan[x]] for x in range(len(train))]
trainInformasi = [train[x] + [informasi[x]] for x in range(len(train))]
# In[57]:
# ## Build Model & Prediction
modelA = RandomForestClassifier(rf_trees=80, rf_samples=1000)
modelL = RandomForestClassifier(rf_trees=80, rf_samples=1000)
modelI = RandomForestClassifier(rf_trees=80, rf_samples=1000)
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
performance = processor.getHammingLoss(labelval,prediction)
