
# coding: utf-8

# In[49]:

from __future__ import division
import csv
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import numpy as np
import math
import processor
from RandomForest import RandomForestClassifier


# In[2]:
# ### Open File
with open('hadits2k1.csv', encoding="utf8") as csvfile:
    lines=csv.reader(csvfile,delimiter=";")
    rawText = list(lines)
rawText=  [i[0] for i in rawText]

# In[3]:


with open('label2k1.csv', encoding ="utf8") as csvfile:
    lines =csv.reader(csvfile)
    label = list(lines)

# In[4]:
#labelpowerset
#labelfix = [label[x][0]+label[x][1]+label[x][2] for x in range(len(label))]
####TAMBAHANNNN#####
#labels1=[label[x][0]+label[x][1] for x in range(len(label))]
#labels2=[label[x][1]+label[x][2] for x in range(len(label))]
#labels3=[label[x][0]+label[x][2] for x in range(len(label))]
#def contigency_table(l):
#    a=l.count('11')
#    b=l.count('01')
#    c=l.count('10')
#    d=l.count('00')
#    return a,b,c,d
#a,b,c,d = contigency_table(labels1)
#chi = ((((a*d)-(b*c))**2)*(a+b+c+d))/((a+b)*(c+d)*(b+d)*(a+c))
dependent_labels = [str((int(label[x][0]) or int(label[x][1]))) for x in range(len(label))]
labellsp=[dependent_labels[x]+dependent_labels[x]+label[x][2] for x in range(len(label))]
labellp =[label[x][0]+label[x][1]+label[x][2] for x in range(len(label))]
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
        temp.append(preprocess[i].count(bow[j]))
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
dataset = [[process[x]]+extraction[x]+[labellp[x]]+[labellsp[x]] for x in range(len(extraction))]
fold1,fold2,fold3,fold4,fold5 =[],[],[],[],[]
for x in range(0,400):
    fold1.append(dataset[x])
for x in range(400,800):
    fold2.append(dataset[x])
for x in range(800,1200):
    fold3.append(dataset[x])
for x in range(1200,1600):
    fold4.append(dataset[x])
for x in range(1600,2000):
    fold5.append(dataset[x])
    
train=[]
test=[]
train.append(fold1+fold2+fold3+fold4)
test.append(fold5)
train.append(fold1+fold2+fold3+fold5)
test.append(fold4)
train.append(fold1+fold2+fold4+fold5)
test.append(fold3)
train.append(fold1+fold3+fold4+fold5)
test.append(fold2)
train.append(fold2+fold3+fold4+fold5)
test.append(fold1)

def split(dtrain,dtest):
    train = [dtrain[x][0:-2] for x in range(len(dtrain))]
    test = [dtest[x][0:-2] for x in range(len(dtest))]
    labeltrain= [dtrain[x][-1]for x in range(len(dtrain))]
    labeltest= [dtest[x][-2]for x in range(len(dtest))]
    return train,test,labeltrain,labeltest
index=[]
performance=[]
for x in range(5):
    trainf,testf,labeltrain,labeltest=split(train[x],test[x])
    model = RandomForestClassifier(rf_trees=80,rf_samples=1000)
    trainfix=[trainf[x][1:]+[labeltrain[x]] for x in range(len(labeltrain))]
    model.fit(trainfix)
    prediction=[]
    for y in range(0,len(testf)):
        prediction.append(model.predict(testf[y][1:]))
    hammingloss,index=(processor.getHammingLoss(labeltest,prediction))
    index.append(index)
    performance.append(hammingloss)
# ## Split Data
#train, validation, labeltrain, labelval = train_test_split(extraction, label, test_size=0.30, random_state=42)
#anjuran = [labeltrain[x][y] for x in range(len(labeltrain)) for y in range(0,3) if y==0] 
#larangan =[labeltrain[x][y] for x in range(len(labeltrain)) for y in range(0,3) if y==1]
#informasi = [labeltrain[x][y] for x in range(len(labeltrain)) for y in range(0,3) if y==2] 




