from __future__ import division
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import xlrd
from sklearn.metrics import hamming_loss
import numpy as np
def getDataSet(filename):
    workbook = xlrd.open_workbook(filename)
    sheet = workbook.sheet_by_index(0)
    data = [[sheet.cell_value(r,c) for c in range(sheet.ncols)] for r in range
             (sheet.nrows)]
    return data[1:][:]


factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocessing(rawdata):
    #filtered = rawdata.encode('ascii','ignore')
    clean = stemmer.stem(rawdata)
    return clean

def getPerformance(target,prediction):
    target=target.astype('str')
    prediction=prediction.astype('str')
    targetfix=[]
    predictionfix=[]
    for x in range (0,len(target)):
        targetfix.append(target[x][0]+target[x][1]+target[x][2])
        predictionfix.append(prediction[x][0]+prediction[x][1]+prediction[x][2])
    targetfix=np.array(targetfix)
    predictionfix=np.array(predictionfix)
    return hamming_loss(targetfix,predictionfix)
def getHammingLoss(target,prediction):
    error=0
    for x in range (0,len(target)):
        for y in range(0,len(target[0])):
            if(prediction[x][y]!=target[x][y]):
                error+=1
    print (error)
    hammingloss=error*(1/3)*(1/(len(target)))
    return hammingloss