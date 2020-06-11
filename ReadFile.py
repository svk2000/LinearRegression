import pandas as pd
import io
import requests
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


# dataset URL https://archive.ics.uci.edu/ml/datasets/Real+estate+valuation+data+set
# Read file from S3 bucket using pandas library
def read_file():
    url = "https://cs6375-vxs190040.s3.amazonaws.com/Real+estate+valuation+data+set.csv"
    content = requests.get(url).content
    records = pd.read_csv(io.StringIO(content.decode('utf-8')))
    return (records.iloc[:, :7], records.iloc[:, 7:8])

def split_data(inputData, outputData):
    return train_test_split(inputData, outputData, test_size=0.2, random_state=0)

def calch(xTrain,yTrain, weights):
    # initialize data calculate data
    h=[0] * xTrain.size
    for ri , row in enumerate(xTrain.values) :
        for ci, cell in enumerate(row[1:]) :
            h[ri] = h[ri] + (cell * weights[ci])
        h[ri] = h[ri] + 10
        if ri < 3 :
            print(h[ri],yTrain.values[ri])
    print("****************")
    return h
        
def mse(h, yTrain) :
    e = []
    # error
    for index, y in enumerate(yTrain.values) :
        e.append(h[index] - y[0])   
    # mean error
    # print("errors ", e[:10])
    esum=0
    for ei in e :
        esum = esum + ei ** 2
    
    mse = .5*esum /yTrain.values.size   

    return (mse, e)

def newWeights(learning, mseTuple , xTrain, weights) :

    newWeights =[]
    n = xTrain.values.size
    for index , oldWeight in enumerate( weights) :
       
        diffErr =0;
        for x in xTrain.values :
           diffErr = diffErr + mseTuple[1][index] * x[index +1 ] # adding +1 bacause of No is there in input list
        # newWeights.append( round(oldWeight - learning/n * ( mseTuple[0] ) + diffErr ,4) )
        newWeights.append( oldWeight - (learning/n)*   diffErr  )
    return newWeights
    

def main():
    # read file 
    inputData, outputData = read_file()
    # seperate traindata and  test data
    xTrain, xTest, yTrain, yTest = split_data(inputData, outputData)

    #initialize weights --by checking first row and removing number column
    weights= [1.0,.3,43,.098,9898,7.987] # * ( xTrain.values[0].size - 1 )

    #get Learning value
   

    for learning in range(0, 1 ) :
        # format of graphData (learning, mse , [weights])
        learning = learning  *.001  + .003
        graphData=[];
        iterations = 100
        # use 1000 iterations
        for iter in range(iterations):
            print("weights ", weights)
            h= calch(xTrain,yTrain,weights)
            
            # print(h)
            mseTuple = mse(h,yTrain)

            # update weights
            weights = newWeights(learning, mseTuple,xTrain, weights)

            
        #update Graph data.
        graphData.append((learning,iterations, mseTuple[0]))
    # print(graphData)

main()    
#print("xtest", type(xTest), xTest)
#print("ytrain",type(yTrain),yTrain)
#print("yTest",type(yTest),yTest)
# rf()