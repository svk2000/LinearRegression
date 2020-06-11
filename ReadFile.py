import pandas as pd
import io
import requests
from sklearn import datasets, linear_model, preprocessing
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
    return train_test_split(inputData, outputData, test_size=0.1, random_state=0)

def calch(xTrain,yTrain, weights):
    # initialize data calculate data
    bias = .005
    h=[0] * xTrain.size
    for ri , row in enumerate(xTrain.values) :
        for ci, cell in enumerate(row[1:]) :
            h[ri] = h[ri] + (cell * weights[ci])
        h[ri] = h[ri] + bias
        # if ri < 3 :
        #     print(h[ri],yTrain.values[ri])
    # print("****************")
    return h
        
def mse(h, yTrain) :
    e = []
    # error
    for index, y in enumerate(yTrain.values) :
        e.append(h[index] - y[0])   
    # mean error
    # print("errors ", e[:3],len(e))
    esum=0
    for ei in e :
        esum = esum + ei ** 2
    
    mse = .5*esum /yTrain.values.size   

    return (mse, e)

def newWeights(learning, errors , xTrain, weights) :

    newWeights =[]
    n = xTrain.values.size
    for colIndex , oldWeight in enumerate( weights) :
       
        diffErr =0;
        for rowNumber, row in enumerate(xTrain.values) :
           diffErr = diffErr + (errors[rowNumber] * row[colIndex +1 ]) # adding +1 bacause of No is there in input list
        # newWeights.append( round(oldWeight - learning/n * ( mseTuple[0] ) + diffErr ,4) )
        newWeights.append( oldWeight - ((learning/n)*   diffErr)  )
    return newWeights
    

def main():
    # read file 
    inputData, outputData = read_file()
    # print(type(inputData), type(outputData))

    inputData =pd.DataFrame(preprocessing.scale(inputData))
    outputData =pd.DataFrame(preprocessing.scale(outputData))

    # seperate traindata and  test data
    xTrain, xTest, yTrain, yTest = split_data(inputData, outputData)
    # print(type(xTrain), type(xTest),  type(yTrain), type(yTrain))

    #initialize weights --by checking first row and removing number column
    weights= [.3,.003,.098,0.4,0.99,.98] # * ( xTrain.values[0].size - 1 )

    # print(type(xTrain),xTrain)
    #preprocessing
    # xTrain =pd.DataFrame(preprocessing.scale(xTrain))
    # yTrain =pd.DataFrame(preprocessing.scale(yTrain))
    # print(type(xTrain),type([]),xTrain )
    #get Learning value
   
    # return
    for learning in range(0, 5 ) :
        # format of graphData (learning, mse , [weights])
        learning =  .1 /(10 ** learning)
        graphData=[];
        iterations = 500
        # use 1000 iterations
        for iter in range(iterations):
            # print("weights ", weights)
            h= calch(xTrain,yTrain,weights)
            
            # print(h)
            mseTuple = mse(h,yTrain)

            # update weights
            weights = newWeights(learning, mseTuple[1],xTrain, weights)
            # print(mseTuple[0])
            # if iter % 100 == 0 :
            #     print(iter, mseTuple[0])
        #update Graph data.
        graphData.append((learning,iterations, mseTuple[0]))
        print(graphData)

main()    
#print("xtest", type(xTest), xTest)
#print("ytrain",type(yTrain),yTrain)
#print("yTest",type(yTest),yTest)
# rf()