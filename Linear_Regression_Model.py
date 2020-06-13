import pandas as pd
import io
import requests
from sklearn import datasets, linear_model, preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from Data_Processor import DataProcessor
from File_Processor import FileProcessor


def main():
    # Learning rates array
    learningRates = [0.1, 0.01]
    file_processor = FileProcessor()
    data_processor = DataProcessor()

    # Read X array and Y array from file
    inputData, outputData = file_processor.read_file()
    # print(inputData, outputData)
    inputData =pd.DataFrame(preprocessing.scale(inputData))
    outputData =pd.DataFrame(preprocessing.scale(outputData))
    # seperate traindata and  test data
    xTrain, xTest, yTrain, yTest = data_processor.split_data(inputData, outputData)
   
    # graphIter=[]
    # graphMse=[]
    #
    # testGraphMse=[]
    # return
    for rateIndex, learningRate in enumerate(learningRates):
        # format of graphData (learning, mse , [weights])
        graphDataIterations=[]
        graphDataMse=[]
        testGraphMse=[]
        weights= [.4,.1,.2,.125,.075,.05,.05]
        for iteration in range(1, 30* (rateIndex+1)):
            iterations = (10 * iteration)
            #training model
            for iter in range(iterations):
                # print("weights ", weights)
                hypothesis= data_processor.calc_hypothesis(xTrain, weights)
                
                # print(h)
                mseTuple = data_processor.calc_mse(hypothesis,yTrain)

                # update weights
                weights = data_processor.calc_new_weights(learningRate, mseTuple[1],xTrain, weights)

            graphDataIterations.append(iteration)
            graphDataMse.append(mseTuple[0])

            #testing model
            hypothesis = data_processor.calc_hypothesis(xTest, weights)
            mseTuple = data_processor.calc_mse(hypothesis, yTest)
            testGraphMse.append(mseTuple[0])
        data = {'Learning Rate': learningRate, 'Number of Iterations': graphDataIterations, 'Train MSE': graphDataMse, 'Test MSE': testGraphMse }
        df = pd.DataFrame(data=data)
        print('----------------')
        print(df.iloc[-1])
        print('Creating Plot for learning Rate: %.2f'% learningRate)
        data_processor.plot_mse_iterations(graphDataIterations, graphDataMse, testGraphMse, learningRate)


main()    
