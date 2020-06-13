import pandas as pd
import io
import requests
from sklearn import datasets, linear_model, preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class DataProcessor:

    # This method splits X attributes array and Y attributes array into Train & Test data
    def split_data(self, inputData, outputData):
        return train_test_split(inputData, outputData, test_size=0.1, random_state=0)

    # Calculate hypothesis values
    def calc_hypothesis(self, xTrain, weights):
        # initialize data calculate data
        hypo = [0] * xTrain.size

        for rowIndex, row in enumerate(xTrain.values):
            for colIndex, colValue in enumerate(row):
                hypo[rowIndex] = hypo[rowIndex] + (colValue * weights[colIndex])
        return hypo

    def calc_mse(self, hypothesis, yTrain):
        errors= []
        # error
        for index, y in enumerate(yTrain.values) :
            errors.append(hypothesis[index] - y[0])
        # mean error
        # print("errors ", e[:3],len(e))
        errors_sum = 0
        for error in errors:
            errors_sum = errors_sum + error ** 2
        # Calculating mean square error
        mse = .5 * errors_sum / yTrain.values.size
        return mse, errors


    def calc_new_weights(self, learningRate, errors , xTrain, weights) :
        newWeights =[]
        n = xTrain.values.size
        for colIndex , oldWeight in enumerate(weights) :
            errorDerivate =0;
            for rowIndex, row in enumerate(xTrain.values) :
                errorDerivate = errorDerivate + (errors[rowIndex] * row[colIndex])
            newWeights.append( oldWeight - ((learningRate/n)*   errorDerivate)  )
        return newWeights

    def plot_mse_iterations(self, iter, modelMse, testMse, learning):
        plt.plot(iter,modelMse,color='red',label='Train-MSE')
        plt.plot(iter,testMse,color='green',label='Test-MSE')
        plt.xlabel('ITERATIONS')
        plt.ylabel('MSE')
        plt.title(f'Learning rate: {learning}')
        plt.legend()
        plt.show()

