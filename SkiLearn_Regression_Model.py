from skimage.metrics import mean_squared_error
from sklearn import linear_model, preprocessing
from Data_Processor import DataProcessor
from File_Processor import FileProcessor
import pandas as pd

def main():
    file_processor = FileProcessor();
    data_processor = DataProcessor();

    # Read X array and Y array from file
    inputData, outputData = file_processor.read_file()
    # print(inputData, outputData)
    inputData =pd.DataFrame(preprocessing.scale(inputData))
    outputData =pd.DataFrame(preprocessing.scale(outputData))
    # seperate traindata and  test data
    xTrain, xTest, yTrain, yTest = data_processor.split_data(inputData, outputData)
    regression = linear_model.LinearRegression()
    regression.fit(xTrain, yTrain)
    yPred = regression.predict(xTest)
    # The coefficients
    print('Weights: \n', regression.coef_[0])
    # The mean squared error
    print('Mean squared error: %.2f'
          % mean_squared_error(yTest.to_numpy(), yPred))


main()
