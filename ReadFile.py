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
    return [records.iloc[:, :7], records.iloc[:, 7:8]]


def split_data(inputData, outputData):
    return train_test_split(inputData, outputData, test_size=0.2, random_state=0)


inputData, outputData = read_file()
xTrain, xTest, yTrain, yTest = split_data(inputData, outputData)
print(xTrain)
print(xTest)
print(yTrain)
print(yTest)
