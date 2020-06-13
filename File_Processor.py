import pandas as pd
import io
import requests


# dataset URL https://archive.ics.uci.edu/ml/datasets/Real+estate+valuation+data+set
# The inputs are as follows
# X0 = Coefficient/Bias
# X1=the transaction date (for example, 2013.250=2013 March, 2013.500=2013 June, etc.)
# X2=the house age (unit: year)
# X3=the distance to the nearest MRT station (unit: meter)
# X4=the number of convenience stores in the living circle on foot (integer)
# X5=the geographic coordinate, latitude. (unit: degree)
# X6=the geographic coordinate, longitude. (unit: degree)
#
# The output is as follow
# Y= house price of unit area (10000 New Taiwan Dollar/Ping, where Ping is a local unit, 1 Ping = 3.3 meter squared)
# Read file from S3 bucket using pandas library
class FileProcessor:

    # Reads dataset for AWS S3 bucket
    def read_file(self):
        url = "https://cs6375-vxs190040.s3.amazonaws.com/Real+estate+valuation+data+set.csv"
        content = requests.get(url).content
        records = pd.read_csv(io.StringIO(content.decode('utf-8')))
        return records.iloc[:, :7], records.iloc[:, 7:8]

