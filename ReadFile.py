import pandas as pd
import io
import requests


# Read file from S3 bucket using pandas library
def read_file():
    url = "https://cs6375-vxs190040.s3.amazonaws.com/auto-mpg.data"
    content = requests.get(url).content
    records = pd.read_csv(io.StringIO(content.decode('utf-8')))
    return records


print(read_file())
