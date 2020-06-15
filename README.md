
Setting up
------------
- Install following packages in python path before executing python files

	- ***pandas***: To read file from S3 bucket and convert to data frames
	- ***sklearn***: For splitting data and running regression models
	- ***matplotlib***: To draw plots

### Copy following files into project folder
	Linear_Regression_Model.py: Implemented Gradient Descent without using any library
	SkiLearn_Regression_Model.py: Implemented Gradient Descent using Sklearn Linear model
	File_Processor.py: This class implements reading file from AWS S3 bucket and converting to Dataframes using pandas
	Data_Processor.py: This class implements data processing (Splitting train & test) and plotting using matplot

### Program Execution

- Executing Gradient Descent without using any library to calculate Mean Squared Error for Train and test data

		python Linear_Regression_Model.py >> output.txt

- Above program outputs two plots with below file names in the same project folder 
along with output txt file which records ***learning rate***, ***Number of iterations***, ***Train MSE***, ***Test MSE***

		MSE_vs_Iterations_LearningRate0.1.png
		MSE_vs_Iterations_LearningRate0.01.png
		output.txt

- Executing Gradient Descent using sklearn library

		python SkiLearn_Regression_Model.py >> output1.txt

- Above program outputs below file names in the same project folder which records ***weights caluclated***, ***Mean Squared Error***

		output1.txt
