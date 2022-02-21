# Neural_Network_Charity_Analysis

## Overview
The Alphabet Soup business gives money to charity. One struggle they face is to know which charity's to donate to and how much money to donate. The purpose of this analysis is to create a binary classifier that is capable of predicting whether or not applicants will be successful if they are funded by Alphabet Soup. This will be done using a Neural Network Model.

## Resources
- Data Sources: charity_data.csv
- Tools: Python 3.8.8, Scikit-Learn, Pandas, TensorFlow, Jupyter Notebook

## Results
- Data Preprocessing
  - The column titled "IS_SUCCESSFUL" would be considered our target variable. It contains binary code (1 for successful and 0 for not successful), which is what we are basing our model off of.
  - The features for our model include the columns "APPLICATION_TYPE", "AFFILIATION", "CLASSIFICATION", "USE_CASE", "ORGANIZATION", "STATUS", "INCOME_AMT", "SPECIAL_CONSIDERATIONS", and "ASK_AMT". These variables all contribute to whether or not an applicant will be successful if donated to.
  - The variables that were removed from our model were "EIN" and "NAME" as these do not contribute important information to the analysis.
- Compiling, Training, and Evaluating the Model
  - Our deep learning model was composed of two hidden layers, the first hidden layer had 80 neurons and the second had 30. The input layer has 43 features and 25,724 total samples.
  - Our first hidden layers used the ReLu activation function, as did our second hidden layer. This activation function returns a value from 0 to infinity, and is this most used function due to its simplifying output. This is why it is used here. Our ouput layer used the Sigmoid activation function because it transforms the output to a range between 0 and 1. We are creating a binary classifier so this is the output we want.
  - The image below shows what the structure of our first model looks like
  - ![Model](https://github.com/RyleeJensen/Neural_Network_Charity_Analysis/blob/main/Images/SequentialModel.png)
  - After compiling, training, and evaluating the model, we got an accuracy score of 72.8%. This is under 75%, so it did not achieve the target model performance.
- Increasing Model Performance
  - There were a handful of different steps we took to attempt to increase the model's performance.
  - First, we added more neurons the first hidden layers. We changed it from the original 80 to 100. The image below shows the new structure of the model.
  - ![AddingNeurons](https://github.com/RyleeJensen/Neural_Network_Charity_Analysis/blob/main/Images/AddingNeurons.png)
  - This hardly increased the accuracy of the model at all. Originally it was 72.8%, and increasing the neurons changed it to 72.9%. Definitely did not help us achieve the target model.
  - Next, we tried to add another hidden layer. The neurons we used were 80, 30, and 10 respectively. The image below shows the new structure of the model.
  - ![AddingHiddenLayers](https://github.com/RyleeJensen/Neural_Network_Charity_Analysis/blob/main/Images/AddingHiddenLayers.png)
  - Again, this did not improve the accuracy of our model to achieve target performance. We got an accuracy of 73.0%
  - Finally, we went back to two hidden layers (keeping the original neurons of 80 and 30) and decided to change the activation functions. Rather than using the ReLu function for both hidden layers, we used the Tanh activation function. Still, this did not help improve the model as we got an accuracy score of 72.8%.

## Summary
Although we tried many different things to optimize our neural network model, we could not seem to reach a target model perfomance of 75% or higher. Everything we did kept the accuracy at about 73% (although nothing we did dramatically decreased the accuracy, which is good!). The purpose of this model was to create a binary classifier, and because of that, I believe we could try using a Supervised Machine Learning Model (such as the Random Forest Classifier). This model would combine many decision trees together to generate a classified output. It would be interesting to see how this model compares against our Neural Network Model as sometimes a more complicated model does not always produce better results.
