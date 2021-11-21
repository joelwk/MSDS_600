"""This script is used to predict churn from customer data. The output will display customerID with an associated value of Churn or No Churn"""
import pandas as pd
from pycaret.classification import setup, compare_models, predict_model, create_model, save_model, load_model
#Load prediction model
model = load_model('nb')

#Loads churn data and creates new feature
def load_data(filepath):
    df = pd.read_csv(filepath, index_col='customerID')
    df['charge_ratio'] =  (df['TotalCharges'] / df['MonthlyCharges']) + df['tenure'] 
    return df

#Renames the output column and values 
def make_predictions(df):
    predictions = predict_model(model, data=df)
    predictions.rename({"Label": "Churn_Prediction"}, axis=1, inplace=True)
    predictions['Churn_Prediction'].replace({1: "Churn", 0:"No Churn"}, inplace=True)
    return predictions['Churn_Prediction']

#Special python variable that gets a value depending on how the script is executed
if __name__ == "__main__":
    df = load_data(r'C:\Users\jwkon\Desktop\School\MSDS600 - Introduction to Data Science\Week 5\Data\new_churn_data.csv')
    predictions = make_predictions(df)
    print('predictions: ')
    print(predictions)

