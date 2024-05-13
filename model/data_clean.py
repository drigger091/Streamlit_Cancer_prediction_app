
import pandas as pd

def clean_data():
    try:
        data = pd.read_csv("G:\Breast_cancer_prediction\data\data.csv")
    except FileNotFoundError:
        print("File not found. Please check the file path.")
        return None
    except Exception as e:
        print("An error occurred while reading the file:", e)
        return None
    
    try:
        data.drop(["Unnamed: 32",'id'], axis=1, inplace=True)
        data['diagnosis'] = data["diagnosis"].map({'M':1,'B':0})
        
        return data
    except Exception as e:
        print("An error occurred while cleaning the data:", e)
        return None



