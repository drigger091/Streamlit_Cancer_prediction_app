from data_clean import clean_data
from create_model import create_model
import pickle as pickle



def main():

    data = clean_data()

    model,scaler = create_model(data)

    with open('model/model.pkl' ,'wb') as f:
        pickle.dump(model,f)
    with open('model/scaler.pkl','wb') as f:
        pickle.dump(scaler,f)
 
   
if __name__ == "__main__":
    main()
