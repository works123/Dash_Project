import numpy as np
import joblib, pickle
import warnings
import pandas
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
import numpy as np

warnings.filterwarnings("ignore")



# saved model 
# load the model from disk
model_full_name = "model_file/finalized_model.sav"
loaded_model = joblib.load(model_full_name)


# consume new data 
def get_new_data(new_data):
    return new_data 

# preprocessing of new data 
def preprocess_data(new_data):
    #preprocessing
    new_transform = np.array(new_data).reshape(-1,1)
    return new_transform

# predict result for new data
def predict_result(new_data_transform):
    prediction = loaded_model.predict(np.array(new_data_transform))
    return round(prediction[0], 2)
  
# predict result for new data specifying a particular model type
def predict_result_with_model_type(model_type, new_data_transform):
    prediction = model_type.predict(new_data_transform)
   
    return round(prediction[0], 2)