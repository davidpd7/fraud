from fraud.assets.model.models import Model
from fraud.assets.packages.predictors.predictors import Predictors
import os
model = Model()
predictors = Predictors()


data = os.path.join("C:\\Users\\David Pérez\\OneDrive\\Documentos\\GitHub\\tfm3\\fraud_detector\\fraud\\assets\\examples\\data.csv")


data = model.get_data_processed(data_path=data)

#predictions = model.predictor(model="model_1",data_path=data)

print(data)
