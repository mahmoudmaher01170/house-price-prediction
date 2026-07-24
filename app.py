
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

xgb_model = joblib.load('xgb_model.pkl')
num_imputer = joblib.load('num_imputer.pkl')
cat_imputer = joblib.load('cat_imputer.pkl')
encoder = joblib.load('encoder.pkl')
top_locations = joblib.load('top_locations.pkl')

class HouseFeatures(BaseModel):
    carpet_area_sqft: float
    floor_num: float
    Bathroom: float
    Balcony: float
    Car_Parking: float
    location: str
    Status: str
    Transaction: str
    Furnishing: str
    facing: str
    overlooking: str
    Ownership: str

@app.post("/predict")
def predict_price(data: HouseFeatures):
    input_data = pd.DataFrame([data.dict()])
    input_data.rename(columns={'Car_Parking': 'Car Parking'}, inplace=True)
    
    input_data['location_grouped'] = np.where(
        input_data['location'].isin(top_locations),
        input_data['location'],
        'other'
    )
    
    num_cols = ['carpet_area_sqft', 'floor_num', 'Bathroom', 'Balcony', 'Car Parking']
    cat_cols = ['location_grouped', 'Status', 'Transaction', 'Furnishing', 'facing', 'overlooking', 'Ownership']
    
    input_data[num_cols] = num_imputer.transform(input_data[num_cols])
    input_data[cat_cols] = cat_imputer.transform(input_data[cat_cols])
    
    encoded_cats = pd.DataFrame(
        encoder.transform(input_data[cat_cols]),
        columns=encoder.get_feature_names_out(cat_cols)
    )
    
    final_features = pd.concat([input_data[num_cols], encoded_cats], axis=1)
    predicted_price = xgb_model.predict(final_features)[0]
    
    return {"predicted_price": round(float(predicted_price), 2)}
