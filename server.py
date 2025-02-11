from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from p_churn_model.p_churn import predict_churn, FEATURES

app = FastAPI()

class ChurnInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    PaperlessBilling: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str 
    TechSupport: str
    StreamingTV: str 
    StreamingMovies: str
    MonthlyCharges: float
    TotalCharges: float
    InternetService: str
    Contract: str
    PaymentMethod: str
    MultipleLines: str

@app.post("/predict")
def predict(churn_input: ChurnInput):
    # Conversion de l'objet Pydantic en dictionnaire
    input_dict = churn_input.dict()
    
    # Vérification que le dictionnaire contient bien les variables attendues. 
    for feature, expected_type in FEATURES.items():
        if feature not in input_dict:
            raise HTTPException(status_code=400, detail=f"Missing feature: {feature}")
        if not isinstance(input_dict[feature], expected_type):
            raise HTTPException(status_code=400, detail=f"Incorrect type for feature: {feature}. Expected {expected_type.__name__}, got {type(input_dict[feature]).__name__}")

    # Appel de la fonction de prédiction
    result = predict_churn(input_dict)
    return result