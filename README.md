# p_churn-ai

## Lancer l'app:
```python
fastapi dev server.py
```
## Lancer une requête post:
```
curl  -X POST \
  'http://127.0.0.1:8000/predict/' \
  --header 'Accept: */*' \
  --header 'Content-Type: application/json' \
  --data-raw '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "Yes",
    "tenure": 34,
    "PhoneService": "Yes",
    "PaperlessBilling": "No",
    "OnlineSecurity": "Yes", 
    "OnlineBackup": "No", 
    "DeviceProtection": "Yes", 
    "TechSupport": "No internet service", 
    "StreamingTV": "Yes", 
    "StreamingMovies": "Yes",
    "MonthlyCharges": 30.5,
    "TotalCharges": 1000,
    "InternetService": "Fiber optic",
    "Contract": "Month-to-month",
    "PaymentMethod": "Electronic check",
    "MultipleLines": "No phone service"
}'
```