import joblib
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Chargement du modèle et des objets de transformation
rf_model = joblib.load(os.path.join(os.path.dirname(__file__), "rf_grid_search.joblib"))
label_encoder = joblib.load(os.path.join(os.path.dirname(__file__), "label_encoder.joblib"))
one_hot = joblib.load(os.path.join(os.path.dirname(__file__), "one_hot.joblib"))
scaler = joblib.load(os.path.join(os.path.dirname(__file__), "scaler.joblib"))  # objet StandardScaler entraîné sur data_corr
pca = joblib.load(os.path.join(os.path.dirname(__file__), "pca.joblib"))         # objet PCA entraîné (avec n_components=5)

# Liste des variables d'entrée attendues (noms conformes aux données d'entraînement)
FEATURES = {
        "gender": str,
        "SeniorCitizen": int,
        "Partner": str,
        "Dependents": str,
        "tenure": int,
        "PhoneService": str,
        "PaperlessBilling": str,
        "OnlineSecurity": str, 
        "OnlineBackup": str, 
        "DeviceProtection": str, 
        "TechSupport": str, 
        "StreamingTV": str, 
        "StreamingMovies": str,
        "MonthlyCharges": float,
        "TotalCharges": float,
        "InternetService": str,
        "Contract": str,
        "PaymentMethod": str,
        "MultipleLines": str
    }

def preprocess_input(input_data: dict) -> pd.DataFrame:
    """
    Cette fonction prend un dictionnaire contenant les 18 variables d'entrée,
    le transforme en DataFrame, effectue la standardisation et la réduction de dimension par PCA.
    """
    # Dans le même ordre que lors de l'entraînement
    df = pd.DataFrame([input_data], columns=FEATURES.keys())
    
    # Encodage des variables binaires
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 
                    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                    'StreamingTV', 'StreamingMovies'
                ]
    for col in binary_cols:
        df[col] = LabelEncoder().fit_transform(df[col])
    
    # Encodage OneHot pour les variables à plusieurs catégories
    multi_cat_cols = ['InternetService', 'Contract', 'PaymentMethod', 'MultipleLines']
    df = pd.concat([df, pd.DataFrame(one_hot.transform(df[multi_cat_cols].values).toarray(), columns=one_hot.get_feature_names_out(multi_cat_cols))], axis=1)

    # Standardisation
    # Utiliser scaler.transform pour appliquer la transformation déjà apprise.
    df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(df[['tenure', 'MonthlyCharges', 'TotalCharges']])

    # Dropping
    df.drop(columns=['InternetService', 'Contract', 'MultipleLines', 'PaymentMethod', 
                    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                    'TechSupport', 'StreamingTV', 'StreamingMovies'
                    ], 
                    inplace=True
            )
    
    # 2. Réduction de dimension par PCA
    df_pca = pca.transform(df)
    
    # Renvoyer le tableau numpy ou le transformer en DataFrame avec des noms de colonnes personnalisés.
    df_pca = pd.DataFrame(df_pca, columns=[f"PC{i+1}" for i in range(pca.n_components_)])
    return df_pca

def predict_churn(input_data: dict) -> dict:
    """
    Cette fonction reçoit un dictionnaire des 18 variables d'entrée,
    réalise le prétraitement et renvoie la prédiction (0 ou 1) du modèle.
    """
    # Prétraiter l'entrée
    processed_data = preprocess_input(input_data)
    
    # Prédiction avec le modèle chargé
    prediction = rf_model.predict_proba(processed_data)
    result = int(prediction[0])

    # Retourner la prédiction et sa probabilité
    probabilities = model.predict_proba(processed_data)
    proba_churn = probabilities[0][1]
    
    return {"churn": result, "probability": proba_churn}
