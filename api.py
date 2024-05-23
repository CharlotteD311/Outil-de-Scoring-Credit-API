# API

# Importation des librairies
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, create_model
import pandas as pd
import joblib
import os
import uvicorn

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Chemins relatifs pour les fichiers de modèle et de données
best_model_path = os.path.join(BASE_DIR, "Models", "best_model.pkl")
explainer_path = os.path.join(BASE_DIR, "Models", "explainer.pkl")
data_path = os.path.join(BASE_DIR, "Data", "test_selected_data.csv")

# Chargement du modèle et de l'explainer SHAP
model = joblib.load(best_model_path)
explainer = joblib.load(explainer_path)

# Chargement des données (obtention des noms de colonnes)
df = pd.read_csv(data_path, index_col=0)
fields = {
    col: (int, ...) if pd.api.types.is_integer_dtype(df[col]) else
    (float, ...) if pd.api.types.is_float_dtype(df[col]) else
    (str, ...) for col in df.columns
}
DynamicClientInput = create_model('DynamicClientInput', **fields)


class PredictionOutput(BaseModel):
    client_id: int
    probability: float
    class_prediction: str
    shap_values: dict


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOutput)
async def predict(data: DynamicClientInput):
    try:
        # Convertion des données d'entrée en df
        input_df = pd.DataFrame([data.dict()])

        # Extraction de l'ID du client
        client_id = input_df.index[0]

        # Prédiction
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_df)[:, 1][0] * 100
        else:
            proba = model.predict(input_df)[0] * 100

        pred = 0 if proba < 52 else 1


        # Calcul des valeurs SHAP pour l'entrée donnée
        shap_values = explainer(input_df)
        shap_values_dict = {f: v for f, v in zip(input_df.columns, shap_values.values[0])}

        # Réponse
        response = {
            "client_id": client_id,
            "probability": round(proba, 2),
            "class_prediction": "Refusé" if pred == 1 else "Accepté",
            "shap_values": shap_values_dict
        }
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
