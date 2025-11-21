import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Préprocessing
target_encoder = {"Y": 1, "N": 0}
preprocessing_params = {}


def encode_categorical_features(df, is_train=True):
    """
    Encoder les variables catégorielles du DataFrame.
    """
    df_encoded = df.copy()

    if "Gender" in df_encoded.columns:
        mode_val = df_encoded["Gender"].mode()[0] if is_train else "Male"
        df_encoded["Male"] = (df_encoded["Gender"].fillna(mode_val) == "Male").astype(
            int
        )
        df_encoded.drop("Gender", axis=1, inplace=True)

    if "Married" in df_encoded.columns:
        mode_val = df_encoded["Married"].mode()[0] if is_train else "Yes"
        df_encoded["Married_Yes"] = (
            df_encoded["Married"].fillna(mode_val) == "Yes"
        ).astype(int)
        df_encoded.drop("Married", axis=1, inplace=True)

    if "Dependents" in df_encoded.columns:
        df_encoded["Dependents"] = (
            df_encoded["Dependents"]
            .fillna("0")
            .replace({"0": 0, "1": 1, "2": 2, "3+": 3})
            .astype(int)
        )

    if "Education" in df_encoded.columns:
        df_encoded["Education_Graduate"] = (
            df_encoded["Education"] == "Graduate"
        ).astype(int)
        df_encoded.drop("Education", axis=1, inplace=True)

    if "Self_Employed" in df_encoded.columns:
        mode_val = df_encoded["Self_Employed"].mode()[0] if is_train else "No"
        df_encoded["Self_Employed_Yes"] = (
            df_encoded["Self_Employed"].fillna(mode_val) == "Yes"
        ).astype(int)
        df_encoded.drop("Self_Employed", axis=1, inplace=True)

    if "Property_Area" in df_encoded.columns:
        df_encoded = pd.concat(
            [
                df_encoded,
                pd.get_dummies(df_encoded["Property_Area"], prefix="Property_Area"),
            ],
            axis=1,
        )
        df_encoded.drop("Property_Area", axis=1, inplace=True)

    if "Credit_History" in df_encoded.columns:
        mode_val = df_encoded["Credit_History"].mode()[0] if is_train else 1.0
        df_encoded["Credit_History"] = df_encoded["Credit_History"].fillna(mode_val)

    if "LoanAmount" in df_encoded.columns:
        df_encoded["LoanAmount"] = df_encoded["LoanAmount"].fillna(
            df_encoded["LoanAmount"].mean()
        )
        # Clip extreme LoanAmount values at inference using stored preprocessing param
        try:
            clip_val = preprocessing_params.get("loanamount_clip", None)
            if (not is_train) and (clip_val is not None):
                df_encoded["LoanAmount"] = df_encoded["LoanAmount"].clip(upper=clip_val)
        except Exception:
            pass

    if "Loan_Amount_Term" in df_encoded.columns:
        df_encoded.drop("Loan_Amount_Term", axis=1, inplace=True)

    return df_encoded


def handle_missing_values(df, is_train=True):
    """
    Gérer les valeurs manquantes et stocker les paramètres de preprocessing.
    """
    df_clean = df.copy()
    if is_train:
        preprocessing_params["mean_LoanAmount"] = df_clean["LoanAmount"].mean()
        preprocessing_params["mode_Credit_History"] = df_clean["Credit_History"].mode()[
            0
        ]
    return df_clean


def preprocess_data(df, is_train=True):
    """Applique toutes les étapes de prétraitement"""
    return handle_missing_values(encode_categorical_features(df, is_train), is_train)


def prepare_data(train_path, test_path):
    """
    Charger et préparer les jeux de données pour l'entraînement et le test.
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    train_processed = preprocess_data(train, is_train=True)
    test_processed = preprocess_data(test, is_train=False)

    # Store a sensible upper bound for LoanAmount (99th percentile)
    try:
        preprocessing_params["loanamount_clip"] = train_processed["LoanAmount"].quantile(0.99)
    except Exception:
        preprocessing_params["loanamount_clip"] = None

    X = train_processed.drop(["Loan_ID", "Loan_Status"], axis=1, errors="ignore")
    y = train_processed["Loan_Status"].map(target_encoder)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_final_test = test_processed.drop(["Loan_ID"], axis=1, errors="ignore")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_final_test": X_final_test,
        "test_ids": (
            test_processed["Loan_ID"] if "Loan_ID" in test_processed.columns else None
        ),
    }


# ----------------------------
# Modèle KNN
# ----------------------------


def train_model(X_train, y_train, n_neighbors=15):
    """
    Entraîner un modèle KNN avec StandardScaler.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train_scaled, y_train)
    return model, scaler


def evaluate_model(model, scaler, X_test, y_test):
    """
    Évaluer le modèle KNN.
    """
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(
        f"Accuracy: {accuracy:.4f}, Correct: {np.sum(y_test==y_pred)}, Incorrect: {np.sum(y_test!=y_pred)}"
    )
    return {"accuracy": accuracy, "predictions": y_pred, "actual": y_test}


def predict(model, scaler, X_data):
    """Faire des prédictions sur de nouvelles données"""
    X_scaled = scaler.transform(X_data)
    return model.predict(X_scaled)


def save_model(model, scaler, file_path, trained_columns=None):
    """Sauvegarder le modèle, le scaler et les colonnes entrainées (optionnel)."""
    data_to_save = {
        "model": model,
        "scaler": scaler,
        "preprocessing_params": preprocessing_params,
        "target_encoder": target_encoder,
    }
    if trained_columns is not None:
        # ensure it's a plain list for portability
        data_to_save["trained_columns"] = list(trained_columns)

    with open(file_path, "wb") as f:
        pickle.dump(data_to_save, f)

    print(f"Modèle sauvegardé: {file_path}")


def load_model(file_path):
    """Charger un modèle, scaler et (optionnel) les colonnes utilisées lors de l'entraînement.

    Retourne `(model, scaler, trained_columns)` où `trained_columns` peut être `None`
    si le fichier n'en contient pas.
    """
    with open(file_path, "rb") as f:
        saved_data = pickle.load(f)
    global preprocessing_params, target_encoder
    preprocessing_params = saved_data.get("preprocessing_params", preprocessing_params)
    target_encoder = saved_data.get("target_encoder", target_encoder)
    trained_columns = saved_data.get("trained_columns", None)
    return saved_data["model"], saved_data["scaler"], trained_columns


def create_submission(predictions, test_ids, file_name="submission.csv"):
    """Créer un fichier CSV de soumission"""
    submission = pd.DataFrame(
        {
            "Loan_ID": test_ids,
            "Loan_Status": ["Y" if p == 1 else "N" for p in predictions],
        }
    )
    submission.to_csv(file_name, index=False)
    print(f"Fichier de soumission créé : {file_name}")
    return submission