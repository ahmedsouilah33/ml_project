from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from model_pipeline import preprocess_data, load_model, predict, prepare_data, save_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

app = Flask(__name__)

# Disable template caching for development
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Load initial model
model, scaler, trained_columns = load_model("knn_loan_model.pkl")


# ----------------------------
# Home & Predict Form
# ----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    global model, scaler, trained_columns

    if request.method == "POST":
        try:
            # Collect form data
            form_data = request.form.to_dict()
            df = pd.DataFrame([form_data])

            # Convert numeric fields
            numeric_fields = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]
            for col in numeric_fields:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Preprocess
            df_processed = preprocess_data(df, is_train=False)

            # Remove Loan_ID if present
            if "Loan_ID" in df_processed.columns:
                df_processed = df_processed.drop("Loan_ID", axis=1)

            # Add missing columns with default value 0
            for col in trained_columns:
                if col not in df_processed.columns:
                    df_processed[col] = 0

            # Reorder columns to match training data
            df_processed = df_processed[trained_columns]

            # Predict
            prediction = predict(model, scaler, df_processed)[0]
            
            # Render result page with prediction
            return render_template("result.html", result={"prediction": prediction})
        
        except Exception as e:
            # Handle errors gracefully
            print(f"Error during prediction: {str(e)}")
            return render_template("index.html", error=f"Error: {str(e)}")

    return render_template("index.html")


# ----------------------------
# Retrain Form
# ----------------------------
@app.route("/retrain", methods=["GET", "POST"])
def retrain():
    global model, scaler, trained_columns

    if request.method == "POST":
        try:
            # Get hyperparameters
            n_neighbors = int(request.form.get("n_neighbors", 19))
            metric = request.form.get("metric", "euclidean")
            weights = request.form.get("weights", "uniform")

            # Load data
            data = prepare_data("data/train.csv", "data/test.csv")
            X_train, X_test = data["X_train"], data["X_test"]
            y_train, y_test = data["y_train"], data["y_test"]

            # Evaluate old model
            old_pred = model.predict(X_test)
            old_acc = accuracy_score(y_test, old_pred)
            old_cm = confusion_matrix(y_test, old_pred).tolist()

            # Train new model
            new_model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, weights=weights)
            new_model.fit(X_train, y_train)

            # Evaluate new model
            new_pred = new_model.predict(X_test)
            new_acc = accuracy_score(y_test, new_pred)
            new_cm = confusion_matrix(y_test, new_pred).tolist()

            # Compare & keep best
            if new_acc > old_acc:
                save_model(new_model, scaler, "knn_loan_model.pkl", trained_columns=X_train.columns)
                model, scaler, trained_columns = load_model("knn_loan_model.pkl")
                chosen = "new"
            else:
                chosen = "old"

            comparison = {
                "old_model": {"accuracy": old_acc, "confusion_matrix": old_cm},
                "new_model": {"accuracy": new_acc, "confusion_matrix": new_cm, 
                             "hyperparameters": {"n_neighbors": n_neighbors, "metric": metric, "weights": weights}},
                "better_model": chosen
            }

            return render_template("retrain.html", comparison=comparison)
        
        except Exception as e:
            print(f"Error during retraining: {str(e)}")
            return render_template("retrain.html", comparison=None, error=f"Error: {str(e)}")

    return render_template("retrain.html", comparison=None)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)