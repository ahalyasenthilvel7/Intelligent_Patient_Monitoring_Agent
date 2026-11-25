
import numpy as np
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier


FEATURES = [
    "heart_rate",
    "spo2",
    "temperature",
    "systolic_bp",
    "diastolic_bp",
    "respiratory_rate"
]


def generate_dataset():

    np.random.seed(42)

    normal_count = 2400
    review_count = 900
    urgent_count = 700

    normal = pd.DataFrame({
        "heart_rate": np.random.randint(60, 101, normal_count),
        "spo2": np.random.randint(95, 101, normal_count),
        "temperature": np.random.uniform(36.0, 37.5, normal_count),
        "systolic_bp": np.random.randint(100, 131, normal_count),
        "diastolic_bp": np.random.randint(60, 86, normal_count),
        "respiratory_rate": np.random.randint(12, 21, normal_count),
        "status": "Normal"
    })

    review = pd.DataFrame({
        "heart_rate": np.random.randint(100, 126, review_count),
        "spo2": np.random.randint(92, 97, review_count),
        "temperature": np.random.uniform(37.5, 38.6, review_count),
        "systolic_bp": np.random.randint(130, 166, review_count),
        "diastolic_bp": np.random.randint(85, 106, review_count),
        "respiratory_rate": np.random.randint(20, 28, review_count),
        "status": "Review"
    })

    urgent = pd.DataFrame({
        "heart_rate": np.random.randint(125, 151, urgent_count),
        "spo2": np.random.randint(85, 93, urgent_count),
        "temperature": np.random.uniform(38.5, 40.0, urgent_count),
        "systolic_bp": np.random.randint(165, 201, urgent_count),
        "diastolic_bp": np.random.randint(100, 121, urgent_count),
        "respiratory_rate": np.random.randint(27, 36, urgent_count),
        "status": "Urgent Review"
    })

    data = pd.concat(
        [normal, review, urgent],
        ignore_index=True
    )

    data = data.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    return data


def train_models():

    print("Generating synthetic dataset...")

    data = generate_dataset()

    X = data[FEATURES]
    y = data["status"]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    models = {

        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            random_state=42,
            class_weight="balanced"
        ),

        "XGBoost": XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.08,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=2
        )
    }

    results = {}
    trained_models = {}

    print("\nMODEL COMPARISON")
    print("=" * 50)

    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        precision = precision_score(
            y_test, predictions, average="macro"
        )

        recall = recall_score(
            y_test, predictions, average="macro"
        )

        f1 = f1_score(
            y_test, predictions, average="macro"
        )

        results[name] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        }

        trained_models[name] = model

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")

        print("\nClassification Report:")
        print(
            classification_report(
                y_test,
                predictions,
                target_names=encoder.classes_,
                zero_division=0
            )
        )

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, predictions))

    best_model_name = max(
        results,
        key=lambda name: results[name]["F1 Score"]
    )

    best_model = trained_models[best_model_name]

    os.makedirs("models", exist_ok=True)

    joblib.dump(
        best_model,
        "models/patient_model.pkl"
    )

    joblib.dump(
        encoder,
        "models/label_encoder.pkl"
    )

    joblib.dump(
        FEATURES,
        "models/features.pkl"
    )

    print("\n" + "=" * 50)
    print("FINAL MODEL SUMMARY")
    print("=" * 50)

    print("Selected Model:", best_model_name)

    print("\nComparison:")
    print(pd.DataFrame(results).T)

    print("\nModel files saved successfully!")

    return {
        "model": best_model,
        "encoder": encoder,
        "features": FEATURES,
        "results": results,
        "selected": best_model_name
    }


if __name__ == "__main__":
    train_models()
