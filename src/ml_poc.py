import pandas as pd
from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent.parent


def build_ml_dataset():

    vendor_a = pd.read_csv(
        BASE_DIR / "vendor_a_sensor_events.csv"
    )

    outcomes = pd.read_csv(
        BASE_DIR / "ground_truth_outcomes.csv"
    )

    # Parse timestamps
    vendor_a["event_time"] = pd.to_datetime(
        vendor_a["event_time"],
        errors="coerce",
        utc=True
    )

    outcomes["outcome_start_at"] = pd.to_datetime(
        outcomes["outcome_start_at"],
        errors="coerce",
        utc=True
    )

    # Binary target
    # 0 = No Issue
    # 1 = Adverse Outcome
    outcomes["target"] = (
        outcomes["outcome_type"] != "no_issue"
    ).astype(int)

    training_rows = []

    for _, outcome in outcomes.iterrows():

        machine_id = outcome["machine_id"]
        outcome_time = outcome["outcome_start_at"]
        window_days = outcome["label_window_days"]

        window_start = (
            outcome_time
            - pd.Timedelta(days=window_days)
        )

        # Use only historical data BEFORE the outcome
        machine_events = vendor_a[
            (vendor_a["machine_id"] == machine_id)
            & (vendor_a["event_time"] >= window_start)
            & (vendor_a["event_time"] < outcome_time)
        ].copy()

        if machine_events.empty:
            continue

        machine_events["temperature_c"] = pd.to_numeric(
            machine_events["temperature_c"],
            errors="coerce"
        )

        machine_events["vibration_mm_s"] = pd.to_numeric(
            machine_events["vibration_mm_s"],
            errors="coerce"
        )

        machine_events["confidence"] = pd.to_numeric(
            machine_events["confidence"],
            errors="coerce"
        )

        training_rows.append(
            {
                "machine_id": machine_id,
                "outcome_time": outcome_time,

                "avg_temperature_c":
                    machine_events["temperature_c"].mean(),

                "max_temperature_c":
                    machine_events["temperature_c"].max(),

                "avg_vibration_mm_s":
                    machine_events["vibration_mm_s"].mean(),

                "max_vibration_mm_s":
                    machine_events["vibration_mm_s"].max(),

                "avg_sensor_confidence":
                    machine_events["confidence"].mean(),

                "event_count":
                    len(machine_events),

                "target":
                    outcome["target"]
            }
        )

    ml_dataset = pd.DataFrame(training_rows)

    return ml_dataset


def train_ml_model(ml_data):

    # Sort chronologically
    ml_data = (
        ml_data
        .sort_values("outcome_time")
        .reset_index(drop=True)
    )

    feature_columns = [
        "avg_temperature_c",
        "max_temperature_c",
        "avg_vibration_mm_s",
        "max_vibration_mm_s",
        "avg_sensor_confidence",
        "event_count"
    ]

    X = ml_data[feature_columns]
    y = ml_data["target"]

    # Time-based split
    split_index = int(len(ml_data) * 0.80)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print("\nTraining rows:", len(X_train))
    print("Testing rows:", len(X_test))

    print("\nTraining target distribution:")
    print(y_train.value_counts())

    print("\nTesting target distribution:")
    print(y_test.value_counts())

    # Simple, explainable POC model
    model = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42
                )
            )
        ]
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    probabilities = (
        model.predict_proba(X_test)[:, 1]
    )

    print("\nML POC Evaluation")
    print("-------------------------")

    print(
        "Accuracy:",
        round(
            accuracy_score(
                y_test,
                predictions
            ),
            3
        )
    )

    print(
        "Precision:",
        round(
            precision_score(
                y_test,
                predictions,
                zero_division=0
            ),
            3
        )
    )

    print(
        "Recall:",
        round(
            recall_score(
                y_test,
                predictions,
                zero_division=0
            ),
            3
        )
    )

    print(
        "F1 Score:",
        round(
            f1_score(
                y_test,
                predictions,
                zero_division=0
            ),
            3
        )
    )

    if y_test.nunique() == 2:
        print(
            "ROC-AUC:",
            round(
                roc_auc_score(
                    y_test,
                    probabilities
                ),
                3
            )
        )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    return model


if __name__ == "__main__":

    ml_data = build_ml_dataset()

    print("\nML training dataset created successfully.")
    print("Rows:", len(ml_data))

    print("\nTarget distribution:")
    print(
        ml_data["target"]
        .value_counts()
    )

    model = train_ml_model(
        ml_data
    )