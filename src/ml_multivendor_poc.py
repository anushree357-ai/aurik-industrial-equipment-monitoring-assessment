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

from ml_feature_builder import build_multivendor_ml_dataset


def train_multivendor_model():

    data = build_multivendor_ml_dataset()

    data = (
        data
        .sort_values("outcome_time")
        .reset_index(drop=True)
    )

    feature_columns = [
        "a_avg_temp",
        "a_max_temp",
        "a_avg_vibration",
        "a_max_vibration",
        "a_avg_confidence",
        "a_event_count",
        "b_avg_temp",
        "b_max_temp",
        "b_avg_power",
        "b_avg_confidence",
        "b_alert_count",
        "c_avg_days_since_service",
        "c_record_count",
        "maintenance_count",
        "maintenance_downtime"
    ]

    X = data[feature_columns]
    y = data["target"]

    # Time-based split
    split_index = int(len(data) * 0.80)

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
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\nMulti-Vendor ML POC Evaluation")
    print("--------------------------------")

    print(
        "Accuracy:",
        round(accuracy_score(y_test, predictions), 3)
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
    model = train_multivendor_model()