import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def build_multivendor_ml_dataset():

    vendor_a = pd.read_csv(BASE_DIR / "vendor_a_sensor_events.csv")
    vendor_b = pd.read_csv(BASE_DIR / "vendor_b_alert_events.csv")
    vendor_c = pd.read_csv(BASE_DIR / "vendor_c_inspection_maintenance.csv")
    maintenance = pd.read_csv(BASE_DIR / "maintenance_records.csv")
    outcomes = pd.read_csv(BASE_DIR / "ground_truth_outcomes.csv")

    # -----------------------------
    # Parse timestamps
    # -----------------------------

    vendor_a["event_time"] = pd.to_datetime(
        vendor_a["event_time"],
        errors="coerce",
        utc=True
    )

    vendor_b["event_time"] = pd.to_datetime(
        vendor_b["timestampMs"],
        unit="ms",
        errors="coerce",
        utc=True
    )

    vendor_c["event_time"] = pd.to_datetime(
        vendor_c["recorded_at"],
        errors="coerce",
        utc=True
    )

    maintenance["opened_at"] = pd.to_datetime(
        maintenance["opened_at"],
        errors="coerce",
        utc=True
    )

    outcomes["outcome_start_at"] = pd.to_datetime(
        outcomes["outcome_start_at"],
        errors="coerce",
        utc=True
    )

    # -----------------------------
    # Standardize machine IDs
    # -----------------------------

    vendor_b = vendor_b.rename(
        columns={"assetCode": "machine_id"}
    )

    vendor_c = vendor_c.rename(
        columns={"machine_ref": "machine_id"}
    )

    # -----------------------------
    # Numeric conversions
    # -----------------------------

    vendor_a["temperature_c"] = pd.to_numeric(
        vendor_a["temperature_c"],
        errors="coerce"
    )

    vendor_a["vibration_mm_s"] = pd.to_numeric(
        vendor_a["vibration_mm_s"],
        errors="coerce"
    )

    vendor_a["confidence"] = pd.to_numeric(
        vendor_a["confidence"],
        errors="coerce"
    )

    vendor_b["temperature_f"] = pd.to_numeric(
        vendor_b["temperature_f"],
        errors="coerce"
    )

    vendor_b["temperature_c"] = (
        vendor_b["temperature_f"] - 32
    ) * 5 / 9

    vendor_b["power_kw"] = pd.to_numeric(
        vendor_b["power_kw"],
        errors="coerce"
    )

    vendor_b["vendorConfidencePct"] = pd.to_numeric(
        vendor_b["vendorConfidencePct"],
        errors="coerce"
    )

    vendor_c["days_since_last_service"] = pd.to_numeric(
        vendor_c["days_since_last_service"],
        errors="coerce"
    )

    maintenance["downtime_minutes"] = pd.to_numeric(
        maintenance["downtime_minutes"],
        errors="coerce"
    )

    # -----------------------------
    # Binary target
    # -----------------------------

    outcomes["target"] = (
        outcomes["outcome_type"] != "no_issue"
    ).astype(int)

    rows = []

    for _, outcome in outcomes.iterrows():

        machine_id = outcome["machine_id"]
        outcome_time = outcome["outcome_start_at"]
        window_days = outcome["label_window_days"]

        window_start = (
            outcome_time
            - pd.Timedelta(days=window_days)
        )

        a = vendor_a[
            (vendor_a["machine_id"] == machine_id)
            & (vendor_a["event_time"] >= window_start)
            & (vendor_a["event_time"] < outcome_time)
        ]

        b = vendor_b[
            (vendor_b["machine_id"] == machine_id)
            & (vendor_b["event_time"] >= window_start)
            & (vendor_b["event_time"] < outcome_time)
        ]

        c = vendor_c[
            (vendor_c["machine_id"] == machine_id)
            & (vendor_c["event_time"] >= window_start)
            & (vendor_c["event_time"] < outcome_time)
        ]

        m = maintenance[
            (maintenance["machine_id"] == machine_id)
            & (maintenance["opened_at"] >= window_start)
            & (maintenance["opened_at"] < outcome_time)
        ]

        rows.append(
            {
                "machine_id": machine_id,
                "outcome_time": outcome_time,

                # Vendor A
                "a_avg_temp": a["temperature_c"].mean(),
                "a_max_temp": a["temperature_c"].max(),
                "a_avg_vibration": a["vibration_mm_s"].mean(),
                "a_max_vibration": a["vibration_mm_s"].max(),
                "a_avg_confidence": a["confidence"].mean(),
                "a_event_count": len(a),

                # Vendor B
                "b_avg_temp": b["temperature_c"].mean(),
                "b_max_temp": b["temperature_c"].max(),
                "b_avg_power": b["power_kw"].mean(),
                "b_avg_confidence": b["vendorConfidencePct"].mean(),
                "b_alert_count": len(b),

                # Vendor C
                "c_avg_days_since_service":
                    c["days_since_last_service"].mean(),

                "c_record_count": len(c),

                # Maintenance
                "maintenance_count": len(m),
                "maintenance_downtime":
                    m["downtime_minutes"].sum(),

                "target": outcome["target"]
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":

    df = build_multivendor_ml_dataset()

    print("\nMulti-vendor ML dataset created.")
    print("Rows:", len(df))
    print("Columns:", df.shape[1])

    print("\nTarget distribution:")
    print(df["target"].value_counts())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nSample:")
    print(df.head())