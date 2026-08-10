import pandas as pd

from data_ingestion import load_data
from normalization import (
    normalize_vendor_a,
    normalize_vendor_b,
    normalize_vendor_c,
)


def build_machine_features():
    """
    Create machine-level features from normalized vendor data.
    """

    asset_master, vendor_a, vendor_b, vendor_c, maintenance = load_data()

    valid_machine_ids = set(asset_master["machine_id"].dropna())

    # Normalize vendor datasets
    vendor_a = normalize_vendor_a(vendor_a, valid_machine_ids)
    vendor_b = normalize_vendor_b(vendor_b, valid_machine_ids)
    vendor_c = normalize_vendor_c(vendor_c, valid_machine_ids)

    # --------------------------------------------------
    # Vendor A features
    # --------------------------------------------------

    vendor_a["temperature_c"] = pd.to_numeric(
        vendor_a["temperature_c"],
        errors="coerce"
    )

    vendor_a["vibration_value"] = pd.to_numeric(
        vendor_a["vibration_value"],
        errors="coerce"
    )

    vendor_a_features = (
        vendor_a.groupby("machine_id")
        .agg(
            avg_temperature_c=("temperature_c", "mean"),
            max_temperature_c=("temperature_c", "max"),
            avg_vibration_mm_s=("vibration_value", "mean"),
            max_vibration_mm_s=("vibration_value", "max"),
            avg_sensor_confidence=("confidence_pct", "mean"),
            vendor_a_event_count=("event_id", "count"),
        )
        .reset_index()
    )

    # --------------------------------------------------
    # Vendor B features
    # --------------------------------------------------

    vendor_b["power_kw"] = pd.to_numeric(
        vendor_b["power_kw"],
        errors="coerce"
    )

    vendor_b_features = (
        vendor_b.groupby("machine_id")
        .agg(
            vendor_b_avg_temperature_c=("temperature_c", "mean"),
            vendor_b_max_temperature_c=("temperature_c", "max"),
            avg_power_kw=("power_kw", "mean"),
            avg_vendor_b_confidence=("confidence_pct", "mean"),
            vendor_b_alert_count=("readingId", "count"),
        )
        .reset_index()
    )

    # --------------------------------------------------
    # Vendor C features
    # --------------------------------------------------

    vendor_c["days_since_last_service"] = pd.to_numeric(
    vendor_c["days_since_last_service"],
    errors="coerce"
)
    confidence_mapping = {
    "high": 100,
    "medium": 60,
    "low": 30
}
    vendor_c["manual_confidence_score"] = (
    vendor_c["confidence_pct"]
    .str.lower()
    .map(confidence_mapping)
)
    vendor_c_features = (
    vendor_c.groupby("machine_id")
    .agg(
        avg_days_since_service=("days_since_last_service", "mean"),
        max_days_since_service=("days_since_last_service", "max"),
        avg_manual_confidence=("manual_confidence_score", "mean"),
        vendor_c_record_count=("record_id", "count"),
    )
    .reset_index()
)
    # --------------------------------------------------
    # Maintenance features
    # --------------------------------------------------

    maintenance["downtime_minutes"] = pd.to_numeric(
        maintenance["downtime_minutes"],
        errors="coerce"
    )

    maintenance_features = (
        maintenance.groupby("machine_id")
        .agg(
            maintenance_count=("work_order_id", "count"),
            total_downtime_minutes=("downtime_minutes", "sum"),
            avg_downtime_minutes=("downtime_minutes", "mean"),
        )
        .reset_index()
    )

    # --------------------------------------------------
    # Combine with asset master
    # --------------------------------------------------

    features = asset_master.copy()

    for feature_df in [
        vendor_a_features,
        vendor_b_features,
        vendor_c_features,
        maintenance_features,
    ]:
        features = features.merge(
            feature_df,
            on="machine_id",
            how="left"
        )

    return features


if __name__ == "__main__":

    machine_features = build_machine_features()

    print("\nMachine-level feature table created successfully.")
    print("Rows:", machine_features.shape[0])
    print("Columns:", machine_features.shape[1])

    print("\nSelected Features:")
    print(
        machine_features[
            [
                "machine_id",
                "avg_temperature_c",
                "max_temperature_c",
                "avg_vibration_mm_s",
                "avg_power_kw",
                "max_days_since_service",
                "maintenance_count",
                "total_downtime_minutes",
            ]
        ].round(2)
    )