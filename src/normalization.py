import pandas as pd

from data_ingestion import load_data


G_TO_MM_S2 = 9.80665


def normalize_vendor_a(df, valid_machine_ids):
    """Normalize Vendor A sensor data."""

    data = df.copy()

    # Remove rows without a valid machine
    data = data[
        data["machine_id"].isin(valid_machine_ids)
    ].copy()

    # Standard column names
    data = data.rename(columns={
        "event_time": "event_timestamp",
        "vibration_mm_s": "vibration_value",
        "temperature_c": "temperature_c",
        "confidence": "confidence_pct"
    })

    data["vendor"] = "Vendor_A"

    # Parse timestamp
    data["event_timestamp"] = pd.to_datetime(
        data["event_timestamp"],
        errors="coerce",
        utc=True
    )

    # Vendor A confidence appears as 0-1, convert to percentage
    data["confidence_pct"] = (
        pd.to_numeric(data["confidence_pct"], errors="coerce") * 100
    )

    return data


def normalize_vendor_b(df, valid_machine_ids):
    """Normalize Vendor B alert data."""

    data = df.copy()

    # Standard machine identifier
    data = data.rename(columns={
        "assetCode": "machine_id",
        "vendorConfidencePct": "confidence_pct"
    })

    # Remove invalid machine IDs
    data = data[
        data["machine_id"].isin(valid_machine_ids)
    ].copy()

    # Convert epoch milliseconds to timestamp
    data["event_timestamp"] = pd.to_datetime(
        data["timestampMs"],
        unit="ms",
        errors="coerce",
        utc=True
    )

    # Fahrenheit -> Celsius
    data["temperature_c"] = (
        pd.to_numeric(data["temperature_f"], errors="coerce") - 32
    ) * 5 / 9

    # IMPORTANT:
    # Vendor B reports vibration in g (acceleration), while Vendor A reports
    # mm/s (velocity). These are different physical quantities and cannot be
    # directly converted without additional frequency information.
    data["vibration_value"] = pd.to_numeric(
        data["vibration_g"],
        errors="coerce"
    )

    data["vibration_unit"] = "g"

    data["vendor"] = "Vendor_B"

    return data


def normalize_vendor_c(df, valid_machine_ids):
    """Normalize Vendor C inspection and maintenance data."""

    data = df.copy()

    # Remove exact duplicate rows
    data = data.drop_duplicates().copy()

    # Standard machine identifier
    data = data.rename(columns={
        "machine_ref": "machine_id",
        "recorded_at": "event_timestamp",
        "manual_confidence": "confidence_pct"
    })

    # Remove missing/invalid machine IDs
    data = data[
        data["machine_id"].isin(valid_machine_ids)
    ].copy()

    data["event_timestamp"] = pd.to_datetime(
        data["event_timestamp"],
        errors="coerce",
        utc=True
    )

    data["vendor"] = "Vendor_C"

    return data


def run_normalization():

    asset_master, vendor_a, vendor_b, vendor_c, maintenance = load_data()

    valid_machine_ids = set(
        asset_master["machine_id"].dropna()
    )

    normalized_a = normalize_vendor_a(
        vendor_a,
        valid_machine_ids
    )

    normalized_b = normalize_vendor_b(
        vendor_b,
        valid_machine_ids
    )

    normalized_c = normalize_vendor_c(
        vendor_c,
        valid_machine_ids
    )

    print("Vendor A normalized rows:", len(normalized_a))
    print("Vendor B normalized rows:", len(normalized_b))
    print("Vendor C normalized rows:", len(normalized_c))

    print("\nVendor B temperature example:")
    print(
        normalized_b[
            ["temperature_f", "temperature_c"]
        ].head()
    )


if __name__ == "__main__":
    run_normalization()