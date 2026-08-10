import pandas as pd

from feature_engineering import build_machine_features


def clamp(score):
    """Keep a score between 0 and 100."""
    return max(0, min(100, score))


def calculate_scores(features):

    data = features.copy()

    # ==================================================
    # 1. HEALTH SCORE
    # ==================================================

    health_scores = []

    for _, row in data.iterrows():

        score = 100

        # Average temperature condition
        if (
            pd.notna(row["avg_temperature_c"])
            and pd.notna(row["rated_max_temp_c"])
        ):

            avg_temp_ratio = (
                row["avg_temperature_c"]
                / row["rated_max_temp_c"]
            )

            if avg_temp_ratio > 1.05:
                score -= 25
            elif avg_temp_ratio > 1.00:
                score -= 15
            elif avg_temp_ratio > 0.90:
                score -= 8

        # Average vibration condition
        if (
            pd.notna(row["avg_vibration_mm_s"])
            and pd.notna(row["rated_max_vibration_mm_s"])
        ):

            avg_vibration_ratio = (
                row["avg_vibration_mm_s"]
                / row["rated_max_vibration_mm_s"]
            )

            if avg_vibration_ratio > 1.05:
                score -= 30
            elif avg_vibration_ratio > 1.00:
                score -= 20
            elif avg_vibration_ratio > 0.85:
                score -= 10

        # Temperature spike
        if (
            pd.notna(row["max_temperature_c"])
            and pd.notna(row["rated_max_temp_c"])
        ):

            max_temp_ratio = (
                row["max_temperature_c"]
                / row["rated_max_temp_c"]
            )

            if max_temp_ratio > 1.30:
                score -= 10
            elif max_temp_ratio > 1.10:
                score -= 5

        # Vibration spike
        if (
            pd.notna(row["max_vibration_mm_s"])
            and pd.notna(row["rated_max_vibration_mm_s"])
        ):

            max_vibration_ratio = (
                row["max_vibration_mm_s"]
                / row["rated_max_vibration_mm_s"]
            )

            if max_vibration_ratio > 1.30:
                score -= 10
            elif max_vibration_ratio > 1.10:
                score -= 5

        # Maintenance/service condition
        if (
            pd.notna(row["max_days_since_service"])
            and pd.notna(row["service_interval_days"])
        ):

            service_ratio = (
                row["max_days_since_service"]
                / row["service_interval_days"]
            )

            if service_ratio > 1.5:
                score -= 15
            elif service_ratio > 1.0:
                score -= 8

        # Historical downtime
        if pd.notna(row["total_downtime_minutes"]):

            if row["total_downtime_minutes"] > 1500:
                score -= 10
            elif row["total_downtime_minutes"] > 750:
                score -= 6
            elif row["total_downtime_minutes"] > 300:
                score -= 3

        health_scores.append(clamp(score))

    data["health_score"] = health_scores

    # ==================================================
    # 2. CONFIDENCE SCORE
    # ==================================================

    confidence_columns = [
        "avg_sensor_confidence",
        "avg_vendor_b_confidence",
        "avg_manual_confidence"
    ]

    data["confidence_score"] = (
        data[confidence_columns]
        .mean(axis=1, skipna=True)
        .clip(0, 100)
        .round(2)
    )

    # ==================================================
    # 3. READINESS SCORE
    # ==================================================

    readiness_scores = []

    for _, row in data.iterrows():

        score = 0

        # Core sensor availability
        if pd.notna(row["avg_temperature_c"]):
            score += 15

        if pd.notna(row["avg_vibration_mm_s"]):
            score += 15

        if pd.notna(row["avg_power_kw"]):
            score += 10

        # Maintenance history
        if (
            pd.notna(row["maintenance_count"])
            and row["maintenance_count"] > 0
        ):
            score += 15

        # Service history
        if pd.notna(row["max_days_since_service"]):
            score += 10

        # Multi-vendor data coverage
        vendor_coverage = 0

        if (
            pd.notna(row["vendor_a_event_count"])
            and row["vendor_a_event_count"] > 0
        ):
            vendor_coverage += 1

        if (
            pd.notna(row["vendor_b_alert_count"])
            and row["vendor_b_alert_count"] > 0
        ):
            vendor_coverage += 1

        if (
            pd.notna(row["vendor_c_record_count"])
            and row["vendor_c_record_count"] > 0
        ):
            vendor_coverage += 1

        score += vendor_coverage * 5

        # Confidence quality
        confidence_values = [
            row["avg_sensor_confidence"],
            row["avg_vendor_b_confidence"],
            row["avg_manual_confidence"]
        ]

        valid_confidence = [
            value
            for value in confidence_values
            if pd.notna(value)
        ]

        if valid_confidence:

            avg_confidence = (
                sum(valid_confidence)
                / len(valid_confidence)
            )

            if avg_confidence >= 80:
                score += 20
            elif avg_confidence >= 60:
                score += 15
            elif avg_confidence >= 40:
                score += 10
            else:
                score += 5

        readiness_scores.append(clamp(score))

    data["readiness_score"] = readiness_scores

    # ==================================================
    # 4. MACHINE STATUS
    # ==================================================

    def assign_status(score):

        if score >= 80:
            return "Healthy"

        elif score >= 60:
            return "Monitor"

        elif score >= 40:
            return "Attention Required"

        else:
            return "Critical"

    data["status"] = (
        data["health_score"]
        .apply(assign_status)
    )

    return data


# ======================================================
# RUN SCORING
# ======================================================

if __name__ == "__main__":

    features = build_machine_features()

    scored_data = calculate_scores(features)

    print(
        "\nMachine Health Scoring completed successfully."
    )

    print(
        scored_data[
            [
                "machine_id",
                "health_score",
                "confidence_score",
                "readiness_score",
                "status"
            ]
        ].to_string(index=False)
    )