from pathlib import Path

from feature_engineering import build_machine_features
from scoring import calculate_scores


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"


def run_pipeline():

    print("Starting Machine Health Scoring Pipeline...")

    # Build machine-level features
    features = build_machine_features()

    # Calculate scores
    scored_data = calculate_scores(features)

    # Create output folder if it does not exist
    OUTPUT_DIR.mkdir(exist_ok=True)

    output_file = OUTPUT_DIR / "machine_health_scores.csv"

    # Select final business-facing columns
    final_output = scored_data[
        [
            "machine_id",
            "plant_id",
            "line_id",
            "machine_type",
            "criticality",
            "health_score",
            "confidence_score",
            "readiness_score",
            "status"
        ]
    ].copy()

    # Save CSV
    final_output.to_csv(
        output_file,
        index=False
    )

    print("\nPipeline completed successfully.")
    print(f"Output saved to: {output_file}")

    print("\nFinal Machine Health Output:")
    print(final_output.to_string(index=False))


if __name__ == "__main__":
    run_pipeline()