import pandas as pd
from pathlib import Path


# Project root folder
BASE_DIR = Path(__file__).resolve().parent.parent

def load_data():
    """
    Load all datasets required for the machine health scoring system.
    """

    asset_master = pd.read_csv(BASE_DIR / "asset_master.csv")

    vendor_a = pd.read_csv(
        BASE_DIR / "vendor_a_sensor_events.csv"
    )

    vendor_b = pd.read_csv(
        BASE_DIR / "vendor_b_alert_events.csv"
    )

    vendor_c = pd.read_csv(
        BASE_DIR / "vendor_c_inspection_maintenance.csv"
    )

    maintenance = pd.read_csv(
        BASE_DIR / "maintenance_records.csv"
    )

    return asset_master, vendor_a, vendor_b, vendor_c, maintenance


if __name__ == "__main__":

    datasets = load_data()

    dataset_names = [
        "Asset Master",
        "Vendor A",
        "Vendor B",
        "Vendor C",
        "Maintenance Records"
    ]

    for name, data in zip(dataset_names, datasets):
        print(f"\n{name}")
        print("Rows:", data.shape[0])
        print("Columns:", data.shape[1])