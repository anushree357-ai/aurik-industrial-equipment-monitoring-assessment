from data_ingestion import load_data


def validate_dataset(name, df, machine_id_column=None, valid_machine_ids=None):
    """
    Perform basic data quality checks on a dataset.
    """

    print(f"\n{'=' * 50}")
    print(f"Validating: {name}")
    print(f"{'=' * 50}")

    # 1. Dataset size
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    # 2. Missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    print("\nMissing Values:")
    if missing.empty:
        print("No missing values found.")
    else:
        print(missing)

    # 3. Duplicate rows
    duplicate_count = df.duplicated().sum()
    print(f"\nDuplicate Rows: {duplicate_count}")

    # 4. Machine ID validation
    if machine_id_column and valid_machine_ids is not None:
        invalid_ids = df.loc[
            ~df[machine_id_column].isin(valid_machine_ids),
            machine_id_column
        ].dropna().unique()

        print("\nInvalid Machine IDs:")

        if len(invalid_ids) == 0:
            print("No invalid machine IDs found.")
        else:
            print(invalid_ids)


def run_validation():

    asset_master, vendor_a, vendor_b, vendor_c, maintenance = load_data()

    valid_machine_ids = set(asset_master["machine_id"].dropna())

    validate_dataset(
        "Asset Master",
        asset_master
    )

    validate_dataset(
        "Vendor A",
        vendor_a,
        "machine_id",
        valid_machine_ids
    )

    validate_dataset(
        "Vendor B",
        vendor_b,
        "assetCode",
        valid_machine_ids
    )

    validate_dataset(
        "Vendor C",
        vendor_c,
        "machine_ref",
        valid_machine_ids
    )

    validate_dataset(
        "Maintenance Records",
        maintenance,
        "machine_id",
        valid_machine_ids
    )


if __name__ == "__main__":
    run_validation()