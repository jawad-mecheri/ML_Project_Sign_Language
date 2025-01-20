
import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset

# for locel  use :
#REF_DATA_PATH = "./data/ref_data.csv"
#PROD_DATA_PATH = "./data/prod_data.csv"
REF_DATA_PATH = "../data/ref_data.csv"
PROD_DATA_PATH = "../data/prod_data.csv"


def debug_label_types(df: pd.DataFrame, name: str):
    """
    Helper function to scan 'target' and 'prediction' columns for any non-string values.
    Prints them if found (for debugging issues like "TypeError: '<' not supported").
    """
    unique_vals = set(df['target'].unique())
    if 'prediction' in df.columns:
        unique_vals.update(df['prediction'].unique())

    for val in unique_vals:
        if not isinstance(val, str):
            print(f"[DEBUG] {name}: Found non-string value => {repr(val)} (type={type(val)})")


def build_static_report():
    """
    Generate a static HTML file (report.html) comparing reference vs. production data,
    then launch the Evidently UI on port 8082 (blocks until stopped).
    """
    # 1. Load data
    ref_data = pd.read_csv(REF_DATA_PATH)
    prod_data = pd.read_csv(PROD_DATA_PATH)

    # 2. Drop any "Unnamed" columns (common when CSVs have trailing commas)
    ref_data = ref_data.loc[:, ~ref_data.columns.str.contains("^Unnamed")]
    prod_data = prod_data.loc[:, ~prod_data.columns.str.contains("^Unnamed")]

    print("Reference data shape BEFORE rename:", ref_data.shape)
    print("Production data shape:", prod_data.shape)

    # 3. Rename reference columns from 0..99,label → PCA_1..PCA_100,target
    ref_columns = [f"PCA_{i+1}" for i in range(100)] + ["target"]
    ref_data.columns = ref_columns
    print("Reference data shape AFTER rename:", ref_data.shape)
    print("Reference columns:", ref_data.columns.tolist())
    print("Production columns:", prod_data.columns.tolist())

    # 4. Ensure reference has a "prediction" column, even if dummy
    if "prediction" not in ref_data.columns:
        ref_data["prediction"] = "dummy_pred"

    # If production does not have "prediction" column, add one too
    if "prediction" not in prod_data.columns:
        prod_data["prediction"] = "dummy_pred"

    # 5. Convert PCA columns to numeric (coerce => NaN), then drop rows that fail
    pca_cols = [f"PCA_{i+1}" for i in range(100)]
    for col in pca_cols:
        ref_data[col] = pd.to_numeric(ref_data[col], errors="coerce")
        prod_data[col] = pd.to_numeric(prod_data[col], errors="coerce")

    ref_data.dropna(subset=pca_cols, inplace=True)
    prod_data.dropna(subset=pca_cols, inplace=True)

    # 6. Force target & prediction to string
    ref_data["target"] = ref_data["target"].apply(str)
    ref_data["prediction"] = ref_data["prediction"].apply(str)
    prod_data["target"] = prod_data["target"].apply(str)
    prod_data["prediction"] = prod_data["prediction"].apply(str)

    # 7. Optional debug checks
    print("\n--- Debug: Checking for non-string values in reference ---")
    debug_label_types(ref_data, name="Reference")
    print("\n--- Debug: Checking for non-string values in production ---")
    debug_label_types(prod_data, name="Production")
    print()

    # 8. Column mapping
    column_mapping = ColumnMapping(
        target="target",
        prediction="prediction",
        numerical_features=pca_cols,
        categorical_features=None,
        datetime_features=None
    )

    # 9. Create the Evidently report
    report = Report(
        metrics=[
            DataDriftPreset(),
            # ClassificationPreset()  # re-enable if you want classification metrics
        ]
    )

    # 10. Run the report
    report.run(
        reference_data=ref_data,
        current_data=prod_data,
        column_mapping=column_mapping,
    )

    # 11. Save HTML
    report.save_html("report.html")
    print("[INFO] Static report has been generated: report.html")



def main():

    # Call the main function to generate the report
    build_static_report()
    
if __name__ == "__main__":
    main()
