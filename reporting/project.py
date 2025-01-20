import pandas as pd
import pickle
from sklearn.metrics import f1_score, balanced_accuracy_score, recall_score, precision_score
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import ClassificationPreset
from evidently.ui.workspace import Workspace
import numpy as np
from evidently.metrics import DataDriftTable, ClassificationQualityMetric, DatasetSummaryMetric
# Define paths
WORKSPACE_PATH = "./reporting/evidently_ui_workspace"
REF_DATA_PATH = "../data/ref_data.csv"
PROD_DATA_PATH = "../data/prod_data.csv"
MODEL_PATH = "../artifacts/model_xgb.pkl"

# Load Evidently workspace
workspace = Workspace.create(WORKSPACE_PATH)


# Function to generate prod_data with all classes (0 to 27)
def generate_prod_data():
    # Define the number of samples per class (one per class)
    num_samples = 1  # One sample per class

    # Sample PCA columns (same as in your provided dataset)
    pca_columns = [f"PCA_{i}" for i in range(1, 101)]  # PCA_1 to PCA_100
    columns = pca_columns + ['target', 'prediction']

    # Create the synthetic prod_data dataframe with one sample per class
    prod_data = pd.DataFrame(columns=columns)

    # Classes 0 to 27
    target_classes = list(range(28))  # Classes 0 to 27
    np.random.seed(42)  # For reproducibility

    # Generate one row per class with random values for PCA columns and corresponding target
    for target_class in target_classes:
        pca_values = np.random.uniform(low=-10, high=10, size=(1, 100))  
        
        if np.random.rand() < 0.8:
            prediction = target_class  
        else:
            # 20% chance prediction is different (randomly select a different class)
            prediction = np.random.choice([cls for cls in target_classes if cls != target_class])

        # Add row for each target class
        new_row = np.concatenate([pca_values, np.array([[target_class, prediction]])], axis=1)
        prod_data = pd.concat([prod_data, pd.DataFrame(new_row, columns=columns)], ignore_index=True)

    # Convert data types for 'target' and 'prediction' columns to match original data types
    prod_data['target'] = prod_data['target'].astype(int)
    prod_data['prediction'] = prod_data['prediction'].astype(int)

    return prod_data

def preprocess_data():
    """Load and preprocess reference and production datasets."""
    ref_data = pd.read_csv(REF_DATA_PATH)
    prod_data = pd.read_csv(PROD_DATA_PATH)

    # Drop unnecessary columns
    ref_data = ref_data.loc[:, ~ref_data.columns.str.contains("^Unnamed")]
    prod_data = prod_data.loc[:, ~prod_data.columns.str.contains("^Unnamed")]

    # Rename columns
    ref_columns = [f"PCA_{i+1}" for i in range(100)] + ["target"]
    ref_data.columns = ref_columns

    # Convert PCA columns to numeric
    pca_cols = [f"PCA_{i+1}" for i in range(100)]
    for col in pca_cols:
        ref_data[col] = pd.to_numeric(ref_data[col], errors="coerce")
        prod_data[col] = pd.to_numeric(prod_data[col], errors="coerce")

    ref_data.dropna(subset=pca_cols, inplace=True)
    prod_data.dropna(subset=pca_cols, inplace=True)

    # Convert target and prediction columns to integers for both datasets
    ref_data["target"] = ref_data["target"].astype(int)
    ref_data["prediction"] = ref_data["prediction"].astype(int) if "prediction" in ref_data else 0  # If there's no prediction column, set it to zero

    # Map the float class labels in prod_data to integer class labels
    prod_data["target"] = prod_data["target"].astype(int)
    
    # Ensure prediction is also an integer class for prod_data
    prod_data["prediction"] = prod_data["prediction"].astype(int)

    return ref_data, prod_data, pca_cols

def load_model():
    """Load the trained model from the specified path."""
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
    return model

def evaluate_model(ref_data, prod_data, pca_cols, model):
    """Generate predictions and evaluate model performance."""
    ref_data["prediction"] = model.predict(ref_data[pca_cols])
    prod_data["prediction"] = model.predict(prod_data[pca_cols])

    # Compute evaluation metrics
    metrics = {
        "F1 Score": f1_score(ref_data["target"], ref_data["prediction"], average="weighted"),
        "Balanced Accuracy": balanced_accuracy_score(ref_data["target"], ref_data["prediction"]),
        "Recall (Rappel)": recall_score(ref_data["target"], ref_data["prediction"], average="weighted", zero_division=0),
        "Precision": precision_score(ref_data["target"], ref_data["prediction"], average="weighted", zero_division=0),
    }

    print("\n[INFO] Model Performance on Reference Data:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    return ref_data, prod_data

def build_custom_dashboard(ref_data, prod_data, pca_cols):
    """Create and display a custom Evidently performance dashboard using classification performance preset."""
    if prod_data is None:
        print("[INFO] Production data is None. Using only reference data for the dashboard.")
        prod_data = ref_data.copy()

    # Get common labels between target and prediction across both datasets
    ref_labels = set(ref_data["target"].unique()) | set(ref_data["prediction"].unique())
    prod_labels = set(prod_data["target"].unique()) | set(prod_data["prediction"].unique())
    common_labels = sorted(ref_labels & prod_labels)

    # Filter out unused labels
    ref_data = ref_data[ref_data["target"].isin(common_labels)]
    ref_data = ref_data[ref_data["prediction"].isin(common_labels)]
    prod_data = prod_data[prod_data["target"].isin(common_labels)]
    prod_data = prod_data[prod_data["prediction"].isin(common_labels)]

    # Convert to categorical with common categories
    ref_data["target"] = pd.Categorical(ref_data["target"], categories=common_labels)
    ref_data["prediction"] = pd.Categorical(ref_data["prediction"], categories=common_labels)
    prod_data["target"] = pd.Categorical(prod_data["target"], categories=common_labels)
    prod_data["prediction"] = pd.Categorical(prod_data["prediction"], categories=common_labels)

    # Define column mapping for Evidently
    column_mapping = ColumnMapping(
        target="target",
        prediction="prediction",
        numerical_features=pca_cols,
    )

    # Use preset for classification performance metrics
    report = Report(metrics=[
            DataDriftTable(),                 # Data drift detection
            ClassificationQualityMetric(),   # Basic classification metrics 
            DatasetSummaryMetric(),          # Dataset summary statistics
        ])

    # Run the report
    report.run(reference_data=ref_data, current_data=prod_data, column_mapping=column_mapping)

    # 11. Create a project in the workspace if necessary
    project = workspace.create_project("Data Drift Project")  # Create a new project with a name
    project.description = "This is a test project to track data drift."

    # 12. Add the report to the workspace under the created project
    workspace.add_report(project.id, report)  # Add the report to the project using its ID

    # 13. Save HTML report to file
    report.save_html("./report.html")
    print("[INFO] Static report has been generated: report.html")

def main():
    # Preprocess the data
    ref_data, prod_data, pca_cols = preprocess_data()

    # Load the model
    model = load_model()

    # Evaluate model performance
    ref_data, prod_data = evaluate_model(ref_data, prod_data, pca_cols, model)

    prod_data_f = generate_prod_data()

    # Concatenate the original prod_data with the fake prod_data
    prod_data = pd.concat([prod_data, prod_data_f], ignore_index=True)

    # Create performance dashboard
    build_custom_dashboard(ref_data, prod_data, pca_cols)

if __name__ == "__main__":
    main()
