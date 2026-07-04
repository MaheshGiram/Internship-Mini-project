import pandas as pd

# Load health dataset
health_df = pd.read_csv(
    "datasets/alzheimers_disease_data.csv"
)

# Load speech dataset
speech_df = pd.read_csv(
    "speechdataset/speech_data.csv"
)

# Keep only first 1010 rows from health dataset
health_df = health_df.iloc[:1010]

# Reset indexes
health_df = health_df.reset_index(drop=True)
speech_df = speech_df.reset_index(drop=True)

# Combine datasets
fusion_df = pd.concat(
    [health_df, speech_df],
    axis=1
)

# Save fusion dataset
fusion_df.to_csv(
    "datasets/fusion_dataset.csv",
    index=False
)

print("Fusion Dataset Created Successfully!")
print("Rows:", fusion_df.shape[0])
print("Columns:", fusion_df.shape[1])