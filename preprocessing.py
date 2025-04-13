import pandas as pd

# Load raw dataset
df = pd.read_csv("./Data Set/energydata_complete.csv")

# Drop datetime (if not needed) and any irrelevant columns
# We'll keep key temperature, humidity, and weather-related columns
df.drop(columns=['date'], inplace=True)

# Optional: Check for duplicates or missing values
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Rename target column to match model training: 'Appliances'
# (Assuming it's already named correctly; otherwise rename it)
# df.rename(columns={"YourTargetCol": "Appliances"}, inplace=True)

# Optionally, reset index
df.reset_index(drop=True, inplace=True)

# Save preprocessed data
df.to_csv("./Processed Data Set/Preprocessed_Energy_Data.csv", index=False)
print("✅ Preprocessed data saved to Preprocessed_Energy_Data.csv")
