import pandas as pd
import numpy as np
import os

# Ensure the folder exists
os.makedirs("data", exist_ok=True)

# Generate 50 dummy rows
data = {
    "id": range(1000, 1050),  # IDs from 1000 to 1049
    "lat": np.random.uniform(12.0, 30.0, 50),  # Random latitudes (approx India/Asia)
    "long": np.random.uniform(77.0, 80.0, 50)  # Random longitudes
}

df = pd.DataFrame(data)

# Save to the specific path your code expects
output_path = "data/test_rooftop_data.csv"
df.to_csv(output_path, index=False)

print(f"✅ Created dummy test file at: {output_path}")
print(df.head())