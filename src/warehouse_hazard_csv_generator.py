import numpy as np
import pandas as pd

np.random.seed(42)
n = 300
load = np.random.randint(100, 1001, n)
inspection = np.random.randint(1, 91, n)
sensors = np.random.randint(1, 6, n)
floor_age = np.random.randint(1, 31, n)
# True risk rule (unknown to the student): high risk when
# load > 500 OR inspection > 45.  Label noise (~20%) simulates
# real-world measurement error.
true_risk = ((load > 500) | (inspection > 45)).astype(float)
flip = np.random.random(n) < 0.20
high_risk = true_risk.copy()
high_risk[flip] = 1 - high_risk[flip]
high_risk = high_risk.astype(int)
df = pd.DataFrame({
    "load_kg": load, "inspection_days": inspection,
    "sensors": sensors, "floor_age_years": floor_age,
    "high_risk": high_risk,
})
df.to_csv("warehouse_hazard.csv", index=False)
print(df.head(10))
print(f"\nClass balance: {high_risk.sum()} high-risk, "
      f"{n - high_risk.sum()} low-risk out of {n}")