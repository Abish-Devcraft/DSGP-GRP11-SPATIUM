import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/layout_raw.csv")

#datacleaning
print("Rows, columns:", df.shape)
print("Missing values:\n", df.isna().sum())
print("Duplicate rows:", df.duplicated().sum())

df = df.drop_duplicates()
df = df[(df.length_m > 0) & (df.width_m > 0)]   # no impossible rooms
df.to_csv("data/layout_clean.csv", index=False)

# eda
print("\nSummary numbers:\n", df.describe())

print("\nFit vs no_fit share:")
print(df["label"].value_counts(normalize=True))

# plot 1: class balance
df["label"].value_counts().plot(kind="bar")
plt.title("Fit (1) vs No fit (0)")
plt.savefig("plots/class_balance.png")
plt.clf()

# plot 2: distributions of every column
df.hist(figsize=(12, 8))
plt.tight_layout()
plt.savefig("plots/histograms.png")
plt.clf()

# plot 3: which columns relate to the answer
print("\nCorrelation with label:")
print(df.corr(numeric_only=True)["label"].sort_values())

# plot 4: fit rate by clearance profile
print("\nFit rate by clearance:")
print(df.groupby("clearance")["label"].mean())
df.groupby("clearance")["label"].mean().plot(kind="bar")
plt.title("Fit rate by clearance profile")
plt.savefig("plots/fit_by_clearance.png")
plt.clf()

# plot 5: fit rate by number of existing items
print("\nFit rate by number of existing items:")
print(df.groupby("n_existing")["label"].mean())
df.groupby("n_existing")["label"].mean().plot(kind="bar")
plt.title("Fit rate by number of existing items")
plt.savefig("plots/fit_by_existing.png")
plt.clf()