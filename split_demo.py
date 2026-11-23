import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/layout_features.csv")
X = df.drop(columns=["label"])
y = df["label"]

Xtr, Xtmp, ytr, ytmp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y)
Xva, Xte, yva, yte = train_test_split(
    Xtmp, ytmp, test_size=0.50, random_state=42, stratify=ytmp)

print(f"{'pile':12s} {'rows':>6s} {'% of data':>10s} {'fit share':>10s}")
for name, part in [("train", ytr), ("validation", yva), ("test", yte)]:
    print(f"{name:12s} {len(part):6d} {len(part)/len(y):10.0%} {part.mean():10.3f}")