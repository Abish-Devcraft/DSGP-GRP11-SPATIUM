import time
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from solver import solve

df = pd.read_csv("data/layout_features.csv")
X = df.drop(columns=["label"])
y = df["label"]

# ---------- split 70 / 15 / 15 ----------
Xtr, Xtmp, ytr, ytmp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y)
Xva, Xte, yva, yte = train_test_split(
    Xtmp, ytmp, test_size=0.50, random_state=42, stratify=ytmp)
print("train/val/test:", len(Xtr), len(Xva), len(Xte))


def report(name, y_true, y_pred):
    print(f"{name:22s} acc={accuracy_score(y_true, y_pred):.3f} "
          f"prec={precision_score(y_true, y_pred):.3f} "
          f"rec={recall_score(y_true, y_pred):.3f} "
          f"f1={f1_score(y_true, y_pred):.3f}")


# ---------- models (defined once, used everywhere) ----------
models = {
    "logistic regression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000)),
    "random forest": RandomForestClassifier(
        n_estimators=300, max_depth=8, random_state=42, n_jobs=-1),
    "xgboost": XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        random_state=42, eval_metric="logloss"),
}

# ---------- 1. VALIDATION: compare models ----------
print("\nVALIDATION SET:")
report("always-fit", yva, [1] * len(yva))

best_t, best_f1 = None, 0
for t in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
    f = f1_score(yva, (Xva["furn_to_free"] <= t).astype(int))
    if f > best_f1:
        best_t, best_f1 = t, f
print("best rule threshold:", best_t)
report("rule-only (area)", yva, (Xva["furn_to_free"] <= best_t).astype(int))

for name, m in models.items():
    m.fit(Xtr, ytr)
    report(name, yva, m.predict(Xva))

# ---------- 2. 5-fold cross-validation on training set ----------
print("\n5-fold CV (F1, mean +/- std):")
for name, m in models.items():
    s = cross_val_score(m, Xtr, ytr, cv=5, scoring="f1")
    print(f"  {name:22s} {s.mean():.3f} +/- {s.std():.3f}")

# ---------- 3. FINAL TEST (run once, never tune after this) ----------
print("\nFINAL TEST SET:")
report("always-fit", yte, [1] * len(yte))
report("rule-only (area)", yte, (Xte["furn_to_free"] <= best_t).astype(int))
for name, m in models.items():
    m.fit(Xtr, ytr)
    report(name, yte, m.predict(Xte))

# ---------- 4. Speed: solver vs learned model ----------
N_RUNS = 200
t0 = time.time()
for _ in range(N_RUNS):
    solve(3.5, 3.0, 2.0, 0.9, 0.6, 0.4, 0.9, [(1.0, 1.0, 1.0, 0.8)])
solver_ms = (time.time() - t0) / N_RUNS * 1000

t0 = time.time()
for _ in range(N_RUNS):
    models["xgboost"].predict(Xte.iloc[:1])
model_ms = (time.time() - t0) / N_RUNS * 1000
print(f"\nsolver: {solver_ms:.2f} ms per call | xgboost: {model_ms:.2f} ms per call")

# ---------- 5. Save the model ----------
joblib.dump(models["xgboost"], "models/layout_v1.joblib")
print("saved models/layout_v1.joblib")