import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from features import make_features
from solver import solve

BEST_T = 0.2          # my best rule threshold from train.py

df = pd.read_csv("data/real_rooms.csv")
model = joblib.load("models/layout_v1.joblib")
X = make_features(df)
human = df["human_label"]

pred_ml = model.predict(X[model.feature_names_in_])
pred_rule = (X["furn_to_free"] <= BEST_T).astype(int)

pred_solver = []
for r in df.itertuples():
    existing = []
    for i in (1, 2, 3):
        w = getattr(r, f"e{i}_w")
        if w > 0:
            existing.append((getattr(r, f"e{i}_x"), getattr(r, f"e{i}_y"),
                             w, getattr(r, f"e{i}_d")))
    fits, _ = solve(r.length_m, r.width_m, r.f_w, r.f_d, r.clearance,
                    r.door_off, r.door_w, existing)
    used = (r.existing_area + r.f_w * r.f_d) / (r.length_m * r.width_m)
    pred_solver.append(1 if (fits and used <= 0.55) else 0)

for name, p in [("ML (xgboost)", pred_ml),
                ("rule-only", pred_rule),
                ("solver", pred_solver)]:
    agree = (human.to_numpy() == pd.Series(p).to_numpy()).sum()
    print(f"{name:14s} agrees with human in {agree} of {len(df)} rooms "
          f"(accuracy {accuracy_score(human, p):.2f})")
    print(confusion_matrix(human, p, labels=[0, 1]))

df["ml_pred"] = pred_ml
df["solver_pred"] = pred_solver
print(df[["length_m", "width_m", "f_w", "f_d", "human_label", "ml_pred", "solver_pred"]])