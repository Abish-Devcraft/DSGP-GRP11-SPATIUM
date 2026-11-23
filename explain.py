import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/layout_features.csv")
X = df.drop(columns=["label"])
y = df["label"]

# same split as train.py, so i explain only unseen test rows
_, Xtmp, _, ytmp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
_, Xte, _, yte = train_test_split(Xtmp, ytmp, test_size=0.50, random_state=42, stratify=ytmp)

model = joblib.load("models/layout_v1.joblib")
Xte = Xte[model.feature_names_in_]          # same column order as training
explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(Xte)

# figure 1: dot summary
shap.summary_plot(sv, Xte, show=False)
plt.tight_layout()
plt.savefig("plots/shap_summary.png")
plt.clf()

# figure 2: bar chart of average importance
shap.summary_plot(sv, Xte, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("plots/shap_bar.png")
plt.clf()

# "why not?" messages
GROUPS = {
    "not enough free floor left": ["free_floor_ratio", "free_floor_area",
                                   "existing_area", "n_existing",
                                   "largest_existing_area"],
    "item is too big for this room": ["furn_to_free", "furn_to_room",
                                      "footprint", "longest_side",
                                      "room_area", "perimeter"],
    "no room for the item plus walking clearance": ["best_slack", "clearance"],
    "existing furniture blocks the door path": ["door_zone_overlap",
                                                "min_dist_to_door", "door_off"],
    "item is too wide for the door": ["door_margin"],
}


def why_not(i):
    pred = int(model.predict(Xte.iloc[[i]])[0])
    if pred == 1:
        return "fit", []
    contrib = pd.Series(sv[i], index=Xte.columns)
    scores = {msg: contrib[[f for f in feats if f in contrib.index]].sum()
              for msg, feats in GROUPS.items()}
    ranked = sorted(scores.items(), key=lambda kv: kv[1])
    reasons = [msg for msg, s in ranked[:2] if s < 0]   # only real "no_fit" pushes
    return "no_fit", reasons

shown = 0
for i in range(len(Xte)):
    verdict, reasons = why_not(i)
    if verdict == "no_fit":
        print(f"row {i}: {verdict} -> {reasons}")
        shown += 1
    if shown == 5:
        break