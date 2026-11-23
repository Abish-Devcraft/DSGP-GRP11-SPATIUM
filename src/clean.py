import pandas as pd
from config import OUT, final_label, CONTEXT_ONLY

df = pd.read_csv(OUT / "inventory.csv")
df["reason"] = ""

#standardize labels using config.py
df["label_raw"] = df["label"]
df["label"] = df["label"].map(final_label)

#separate context images (room_scenes) from recommendable furniture items
df["role"] = "item"
df.loc[df["label"].isin(CONTEXT_ONLY), "role"] = "context"


def flag(mask, why):
    """Mark rows with a reason, without overwriting an existing reason."""
    df.loc[mask & (df["reason"] == ""), "reason"] = why


#unreadable or empty files
flag(~df["ok"] | (df["kb"] == 0), "unreadable")
good = df["ok"] & (df["kb"] > 0)

#label conflicts (same picture sits in two different classes across datasets)
lab_sha = df[good].groupby("sha")["label"].nunique()
lab_ph = df[good].groupby("phash")["label"].nunique()
conflict = df["sha"].isin(lab_sha[lab_sha > 1].index) | df["phash"].isin(lab_ph[lab_ph > 1].index)
df[good & conflict].sort_values("phash").to_csv(OUT / "conflicting_labels.csv", index=False)
flag(good & conflict, "label_conflict")

#exact copies, then near-duplicates (keep largest file size first)
df = df.sort_values("kb", ascending=False)
good = df["ok"] & (df["kb"] > 0)

still_ok = good & (df["reason"] == "")
flag(still_ok & df[still_ok.values].duplicated("sha").reindex(df.index, fill_value=False), "exact_duplicate")

still_ok = good & (df["reason"] == "")
flag(still_ok & df[still_ok.values].duplicated("phash").reindex(df.index, fill_value=False), "near_duplicate")

#size flags (informational only)
df["short_side"] = df[["w", "h"]].min(axis=1)
df["small_kb"] = df["kb"] < 10
df["tiny_px"] = df["short_side"] < 64

df["keep"] = df["reason"] == ""
df.to_csv(OUT / "metadata_flagged.csv", index=False)

print("--- Flagging summary ---")
print(df["reason"].replace("", "kept").value_counts())
print("\n--- Kept images per final class ---")
print(df[df["keep"]]["label"].value_counts())