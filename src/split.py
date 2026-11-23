import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from config import OUT

df = pd.read_csv(OUT / "metadata_clip.csv")
embs = np.load(OUT / "clip_embs_kept.npy")
assert len(df) == len(embs), "Run clip_audit.py first so CSV and embeddings match."

#save each row's original index so we can match it to clip_embs_kept.npy later
df["emb_idx"] = np.arange(len(df))

#drop true junk (keep multi_item_collage_poster since those are clean 800x800 photos)
true_junk = df["is_junk"] & (df["clip_pred"] != "multi_item_collage_poster")
clean_df = df[~true_junk].copy()
print(f"Dropped {int(true_junk.sum())} junk images. Clean images remaining: {len(clean_df)}")

#separate furniture items from context images (room_scenes)
items = clean_df[clean_df["role"] == "item"].copy()
context = clean_df[clean_df["role"] == "context"].copy()

#stratified split: 70% train, 15% val, 15% test
train_df, temp_df = train_test_split(
    items, test_size=0.30, stratify=items["label"], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42
)

train_df["split"] = "train"
val_df["split"] = "val"
test_df["split"] = "test"
context["split"] = "context"

#combine and save
split_df = pd.concat([train_df, val_df, test_df, context], ignore_index=True)
split_df.to_csv(OUT / "metadata_split.csv", index=False)

print("\n--- Final Split Summary ---")
print(split_df["split"].value_counts())

print("\n--- Clean Items per Class across Splits ---")
print(pd.crosstab(
    split_df["label"],
    split_df["split"],
    margins=True
))