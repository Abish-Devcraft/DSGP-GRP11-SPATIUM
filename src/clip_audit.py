import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from config import DATA, OUT

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running CLIP on:", device)

model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id).to(device).eval()
proc = CLIPProcessor.from_pretrained(model_id)

df = pd.read_csv(OUT / "metadata_flagged.csv")
kept = df[df["keep"]].reset_index(drop=True)

furniture_labels = sorted(kept["label"].unique().tolist())
junk_labels = [
    "person",
    "human_portrait",
    "kitchen_sink_or_stove",
    "electronics_or_laptop",
    "bathroom_mirror_or_vanity",
    "blurry_fabric_or_wall_crop",
    "multi_item_collage_poster",
    "piles_of_clothes_or_clutter",
]
all_labels = furniture_labels + junk_labels

PROMPT_OVERRIDES = {
    "room_scenes": "a photo of a furnished room interior",
    "tv_stand": "a photo of a TV stand",
}
furniture_prompts = [
    PROMPT_OVERRIDES.get(lbl, f"a photo of a {lbl.replace('_', ' ')}") for lbl in furniture_labels
]
junk_prompts = [
    "a photo of a person or human",
    "a portrait of a person standing",
    "a photo of a kitchen sink, stove, or built-in kitchen counter",
    "a photo of a computer keyboard, laptop, or electronics",
    "a photo of a bathroom wall mirror or bathroom sink",
    "an unrecognizable blurry close-up of plain fabric, wood texture, or wall",
    "a promotional furniture catalog collage showing multiple separate items and text",
    "a close-up of folded clothes, boxes, or messy household clutter",
]
prompts = furniture_prompts + junk_prompts
assert len(prompts) == len(all_labels)


def to_tensor(out):
    return out if isinstance(out, torch.Tensor) else out.pooler_output


def load_rgb(rel_path):
    try:
        with Image.open(DATA / rel_path) as im:
            return im.convert("RGB")
    except Exception as e:
        raise RuntimeError(f"Could not open {rel_path}: {e}")


with torch.no_grad():
    t = proc(text=prompts, return_tensors="pt", padding=True).to(device)
    text_emb = to_tensor(model.get_text_features(**t))
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

paths = kept["rel"].tolist()
embs, preds, confs = [], [], []
batch_size = 64

for i in tqdm(range(0, len(paths), batch_size)):
    imgs = [load_rgb(p) for p in paths[i : i + batch_size]]
    x = proc(images=imgs, return_tensors="pt").to(device)
    with torch.no_grad():
        e = to_tensor(model.get_image_features(**x))
        e = e / e.norm(dim=-1, keepdim=True)
        probs = (100.0 * e @ text_emb.T).softmax(dim=-1)

    embs.append(e.cpu().numpy())
    top_conf, top_idx = probs.max(dim=-1)
    preds.extend([all_labels[j] for j in top_idx.tolist()])
    confs.extend(top_conf.tolist())

np.save(OUT / "clip_embs_kept.npy", np.vstack(embs))

kept["clip_pred"] = preds
kept["clip_conf"] = confs

#only flag junk/disagreements on furniture items (not room_scenes)
is_item = kept["role"] == "item"
kept["is_junk"] = is_item & kept["clip_pred"].isin(junk_labels)
kept["disagree"] = (
    is_item
    & (~kept["clip_pred"].isin(junk_labels))
    & (kept["clip_pred"] != kept["label"])
    & (kept["clip_conf"] > 0.45)
)

kept.to_csv(OUT / "metadata_clip.csv", index=False)

print("\n--- Junk images detected ---")
print(kept[kept["is_junk"]]["clip_pred"].value_counts())
print("Total junk flagged:", int(kept["is_junk"].sum()))
print("Total label disagreements (>0.45 conf):", int(kept["disagree"].sum()))