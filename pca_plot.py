import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

df = pd.read_csv("data/layout_features.csv")
X = df.drop(columns=["label"])
y = df["label"]
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

sc = StandardScaler().fit(Xtr)               # fit on train only
Xtr_s, Xva_s = sc.transform(Xtr), sc.transform(Xva)

# 1. 2D picture
pca2 = PCA(n_components=2).fit(Xtr_s)
Z = pca2.transform(Xtr_s)
print("variance kept by 2 components:", pca2.explained_variance_ratio_.sum().round(3))
plt.scatter(Z[:, 0], Z[:, 1], c=ytr, s=3, alpha=0.5, cmap="coolwarm")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of layout features (red = fit)")
plt.savefig("plots/pca_2d.png")
plt.clf()

# 2. how many components are needed
pca_all = PCA().fit(Xtr_s)
cum = np.cumsum(pca_all.explained_variance_ratio_)
plt.plot(range(1, len(cum) + 1), cum, marker="o")
plt.axhline(0.95, linestyle="--")
plt.xlabel("number of components")
plt.ylabel("cumulative variance")
plt.savefig("plots/pca_variance.png")
plt.clf()
k95 = int(np.argmax(cum >= 0.95)) + 1
print("components needed for 95% variance:", k95, "of", X.shape[1])

# 3. experiment: with vs without PCA
base = LogisticRegression(max_iter=1000).fit(Xtr_s, ytr)
f_base = f1_score(yva, base.predict(Xva_s))
pca_k = PCA(n_components=k95).fit(Xtr_s)
lr_pca = LogisticRegression(max_iter=1000).fit(pca_k.transform(Xtr_s), ytr)
f_pca = f1_score(yva, lr_pca.predict(pca_k.transform(Xva_s)))
print(f"LR without PCA f1={f_base:.3f} | with PCA ({k95} comps) f1={f_pca:.3f}")