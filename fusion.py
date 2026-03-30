# =========================================
# SOIL MOISTURE FUSION PROJECT (PERSON B)
# =========================================

# 🔹 1. Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor

# 🔹 2. Load data
smap = np.load("smap.npy")
sentinel = np.load("sentinel.npy")
ndvi = np.load("ndvi.npy")

print("Shapes:", smap.shape, sentinel.shape, ndvi.shape)


# =========================================
# 🔹 3. VISUALIZE INPUT DATA
# =========================================

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(smap)
plt.title("SMAP")
plt.colorbar()

plt.subplot(1,3,2)
plt.imshow(sentinel)
plt.title("Sentinel")
plt.colorbar()

plt.subplot(1,3,3)
plt.imshow(ndvi)
plt.title("NDVI")
plt.colorbar()

plt.tight_layout()
plt.show()


# =========================================
# 🔹 4. PREPARE ML DATA (FEATURE ENGINEERING)
# =========================================

X = np.stack([
    smap,
    sentinel,
    ndvi,
    smap - sentinel,
    smap * ndvi,
    sentinel * ndvi
], axis=-1)

X = X.reshape(-1, 6)

# Target (fusion of SMAP + Sentinel)
y = (0.4 * smap + 0.6 * sentinel).reshape(-1)


# =========================================
# 🔹 5. TRAIN MODEL (FUSION)
# =========================================

model = GradientBoostingRegressor()
model.fit(X, y)

pred = model.predict(X)
fused = pred.reshape(smap.shape)


# =========================================
# 🔹 6. SHOW FUSED OUTPUT
# =========================================

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(smap)
plt.title("SMAP")

plt.subplot(1,3,2)
plt.imshow(sentinel)
plt.title("Sentinel")

plt.subplot(1,3,3)
plt.imshow(fused)
plt.title("FUSED")

plt.tight_layout()
plt.show()


# =========================================
# 🔹 7. DIFFERENCE MAP (PROOF)
# =========================================

diff = fused - smap

plt.imshow(diff)
plt.title("Difference (Fused - SMAP)")
plt.colorbar()
plt.show()


# =========================================
# 🔹 8. NUMERIC PROOF
# =========================================

mean_diff = np.mean(np.abs(fused - smap))
print("Mean Absolute Difference:", mean_diff)


# =========================================
# 🔹 9. HISTOGRAM COMPARISON
# =========================================

plt.hist(smap.flatten(), bins=20, alpha=0.5, label="SMAP")
plt.hist(fused.flatten(), bins=20, alpha=0.5, label="Fused")

plt.legend()
plt.title("Histogram Comparison")
plt.show()


# =========================================
# 🔹 10. SAVE OUTPUT
# =========================================

np.save("fused.npy", fused)

print("Fused data saved as fused.npy")