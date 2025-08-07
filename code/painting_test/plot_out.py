import os
import random
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# -------------------------------
# Config
# -------------------------------

step = "step35000"
base_test_path = Path("/u/wangyd/mpib/chm-artistic-social-determinism/Data/test") / step / "generated"
real_path = Path("/u/wangyd/mpib/chm-artistic-social-determinism/Data/dreambooth_dataset/train")
shared_output_path = Path("/u/wangyd/mpib/chm-artistic-social-determinism/Art_social_determinism/imgs") / f"painting_{step}"
shared_output_path.mkdir(parents=True, exist_ok=True)

# Folder and label mappings
keys_order = ["real", "c", "t1", "t2", "t3", "t4", "t5", "t6", "t-all"]
folders = {
    "real": None,
    "c": "C",
    "t1": "T1",
    "t2": "T2",
    "t3": "T3",
    "t4": "T4",
    "t5": "T5",
    "t6": "T6",
    "t-all": "T_all"
}
labels = {
    "real": "Real Painting",
    "c": "Content Only",
    "t1": "Content + Year",
    "t2": "Content + Style",
    "t3": "Content + Gender",
    "t4": "Content + Location",
    "t5": "Content + Connection",
    "t6": "Content + Name",
    "t-all": "Content + All"
}

# -------------------------------
# Get valid IDs from "C"
# -------------------------------

c_folder = base_test_path / "C"
available_ids = [int(f.replace("_c.png", "")) for f in os.listdir(c_folder) if f.endswith("_c.png") and f.replace("_c.png", "").isdigit()]

# You can random.sample(available_ids, 2) to just pick 2
random_ids = available_ids

# -------------------------------
# Plotting per ID (2 rows × 5 cols)
# -------------------------------

for id_val in random_ids:
    fig, axs = plt.subplots(2, 5, figsize=(12, 5))  # smaller figure size
    axs = axs.flatten()

    for idx, key in enumerate(keys_order + ["blank"]):  # Add one empty tile for symmetry
        ax = axs[idx]

        if key == "blank":
            ax.axis("off")
            continue

        # Determine file path
        if key == "real":
            img_path = real_path / f"{id_val}.jpg"
        else:
            folder_name = folders[key]
            img_path = base_test_path / folder_name / f"{id_val}_{key}.png"

        if img_path.exists():
            try:
                img = Image.open(img_path)
                ax.imshow(img)
            except Exception:
                ax.text(0.5, 0.5, "Corrupt", ha="center", va="center", fontsize=10)
        else:
            ax.text(0.5, 0.5, "Missing", ha="center", va="center", fontsize=10)

        ax.axis("off")
        ax.set_title(labels.get(key, ""), fontsize=10)

    plt.tight_layout()

    # Save to both locations
    fname = f"painting_{id_val}.png"
    plt.savefig(base_test_path / fname, dpi=200)
    plt.savefig(shared_output_path / fname, dpi=200)
    plt.close()

    print(f"✅ Saved: {fname} to both output directories.")
