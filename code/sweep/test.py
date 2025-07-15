import subprocess
from pathlib import Path
from pprint import pprint

from datasets import load_dataset


# ────────────────────────────────────────────────────────────────────────────────
# CONFIG –---------- edit these four lines …                                    │
#               or export them as environment variables and use os.getenv().    │
# ────────────────────────────────────────────────────────────────────────────────
DATASET_DIR   = Path("/raven/u/wangyd/mpib/chm-artistic-social-determinism/Data/dreambooth_dataset")  # folder ABOVE train/
IMAGE_COLUMN  = "image"
CAPTION_COLUMN = "text"        # change to "prompt" if you kept that key
MODEL_DIR     = None           # e.g. Path("/u/wangyd/sd3-custom") after you added special tokens
#TRAIN_SCRIPT  = Path("/path/to/train_dreambooth_lora_sd3.py")
# ────────────────────────────────────────────────────────────────────────────────


def test_dataset(dataset_dir: Path, img_col: str = "image", cap_col: str = "text", n: int = 3):
    print(f"\n▶ Loading dataset from {dataset_dir} …")
    ds = load_dataset("imagefolder", data_dir=str(dataset_dir), split="train")
    print(f"✓ Loaded {len(ds):,} examples; columns = {list(ds.features.keys())}")

    for i in range(min(n, len(ds))):
        ex = ds[i]
        img = ex[img_col]
        print(f"\nExample {i}")
        print(f"  caption : {ex[cap_col]!r}")
        print(f"  image   : mode={img.mode}, size={img.size}, mean-pixel={sum(img.getextrema()[1])//3}")


def dry_run(train_script: Path, dataset_dir: Path, img_col: str, cap_col: str, model_dir: Path | None):
    """Launches the trainer for exactly one optimisation step via accelerate."""
    print("\n▶ Running a one-step dry-run …")

    cmd = [
        "accelerate", "launch", str(train_script),
        "--pretrained_model_name_or_path", "stabilityai/stable-diffusion-3-medium",
        "--dataset_name", str(dataset_dir),
        "--split", "train",
        "--image_column", img_col,
        "--caption_column", cap_col,
        "--max_train_steps", "1",
        "--validation_steps", "1",
        "--train_batch_size", "1",
        "--gradient_accumulation_steps", "1",
        "--learning_rate", "1e-5",
        "--output_dir", "/tmp/sd3_lora_smoketest",
        "--logging_steps", "1",
        "--checkpointing_steps", "1",
        "--validation_prompts", "A test prompt",
    ]

    if model_dir:
        cmd.extend(["--tokenizer_name", str(model_dir / "tokenizer"),
                    "--text_encoder_path", str(model_dir / "text_encoder")])

    print("Command line:\n", " ".join(cmd), "\n")
    completed = subprocess.run(cmd, check=False)
    if completed.returncode == 0:
        print("✓ Dry run finished without error.")
    else:
        print(f"❌ Dry run exited with code {completed.returncode}.")


if __name__ == "__main__":
    test_dataset(DATASET_DIR, IMAGE_COLUMN, CAPTION_COLUMN, n=3)