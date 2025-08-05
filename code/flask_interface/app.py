from flask import Flask, render_template, request, send_file
import torch
from diffusers import StableDiffusion3Pipeline
from pathlib import Path
from PIL import Image
import json
import os
import argparse
from io import BytesIO
from datetime import datetime

# Optional MinIO
try:
    from minio import Minio
    MINIO_AVAILABLE = True
except ImportError:
    MINIO_AVAILABLE = False

# ------------------- Argument Parsing -------------------
parser = argparse.ArgumentParser()
parser.add_argument("--token_dict", required=True, help="Path to special_token_dict.json")
parser.add_argument("--lora_dir", required=True, help="Path to LoRA checkpoint directory")
parser.add_argument("--output_dir", help="Directory to save local images")
parser.add_argument("--use_minio", action="store_true", help="If set, upload images to MinIO")
args = parser.parse_args()

if not args.use_minio and not args.output_dir:
    raise ValueError("You must specify --output_dir if MinIO is not used.")

# ------------------- Flask Setup -------------------
app = Flask(__name__)

# Load token dictionary
with open(args.token_dict, "r", encoding="utf-8") as f:
    token_dict = json.load(f)
for key in token_dict:
    token_dict[key].insert(0, "")
token_dict["gender"] = ["", "as male", "as female"]

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load SD3 pipeline
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
).to(device)
pipe.load_lora_weights(args.lora_dir)
pipe.set_progress_bar_config(disable=True)

# ------------------- MinIO Setup -------------------
minio_client = None
minio_bucket = None
minio_endpoint = None

if args.use_minio:
    if not MINIO_AVAILABLE:
        raise RuntimeError("MinIO not installed. Run: pip install minio")

    minio_endpoint = os.environ.get("MINIO_ENDPOINT")
    minio_access_key = os.environ.get("MINIO_ACCESS_KEY")
    minio_secret_key = os.environ.get("MINIO_SECRET_KEY")
    minio_bucket = os.environ.get("MINIO_BUCKET")

    if not all([minio_endpoint, minio_access_key, minio_secret_key, minio_bucket]):
        raise ValueError("Missing MinIO credentials in environment variables.")

    minio_client = Minio(
        minio_endpoint,
        access_key=minio_access_key,
        secret_key=minio_secret_key,
        secure=True
    )

    if not minio_client.bucket_exists(minio_bucket):
        minio_client.make_bucket(minio_bucket)

# ------------------- Routes -------------------
@app.route("/", methods=["GET", "POST"])
def index():
    image_url = None

    if request.method == "POST":
        content = request.form.get("content", "")
        artist = request.form.get("artist", "")
        year = request.form.get("year", "")
        gender = request.form.get("gender", "")
        style = request.form.get("style", "")
        loc = request.form.get("location", "")
        net = request.form.get("interaction", "")
        steps = int(request.form.get("steps", 30))

        tokens = [t for t in [artist, year, gender, style, loc, net] if t]
        prompt = f"A painting of {content.strip()} {' '.join(tokens)}."
        print("Prompt:", prompt)

        generator = torch.Generator(device=device).manual_seed(42)
        image = pipe(prompt, num_inference_steps=steps, generator=generator).images[0]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generated_{timestamp}.png"

        if args.use_minio:
            minio_path = f"generated/{filename}"
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            buffer.seek(0)

            minio_client.put_object(
                bucket_name=minio_bucket,
                object_name=minio_path,
                data=buffer,
                length=buffer.getbuffer().nbytes,
                content_type="image/png"
            )

            # Try presigned URL first (for private buckets)
            try:
                image_url = minio_client.presigned_get_object(
                    bucket_name=minio_bucket,
                    object_name=minio_path,
                    expiry=3600
                )
            except Exception:
                # Fallback to public URL
                image_url = f"https://{minio_endpoint}/{minio_bucket}/{minio_path}"

        else:
            out_path = Path(args.output_dir) / filename
            out_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(out_path)
            image_url = f"/generated/{filename}"

    return render_template(
        "index.html",
        token_dict=token_dict,
        image_url=image_url
    )

@app.route("/generated/<filename>")
def serve_image(filename):
    if args.use_minio:
        return "Local serving is disabled when using MinIO.", 404
    return send_file(Path(args.output_dir) / filename, mimetype="image/png")

# ------------------- Run -------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)
