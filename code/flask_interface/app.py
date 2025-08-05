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
import boto3
from botocore.client import Config
from typing import Optional

# ------------------- S3 Helpers -------------------

def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["MINIO_URL"],
        aws_access_key_id=os.environ["MINIO_ACCESS_KEY"],
        aws_secret_access_key=os.environ["MINIO_SECRET_KEY"],
        config=Config(signature_version="s3v4")
    )

def upload_bytes_to_s3(
    image_bytes: BytesIO,
    bucket_name: Optional[str],
    s3_key: str,
    content_type: str = "image/png"
):
    if bucket_name is None:
        bucket_name = os.environ.get("MINIO_BUCKET")
    assert bucket_name, "MINIO_BUCKET environment variable must be set"

    s3_client = get_s3_client()
    s3_client.upload_fileobj(
        Fileobj=image_bytes,
        Bucket=bucket_name,
        Key=s3_key,
        ExtraArgs={"ContentType": content_type}
    )

def clear_generated_images_from_s3(bucket_name: str, prefix: str = "generated/"):
    s3_client = get_s3_client()
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

    if "Contents" in response:
        objects_to_delete = [{"Key": obj["Key"]} for obj in response["Contents"]]
        s3_client.delete_objects(
            Bucket=bucket_name,
            Delete={"Objects": objects_to_delete}
        )


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
pipe.load_lora_weights(
    args.lora_dir,
    weight_name="pytorch_lora_weights.safetensors",
    local_files_only=True
)
pipe.set_progress_bar_config(disable=True)

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
        minio_path = f"generated/{filename}"

        if args.use_minio:
            minio_bucket = os.environ["MINIO_BUCKET"]
            # Clear previous images
            clear_generated_images_from_s3(minio_bucket, prefix="generated/")
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            buffer.seek(0)
            upload_bytes_to_s3(buffer, bucket_name=minio_bucket, s3_key=minio_path)
            image_url = f"/image/{minio_path}"
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
def serve_image_local(filename):
    if args.use_minio:
        return "Local image serving is disabled when using MinIO.", 404
    return send_file(Path(args.output_dir) / filename, mimetype="image/png")

@app.route("/image/<path:key>")
def serve_image_minio(key):
    bucket = os.environ["MINIO_BUCKET"]
    s3 = get_s3_client()
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return send_file(obj["Body"], mimetype=obj["ContentType"])
    except Exception as e:
        return f"Failed to retrieve: {e}", 404

# ------------------- Run -------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)
