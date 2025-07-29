from flask import Flask, render_template, request, send_file, url_for
import torch
from diffusers import StableDiffusion3Pipeline
from transformers import CLIPTokenizer, CLIPTextModel
from huggingface_hub import login
from pathlib import Path
from PIL import Image
import json
import os
import argparse

# ------------------- Argument Parsing -------------------
parser = argparse.ArgumentParser()
parser.add_argument("--token_dict", required=True, help="Path to special_token_dict.json")
parser.add_argument("--lora_dir", required=True, help="Path to LoRA checkpoint directory")
parser.add_argument("--output_dir", required=True, help="Directory to save generated images")
args = parser.parse_args()

# ------------------- Flask Setup -------------------
app = Flask(__name__)

# Load token dictionary
with open(args.token_dict, "r", encoding="utf-8") as f:
    token_dict = json.load(f)
for key in token_dict:
    token_dict[key].insert(0, "")

token_dict["gender"] = ["", "as male", "as female"]

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load SD3 pipeline with optional custom components
pipe_kwargs = {
    "pretrained_model_name_or_path": "stabilityai/stable-diffusion-3-medium-diffusers",
    "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
}

pipe = StableDiffusion3Pipeline.from_pretrained(**pipe_kwargs).to(device)

# Load LoRA weights
pipe.load_lora_weights(args.lora_dir)
pipe.set_progress_bar_config(disable=True)

# ------------------- Web Routes -------------------
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
        print(prompt)

        generator = torch.Generator(device=device).manual_seed(42)
        image = pipe(prompt, num_inference_steps=steps, generator=generator).images[0]

        out_path = Path(args.output_dir) / "generated.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path)
        image_url = "generated.png"

    return render_template("index.html", token_dict=token_dict, image_url=image_url)

@app.route("/generated/<filename>")
def serve_image(filename):
    return send_file(Path(args.output_dir) / filename, mimetype="image/png")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)
