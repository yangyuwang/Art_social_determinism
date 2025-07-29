"""
Generate SD‑3 images for a fixed set of prompt conditions and
compute CLIP similarity vs. the ground‑truth image.

This script supports JSONL format where each record looks like::

    {
      "file_name": "/path/to/img.jpg",
      "content": "A painting of …",
      "tokens": {
        "artist": "<artist_…>",
        "year":   "<year_…>",
        "style":  ["<style_…>", …],
        "gender": "<gender_…>",
        "loc":    ["<loc_…>", …],
        "net":    "<net_…>"            # optional
      }
    }

For every record we generate an image for the following **nine** conditions
(A, C, T1–T_all) and measure its cosine similarity to the real image using CLIP
(ViT‑L/14):

A   "A painting"
C   "A painting of …" (the *content* field only)
T1  content + <year_…>
T2  content + <style_…> (all style tokens joined)
T3  content + <gender_…>
T4  content + <loc_…>   (all loc tokens joined)
T5  content + <net_…>   (if available)
T6  content + <artist_…>
T_all  content + *all* tokens (year, artist, style, gender, loc, net)

Images are saved for every 10,000th real image, with all 8 conditions included.

Results are written to **results.csv** with columns::

    file_name, condition, similarity

and the generated images are saved to ``<out_root>/generated/<condition>/``.

Usage example
-------------
$ python quant_test.py \
      --base_model   /sd3-custom \
      --lora_dir     /ptmp/wangyd/sd3_lora_runs/array_20557153_5 \
      --jsonl        /mpib/chm-artistic-social-determinism/Data/test_data.jsonl \
      --out_root     /mpib/chm-artistic-social-determinism/Data/quant_eval \
      --steps 25 --seed 42
"""


import argparse, json, accelerate
import os
from pathlib import Path
from typing import Dict, List, Union
from safetensors.torch import load_file
import itertools
import re

import torch
import torchvision.transforms as T
from PIL import Image
from tqdm.auto import tqdm
import pandas as pd

import random

from diffusers import StableDiffusion3Pipeline, AutoencoderKL, Transformer2DModel
from transformers import (
    CLIPTokenizer, CLIPTextModelWithProjection,
    T5TokenizerFast, T5EncoderModel,
)

# ───────────────────────────── helpers ────────────────────────────────────────

def _sample_one(tokens: Dict[str, Union[str, List[str]]],
                rng: random.Random) -> Dict[str, str]:
    """
    Return a **copy** of *tokens* in which every list value is replaced by
    a single, randomly‑chosen element.
    """
    out: Dict[str, str] = {}
    for k, v in tokens.items():
        if not v:
            continue
        if isinstance(v, (list, tuple)):
            out[k] = rng.choice(v) 
        else:
            out[k] = v 
    return out

def slug(s: str, max_len: int = 80) -> str:
    """Return a filesystem‑safe slug (lower‑case, ASCII, hyphen‑separated)."""
    return re.sub(r"[^a-z0-9\-]+", "-", s.lower())[:max_len].strip("-") or "img"


def _flatten_tokens(tokens: Dict[str, Union[str, List[str]]]) -> List[str]:
    """Return a flat list of **all** token strings in *tokens*."""
    flat: List[str] = []
    for v in tokens.values():
        if not v:
            continue
        if isinstance(v, (list, tuple)):
            flat.extend(v)
        else:
            flat.append(v)
    return flat


def _get_token(tokens: Dict[str, Union[str, List[str]]], key: str) -> str:
    """Safely extract a token value (join lists with a space)."""
    v = tokens.get(key)
    if not v:
        return ""
    if isinstance(v, (list, tuple)):
        return " ".join(v)
    return v


# ───────────────────────────── main ──────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--lora_dir", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_root)
    gen_root = out_dir / "generated"
    gen_root.mkdir(parents=True, exist_ok=True)

    # 1) ─── read JSONL ───────────────────────────────────────────────────────
    records = []
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    # 2) ─── SD‑3 pipeline (with LoRA) ────────────────────────────────────────
    base = args.base_model
    weight_dtype = torch.float16
    device       = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusion3Pipeline.from_pretrained(
        base, torch_dtype=torch.float16,
    ).to(device)

    # Load LoRA weights manually
    pipe.load_lora_weights(args.lora_dir)

    pipe.set_progress_bar_config(disable=True)

    # 3) ─── CLIP model (ViT‑L/14) ────────────────────────────────────────────
    import clip  # pip install git+https://github.com/openai/CLIP.git

    clip_model, clip_pre = clip.load("ViT-L/14", device="cuda", jit=False)

    # 4) ─── iterate records / conditions ─────────────────────────────────────
    rows = []
    gen_id = 0  # used to vary the seed per prompt deterministically
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("file_name,A_sim,C_sim,T1_sim,T2_sim,T3_sim,T4_sim,T5_sim,T6_sim,T_all_sim,T1,T2,T3,T4,T5,T6\n")  # write CSV header

    for idx, rec in enumerate(tqdm(records, desc="Records")):
        real_path = Path(rec["file_name"])
        real_img = Image.open(real_path).convert("RGB")
        real_emb = clip_model.encode_image(
            clip_pre(real_img).unsqueeze(0).to("cuda")
        ).float()

        tokens = rec["metadata"]
        rng = random.Random(args.seed + idx)
        tokens_one = _sample_one(tokens, rng)
        content    = rec["content"].strip()

        # condition → prompt builder ------------------------------------------------
        def build_prompts() -> Dict[str, str]:
            t = tokens_one
            
            net_token = _get_token(t, 'net')
            if net_token:
                prompt_dict = {
                    "A": "A painting",
                    "C": f"{content}.",
                    "T1": f"{content} by the artist {_get_token(t, 'year')}.",
                    "T2": f"{content} by the artist {_get_token(t, 'style')}.",
                    "T3": f"{content} by the artist {_get_token(t, 'gender')}.",
                    "T4": f"{content} by the artist {_get_token(t, 'loc')}.",
                    "T5": f"{content} by the artist  {_get_token(t, 'net')}.",
                    "T6": f"{content} by the artist  {_get_token(t, 'artist')}.",
                    "T_all": f"{content} by the artist {' '.join(_flatten_tokens(t))}.",
                }
            else:
                prompt_dict = {
                    "A": "A painting",
                    "C": content,
                    "T1": f"{content} {_get_token(t, 'year')}",
                    "T2": f"{content} {_get_token(t, 'style')}",
                    "T3": f"{content} {_get_token(t, 'gender')}",
                    "T4": f"{content} {_get_token(t, 'loc')}",
                    "T5": "",
                    "T6": f"{content} {_get_token(t, 'artist')}",
                    "T_all": f"{content} {' '.join(_flatten_tokens(t))}",
                }
                
            return prompt_dict

        prompts = build_prompts()
        sim_list =[]
        token_list = [_get_token(tokens_one, "year"),
                _get_token(tokens_one, "style"),
                _get_token(tokens_one, "gender"),
                _get_token(tokens_one, "loc"),
                _get_token(tokens_one, "net"),
                _get_token(tokens_one, "artist")]
        print(prompts)

        save_images = (idx % 100 == 0)

        for cond, prompt in prompts.items():
            if prompt:
                gen_id += 1
                # ensure deterministic but different seeds
                g = torch.Generator(device="cuda").manual_seed(args.seed + gen_id)

                # ─── generate image ────────────────────────────────────────────
                token_ids = pipe.tokenizer(prompt, return_tensors="pt")["input_ids"][0]
                decoded_tokens = pipe.tokenizer.convert_ids_to_tokens(token_ids)
                print(f"CLIP tokenized ({len(decoded_tokens)}):", decoded_tokens)

                gen_img = pipe(prompt, num_inference_steps=args.steps, generator=g).images[0]

                # ─── similarity ───────────────────────────────────────────────
                gen_emb = clip_model.encode_image(
                    clip_pre(gen_img).unsqueeze(0).to("cuda")
                ).float()
                sim = torch.cosine_similarity(real_emb, gen_emb).item()
                sim_list.append(f"{sim:.6f}")

                # save image if this record is selected
                if save_images:
                    cond_dir = gen_root / cond
                    cond_dir.mkdir(exist_ok=True, parents=True)
                    fname = f"{real_path.stem}_{slug(cond)}.png"
                    gen_img_path = cond_dir / fname
                    gen_img.save(gen_img_path)
            else:
                sim_list.append("")

        with open(csv_path, "a", encoding="utf-8") as f:
            f.write(f"{real_path.name},{','.join(sim_list)},{','.join(token_list)}\n")

    print("\nCSV saved to", csv_path)
    print("Generated images under", gen_root)


if __name__ == "__main__":
    main()

