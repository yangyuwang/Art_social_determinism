import json
from pathlib import Path
import pandas as pd
from collections import Counter
import re
import numpy as np
from itertools import combinations
from random import shuffle, seed
import os
# from transformers import (
#     CLIPTokenizer, CLIPTextModel, T5TokenizerFast, 
#     T5EncoderModel, PretrainedConfig,
# )

# from diffusers import StableDiffusion3Pipeline
# import torch

np.random.seed(42)
seed(42)

# base_model = "stabilityai/stable-diffusion-3-medium-diffusers"
# target_dir = Path("/u/wangyd/sd3-custom")

# print("Loading SD3 pipeline...")
# pipe = StableDiffusion3Pipeline.from_pretrained(
#     base_model,
#     torch_dtype=torch.float16,
#     variant="fp16"
# )

# pipe.save_pretrained(target_dir)

# print("model saved!")

# Demographic Info
path_info = Path("/raven/u/wangyd/mpib/chm-artistic-social-determinism/Data/demographic_information.json") 
with path_info.open(encoding="utf-8") as f:
    data_info = json.load(f) 

# Location and Gender Extraction
artist_loc_dict = {}
artist_gender_dict = {}
artist_interact_dict = {}

for k, v in data_info.items():
    loc_lst = {}
    net_lst = {}

    # birth to death
    b_loc = v["birth"]["location"]
    d_loc = v["death"]["location"]
    b_year = d_year = None

    if b_loc:
        b_year = v["birth"]["year"]
    if d_loc:
        d_year = v["death"]["year"]

    if b_year and d_year:
        for year in range(b_year, d_year + 1):
            net_lst[year] = set()
            if year < d_year:
                loc_lst[year] = tuple(b_loc.values())
            else:
                loc_lst[year] = tuple(d_loc.values())

    # residences
    r_locs = v["residences"]

    if b_year and d_year and r_locs:
        for r_loc in r_locs:
            r_st = r_loc["start_year"]
            r_ed = r_loc["end_year"]
            
            if r_st and r_ed:
                for year in range(r_st, r_ed + 1):
                    loc_lst[year] = tuple(r_loc["location"].values())
            elif r_st:
                for year in range(r_st, d_year + 1):
                    loc_lst[year] = tuple(r_loc["location"].values())
            elif r_ed:
                for year in range(b_year, r_ed + 1):
                    loc_lst[year] = tuple(r_loc["location"].values())
    # interactions
    r_nets = v["interactions"]

    if b_year and d_year and r_nets:
        for r_net in r_nets:
            r_year = r_net["year"]
            
            if r_year:
                for year in range(r_year, d_year + 1):
                    net_lst[year].update([r_net["name"].lower()])

    artist_gender_dict[k] = v["gender"]
    artist_loc_dict[k] = loc_lst
    artist_interact_dict[k] = net_lst

# Content Extraction
content = {}
path_content = Path("/raven/u/wangyd/mpib/chm-artistic-social-determinism/Data/painting_content.jsonl") 
with path_content.open() as f:
    for line in f:
        if line.strip():
            content.update(json.loads(line))

content_clean = {k: re.search(r'This painting depicts (.+?)\.', c).group(1)  for k, c in content.items() if re.search(r'This painting depicts (.+?)\.', c)}

# Template Making

artwork_path = Path("/raven/u/wangyd/mpib/chm-artistic-social-determinism/Data/artwork_data_merged.csv")
artwork = pd.read_csv(artwork_path)

valid_years = artwork["Year"].dropna().astype(str).str.strip()
valid_years = valid_years[valid_years.str.isdigit()].astype(int)
year_counts = Counter(valid_years)


jsonl_out  = Path("/raven/u/wangyd/mpib/chm-artistic-social-determinism/Data/dreambooth_dataset/train/metadata.jsonl")
jsonl_out_full = Path("/raven/u/wangyd/mpib/chm-artistic-social-determinism/Data/test_data.jsonl")
counts = Counter()
all_entries = []
all_entries_full = []

artist_tokens = set()
year_tokens = set()
style_tokens = set()

gender_tokens = set()
loc_tokens = set()
net_tokens = set()
n = 0

for artist, image_n, year, styles in zip(artwork["Artist_name"], artwork["image_n"], artwork["Year"], artwork["Style"]):
    if str(year).strip().isdigit() and int(year) >= 1400 and pd.notna(image_n):
        n += 1

        artist = re.sub(r"^en/", "", str(artist))
        real_year = int(year)
        count_in_year = year_counts[real_year]
        std_dev = max(1, 5 / (count_in_year**0.5))
        sampled_year = int(np.random.normal(loc=real_year, scale=std_dev))
        if sampled_year < 1400: sampled_year = 1400
        if sampled_year > 2024: sampled_year = 2024

        if data_info.get(artist, None) and content_clean.get(str(int(image_n)), None):
            image_path = Path(f"{str(int(image_n))}.jpg")
            image_path_full = Path(f"/raven/u/wangyd/mpib/chm-artistic-social-determinism/Data/dreambooth_dataset/train/{str(int(image_n))}.jpg")

            # Artist token and year token
            token_dict = {"artist": f"<artist_{artist}>",
                          "year" : f"<year_{sampled_year}>"}
            artist_tokens.update([f"<artist_{artist}>"])
            year_tokens.update([f"<year_{sampled_year}>"])

            # Style token
            if styles and pd.notna(styles):
                style_lst = []
                for style in styles.split("|"):
                    style = re.sub(r"\s*\(.*?\)", "", style).strip()
                    clean_style = "_".join(style.lower().split(" "))
                    style_tokens.update([f"<style_{clean_style}>"])
                    style_lst.append(f"<style_{clean_style}>")
                
                if style_lst: token_dict["style"] = style_lst
                
            # Gender token
            if artist_gender_dict.get(artist, None):
                token_dict["gender"] = f"<gender_{artist_gender_dict.get(artist, None).lower()}>"
                gender_tokens.update([f"<gender_{artist_gender_dict.get(artist, None).lower()}>"])

            # Location token
            if artist_loc_dict.get(artist, None):
                loc_dict = artist_loc_dict.get(artist, None)
                if int(year) in loc_dict.keys():
                    loc_list = []
                    for ind, loc in enumerate(loc_dict[int(year)]):
                        if loc:
                            clean_loc = "_".join(loc.lower().split(" "))
                            clean_loc = re.sub(r"\.", "", clean_loc).strip()
                            clean_loc = re.sub(r",", "", clean_loc).strip()
                            if clean_loc != "n/a":
                                loc_list.append(f"<loc_{clean_loc}>")
                                loc_tokens.update([f"<loc_{clean_loc}>"])

                    if loc_list: token_dict["loc"] = loc_list
        
            # Network token
            if artist_interact_dict.get(artist, None):
                net_dict = artist_interact_dict.get(artist, None)
                if int(year) in net_dict.keys():
                    net_list = []
                    for ind, net in enumerate(list(net_dict[int(year)])):
                        if net:
                            clean_net = "_".join(net.lower().split(" "))
                            clean_net = re.sub(r"\.", "", clean_net).strip()
                            clean_net = re.sub(r",", "", clean_net).strip()
                            if clean_net != "n/a":
                                net_list.append(f"<net_{clean_net}>")
                                net_tokens.update([f"<net_{clean_net}>"])
                    
                    if net_list: token_dict["net"] = net_list

            # Clean up all tokens
            token_list_clean = set()
            for tokens in token_dict.values():
                if isinstance(tokens, list):
                    token_list_clean.update(tokens)
                else:
                    token_list_clean.update([tokens])
            token_list_clean = list(token_list_clean)
            
            counts.update(token_list_clean)

            if len(token_dict.keys()) >= 5:
                caption_content = f"A painting of {content_clean[str(int(image_n))]}."
                all_entries_full.append({
                    "file_name": str(image_path_full),
                    "content": caption_content,
                    "tokens": token_dict
                    })
            
                # combinations of tokens
                token_pairs = list(combinations(token_list_clean, 2))
                shuffle(token_pairs)

                if len(token_pairs) >= 5:
                    max_prompts_per_image = 5

                else:
                    max_prompts_per_image = len(token_pairs)

                for pair in token_pairs[:max_prompts_per_image]:
                    caption = f"A painting of {content_clean[str(int(image_n))]}.{' '.join(pair)}"
                    all_entries.append({
                        "file_name": str(image_path),
                        "text": caption
                    })


shuffle(all_entries)
shuffle(all_entries_full)
special_token_dict = {"artist": list(artist_tokens), 
                      "year": list(year_tokens), 
                      "style": list(style_tokens),
                      "gender": list(gender_tokens), 
                      "location": list(loc_tokens),
                      "interaction": list(net_tokens)}

special_token = set()
for token_set in special_token_dict.values():
    special_token.update(token_set)

print(f"Training sample size: {len(all_entries)} \n Training image size: {n}")
with jsonl_out.open("w", encoding="utf-8") as fp:
    for entry in all_entries:
        fp.write(json.dumps(entry, ensure_ascii=False) + "\n")

with jsonl_out_full.open("w", encoding="utf-8") as fp:
    for entry in all_entries_full[:2000]:
        fp.write(json.dumps(entry, ensure_ascii=False) + "\n")

with open("/u/wangyd/mpib/chm-artistic-social-determinism/Data/special_token.txt", "w", encoding="utf-8") as f:
   for tok in sorted(special_token):
       f.write(tok + "\n")

with open("/u/wangyd/mpib/chm-artistic-social-determinism/Data/special_token_dict.json", "w", encoding="utf-8") as f:
    json.dump(special_token_dict, f, ensure_ascii=False, indent=2)

for tk, n in counts.most_common(10):
   print(f"{tk:20} {n}")

# tokenizer = CLIPTokenizer.from_pretrained(target_dir / "tokenizer")
# print(f"➕ Adding {len(special_token)} special tokens...")
# num_added = tokenizer.add_tokens(list(special_token))
# print(f"{num_added} tokens added.")

# print("Resizing text encoder embedding layer...")
# text_encoder = CLIPTextModel.from_pretrained(target_dir / "text_encoder")
# text_encoder.resize_token_embeddings(len(tokenizer))

# tokenizer.save_pretrained(target_dir / "tokenizer")
# text_encoder.save_pretrained(target_dir / "text_encoder")

# pipe = StableDiffusion3Pipeline.from_pretrained(
#     "stabilityai/stable-diffusion-3-medium-diffusers",
#     tokenizer=tokenizer,
#     text_encoder=text_encoder,
# )

# pipe.save_pretrained("/u/wangyd/sd3-custom")

# print("Done! Model with injected tokens saved to:")
# print(f"   {target_dir}")

# def inject_special_tokens(
#     model_root: str | Path,
#     special_tokens: list[str] | set[str],
#     *,
#     suffixes: tuple[str, ...] = ("", "_2", "_3"),   # check tokenizers, tokenizer_2, tokenizer_3
# ) -> None:
#     """
#     Add `special_tokens` to every tokenizer inside an SD3 checkpoint and
#     resize the corresponding text‑encoder embedding layers.

#     Parameters
#     ----------
#     model_root : str | Path
#         Folder that contains sub‑folders like `tokenizer`, `tokenizer_2`,
#         `text_encoder`, `text_encoder_2`, …
#     special_tokens : list[str] | set[str]
#         Tokens you want to inject (must be strings without spaces).
#     suffixes : tuple[str, ...], optional
#         Which suffixes to try.  Default ("", "_2", "_3") covers
#         tokenizer, tokenizer_2, tokenizer_3.
#     """
#     model_root = Path(model_root)

#     for suff in suffixes:
#         tok_dir = model_root / f"tokenizer{suff}"
#         enc_dir = model_root / f"text_encoder{suff}"
#         if not tok_dir.exists() or not enc_dir.exists():
#             # Skip if that pair doesn't exist in this checkpoint
#             continue

#         # ---------- Detect whether this pair is CLIP or T5 ----------
#         cfg = PretrainedConfig.from_pretrained(enc_dir)
#         arch = cfg.architectures[0]

#         if arch.startswith("CLIP"):
#             TokCls, EncCls = CLIPTokenizer, CLIPTextModel
#         elif arch.startswith("T5"):
#             TokCls, EncCls = T5TokenizerFast, T5EncoderModel
#         else:
#             print(f"Unknown architecture {arch} in {enc_dir}, skipping.")
#             continue

#         print(f"Updating {tok_dir.name} / {enc_dir.name} ({arch})")

#         tokenizer = TokCls.from_pretrained(tok_dir)
#         n_added = tokenizer.add_tokens(list(special_tokens))
#         if n_added == 0:
#             print("all tokens already present, nothing to do.")
#             continue

#         text_encoder = EncCls.from_pretrained(enc_dir)
#         text_encoder.resize_token_embeddings(len(tokenizer))

#         tokenizer.save_pretrained(tok_dir)
#         text_encoder.save_pretrained(enc_dir)
#         print(f"added {n_added} tokens; new vocab size: {len(tokenizer)}")

#     print("All available tokenizers updated.\n")

# inject_special_tokens(target_dir, special_token)