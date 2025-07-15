import json
from pathlib import Path
import pandas as pd
from collections import Counter
import re
import numpy as np
from itertools import combinations
from random import shuffle, seed
import os
#from transformers import CLIPTokenizer, CLIPTextModel

#from diffusers import StableDiffusion3Pipeline
#import torch

np.random.seed(42)
seed(42)

#base_model = "stabilityai/stable-diffusion-3-medium-diffusers"
#target_dir = Path("/u/wangyd/sd3-custom")

#print("Loading SD3 pipeline...")
#pipe = StableDiffusion3Pipeline.from_pretrained(
#    base_model,
#    torch_dtype=torch.float16,
#    variant="fp16"
#)

#pipe.save_pretrained(target_dir)

#print("model saved!")

# Demographic Info
path_info = Path("/raven/u/wangyd/mpib/chm-artistic-social-determinism/Data/demographic_information.json") 
with path_info.open(encoding="utf-8") as f:
    data_info = json.load(f) 

# Location and Gender Extraction
artist_loc_dict = {}
artist_gender_dict = {}

for k, v in data_info.items():
    lst = {}

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
            if year < d_year:
                lst[year] = tuple(b_loc.values())
            else:
                lst[year] = tuple(d_loc.values())

    # residences
    r_locs = v["residences"]

    if b_year and d_year and r_locs:
        for r_loc in r_locs:
            r_st = r_loc["start_year"]
            r_ed = r_loc["end_year"]
            
            if r_st and r_ed:
                for year in range(r_st, r_ed + 1):
                    lst[year] = tuple(r_loc["location"].values())
            elif r_st:
                for year in range(r_st, d_year + 1):
                    lst[year] = tuple(r_loc["location"].values())
            elif r_ed:
                for year in range(b_year, r_ed + 1):
                    lst[year] = tuple(r_loc["location"].values())

    artist_gender_dict[k] = v["gender"]
    artist_loc_dict[k] = lst

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
counts = Counter()
special_token = set()
all_entries = []

for artist, image_n, year in zip(artwork["Artist_name"], artwork["image_n"], artwork["Year"]):
    if str(year).strip().isdigit() and int(year) >= 1400 and pd.notna(image_n):
        artist = re.sub(r"^en/", "", str(artist))
        real_year = int(year)
        count_in_year = year_counts[real_year]
        std_dev = max(1, 50 / (count_in_year**0.5))
        sampled_year = int(np.random.normal(loc=real_year, scale=std_dev))

        if data_info.get(artist, None) and content_clean.get(str(int(image_n)), None):
            image_path = Path(f"{str(int(image_n))}.jpg")

            token_list = [f"<artist_{artist}>", f"<year_{sampled_year}>"]

            if artist_gender_dict.get(artist, None):
                token_list.append(f"<gender_{artist_gender_dict.get(artist, None).lower()}>")

            if artist_loc_dict.get(artist, None):
                loc_dict = artist_loc_dict.get(artist, None)
                if int(year) in loc_dict.keys():
                    for loc in loc_dict[int(year)]:
                        if loc:
                            token_list.append(f"<loc_{loc.lower()}>")
            
            token_list_clean = ["_".join(token.split(" ")) for token in token_list]
            special_token.update(token_list_clean)
            counts.update(token_list_clean)
            caption = f"A painting of {content_clean[str(int(image_n))]} {' '.join(token_list_clean)}"
            all_entries.append({
                "file_name": str(image_path),
                "text": caption
                })

            # combinations of tokens
            #token_pairs = list(combinations(token_list_clean, 2))
            #shuffle(token_pairs)

            #if len(token_pairs) >= 3:
            #    max_prompts_per_image = 3
            #else:
            #    max_prompts_per_image = len(token_pairs)

            #for pair in token_pairs[:max_prompts_per_image]:
            #    caption = f"A painting of {content_clean[str(int(image_n))]} {' '.join(pair)}"
            #    all_entries.append({
            #        "file_name": str(image_path),
            #        "text": caption
            #    })

shuffle(all_entries)

with jsonl_out.open("w", encoding="utf-8") as fp:
    for entry in all_entries:
        fp.write(json.dumps(entry, ensure_ascii=False) + "\n")
#with open("/ptmp/wangyd/special_token.txt", "w") as f:
#    for tok in sorted(special_token):
#        f.write(tok + "\n")

for tk, n in counts.most_common(20):
    print(f"{tk:20} {n}")


#tokenizer = CLIPTokenizer.from_pretrained(target_dir / "tokenizer")
#print(f"➕ Adding {len(special_token)} special tokens...")
#num_added = tokenizer.add_tokens(list(special_token))
#print(f"{num_added} tokens added.")

#print("Resizing text encoder embedding layer...")
#text_encoder = CLIPTextModel.from_pretrained(target_dir / "text_encoder")
#text_encoder.resize_token_embeddings(len(tokenizer))

#tokenizer.save_pretrained(target_dir / "tokenizer")
#text_encoder.save_pretrained(target_dir / "text_encoder")

#print("Done! Model with injected tokens saved to:")
#print(f"   {target_dir}")
