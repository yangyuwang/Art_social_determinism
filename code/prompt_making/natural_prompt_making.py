import json
from pathlib import Path
import pandas as pd
from collections import Counter
import re
import numpy as np
from itertools import combinations
import os
from random import shuffle, seed, sample
import random
from tqdm import tqdm
from collections import defaultdict

np.random.seed(42)
seed(42)

# Paths
# Paths
artwork_path = Path("/raven/u/wangyd/mpib/chm-artistic-social-determinism/Data/artwork_data_merged.csv")
path_info = Path("/raven/u/wangyd/mpib/chm-artistic-social-determinism/Data/demographic_information.json")
jsonl_out = Path("/raven/u/wangyd/mpib/chm-artistic-social-determinism/Data/dreambooth_dataset/train/metadata.jsonl")
jsonl_out_full = Path("/raven/u/wangyd/mpib/chm-artistic-social-determinism/Data/natural_prompt_dataset/full.jsonl")
output_path = Path("/raven/u/wangyd/mpib/chm-artistic-social-determinism/Data/natural_prompt_dataset/special_token_dict.json")


os.makedirs("/raven/u/wangyd/mpib/chm-artistic-social-determinism/Data/natural_prompt_dataset", exist_ok=True)

# Load metadata
with path_info.open(encoding="utf-8") as f:
    data_info = json.load(f)

# Load painting_content_{i}.jsonl files
content = {}
for i in range(8):
    path_content = Path(f"/raven/u/wangyd/mpib/chm-artistic-social-determinism/Data/painting_content_{i}.jsonl")
    with path_content.open() as f:
        for line in f:
            if line.strip():
                content.update(json.loads(line))

artwork = pd.read_csv(artwork_path)
valid_years = artwork["Year"].dropna().astype(str).str.strip()
valid_years = valid_years[valid_years.str.isdigit()].astype(int)
year_counts = Counter(valid_years)

# Build artist dicts
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

# Output containers
all_entries = []
all_entries_full = []
special_token_dict = defaultdict(set)
n_images = 0

# Main loop
for artist, image_n, year, styles in tqdm(zip(artwork["Artist_name"], artwork["image_n"], artwork["Year"], artwork["Style"])):
    if str(year).strip().isdigit() and int(year) >= 1400 and pd.notna(image_n):
        artist = re.sub(r"^en/", "", str(artist))
        real_year = int(year)
        count_in_year = year_counts[real_year]
        std_dev = max(1, 5 / (count_in_year**0.5))
        sampled_year = int(np.random.normal(loc=real_year, scale=std_dev))
        sampled_year = max(1400, min(sampled_year, 2024))

        img_id_str = str(int(image_n))
        if data_info.get(artist) and img_id_str in content:
            n_images += 1
            token_dict = {}
            image_path = Path(f"{img_id_str}.jpg")
            image_path_full = Path(f"/raven/u/wangyd/mpib/chm-artistic-social-determinism/Data/dreambooth_dataset/train/{img_id_str}.jpg")

            tokens = []

            # Artist
            artist_tok = ' '.join(part.capitalize() for part in artist.split('-'))
            tokens.append(artist_tok)
            token_dict["artist"] = artist_tok
            special_token_dict["artist"].add(artist_tok)

            # Gender
            gender = artist_gender_dict.get(artist)
            if gender:
                gender_tok = f"as {gender.lower()}"
                tokens.append(gender_tok)
                token_dict["gender"] = gender_tok
                special_token_dict["gender"].add(gender_tok)

            # Year
            year_tok = f"in the year of {sampled_year}"
            tokens.append(year_tok)
            token_dict["year"] = year_tok
            special_token_dict["year"].add(year_tok)

            # Style
            if pd.notna(styles):
                style_lst = [re.sub(r"\\s*\\(.*?\\)", "", s).strip() for s in styles.split("|")]
                if style_lst:
                    style_tok = f"in the style of {', '.join(style_lst)}"
                    tokens.append(style_tok)
                    token_dict["style"] = style_tok
                    special_token_dict["style"].add(style_tok)

            # Location
            loc_years = artist_loc_dict.get(artist, {})
            if real_year in loc_years:
                locs = [loc.strip().replace('.', '').replace(',', '') for loc in loc_years[real_year] if loc and loc.lower() != "n/a"]
                if locs:
                    loc_tok = f"in the place of {', '.join(locs)}"
                    tokens.append(loc_tok)
                    token_dict["loc"] = loc_tok
                    special_token_dict["loc"].add(loc_tok)

            # Network
            net_years = artist_interact_dict.get(artist, {})
            if real_year in net_years:
                nets = [net.strip().replace('.', '').replace(',', '') for net in net_years[real_year] if net]
                if nets:
                    net_tok = f"knew {', '.join(nets)}"
                    tokens.append(net_tok)
                    token_dict["net"] = net_tok
                    special_token_dict["net"].add(net_tok)

            # Select 5 random caption contents
            for i in range(4):
                caption_content = f"A painting of {random.choice(content[img_id_str])}"
                n_style = random.randint(2, len(tokens))
                indices = sorted(sample(range(len(tokens)), n_style))
                style_sample = [tokens[i] for i in indices]
                prompt = f"{caption_content} by the artist {' '.join(style_sample)}."
                all_entries.append({
                    "file_name": str(image_path),
                    "text": prompt
                })

                if n_images % 10000 == 0:
                    print(all_entries[-1])

                if i == 0:
                    full_prompt = f"{caption_content} by the artist {' '.join(tokens)}."
                    all_entries_full.append({
                        "file_name": str(image_path_full),
                        "content": caption_content,
                        "metadata": token_dict,
                    })

# Save training prompts
with jsonl_out.open("w", encoding="utf-8") as f:
    for entry in all_entries:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# Save full prompts (limit to 2000)
shuffle(all_entries_full)
with jsonl_out_full.open("w", encoding="utf-8") as f:
    for entry in all_entries_full[:2000]:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# Save special token dictionary
special_token_dict = {k: sorted(list(v)) for k, v in special_token_dict.items()}
with output_path.open("w", encoding="utf-8") as f:
    json.dump(special_token_dict, f, ensure_ascii=False, indent=2)

print(f"✅ Saved {len(all_entries)} training prompts and {len(all_entries_full[:2000])} full prompts for {n_images} images.")
print(f"📁 Output directory: {jsonl_out.parent}")
print(f"✅ special_token_dict saved to: {output_path}")
