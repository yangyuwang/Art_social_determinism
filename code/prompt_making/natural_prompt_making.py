import json
from pathlib import Path
import pandas as pd
from collections import Counter
import re
import numpy as np
from itertools import combinations
from random import shuffle, seed
import os
from random import shuffle, seed

from collections import defaultdict

np.random.seed(42)
seed(42)

# Paths
artwork_path = Path("/raven/u/wangyd/mpib/chm-artistic-social-determinism/Data/artwork_data_merged.csv")
path_info = Path("/raven/u/wangyd/mpib/chm-artistic-social-determinism/Data/demographic_information.json")
path_content = Path("/raven/u/wangyd/mpib/chm-artistic-social-determinism/Data/painting_content.jsonl")
jsonl_out = Path("/raven/u/wangyd/mpib/chm-artistic-social-determinism/Data/dreambooth_dataset/train/metadata.jsonl")
jsonl_out_full = Path("/raven/u/wangyd/mpib/chm-artistic-social-determinism/Data/natural_prompt_dataset/full.jsonl")

os.makedirs("/raven/u/wangyd/mpib/chm-artistic-social-determinism/Data/natural_prompt_dataset", exist_ok=True)

# Load metadata
with path_info.open(encoding="utf-8") as f:
    data_info = json.load(f)

content = dict()
with path_content.open() as f:
    for line in f:
        if line.strip():
            content.update(json.loads(line))

content_clean = {k: re.search(r'This painting depicts (.+?)\.', c).group(1)  for k, c in content.items() if re.search(r'This painting depicts (.+?)\.', c)}

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

n = 0

# Main loop
for artist, image_n, year, styles in zip(artwork["Artist_name"], artwork["image_n"], artwork["Year"], artwork["Style"]):
   if str(year).strip().isdigit() and int(year) >= 1400 and pd.notna(image_n):
        
        artist = re.sub(r"^en/", "", str(artist))
        real_year = int(year)
        count_in_year = year_counts[real_year]
        std_dev = max(1, 5 / (count_in_year**0.5))
        sampled_year = int(np.random.normal(loc=real_year, scale=std_dev))
        if sampled_year < 1400: sampled_year = 1400
        if sampled_year > 2024: sampled_year = 2024

        if data_info.get(artist, None) and content_clean.get(str(int(image_n)), None):
            n += 1
            token_dict = {}
            image_path = Path(f"{str(int(image_n))}.jpg")
            image_path_full = Path(
                f"/raven/u/wangyd/mpib/chm-artistic-social-determinism/Data/dreambooth_dataset/train/{str(int(image_n))}.jpg"
            )

            phrases = [
                f"{' '.join(part.capitalize() for part in artist.split('-'))}"
            ]
            token_dict["artist"] = f"{' '.join(part.capitalize() for part in artist.split('-'))}"
            special_token_dict["artist"].add(f"{' '.join(part.capitalize() for part in artist.split('-'))}")

            # Gender
            gender = artist_gender_dict.get(artist)
            if gender:
                phrases.append(f"as {gender.lower()}")
                token_dict["gender"] = f"as {gender.lower()}"
                special_token_dict["gender"].add(f"as {gender.lower()}")
            
            phrases.append(f"in the year of {sampled_year}")
            token_dict["year"] = f"in the year of {sampled_year}"
            special_token_dict["year"].add(f"in the year of {sampled_year}")

            # Style
            if pd.notna(styles):
                style_col = []
                for style in styles.split("|"):
                    clean = re.sub(r"\s*\(.*?\)", "", style).strip()
                    style_col.append(clean)
                
                if style_col:
                    phrases.append(f"in the style of {', '.join(style_col)}")
                    token_dict["style"] = f"in the style of {', '.join(style_col)}"
                    special_token_dict["style"].add(f"in the style of {', '.join(style_col)}")

            # Location
            loc_years = artist_loc_dict.get(artist, {})
            if real_year in loc_years:
                loc_col = []
                for loc in loc_years[real_year]:
                    if loc and loc.lower() != "n/a":
                        loc_col.append(loc.replace('.', '').replace(',', '').strip())
                
                if loc_col:
                    phrases.append(f"in the place of {', '.join(loc_col)}")
                    token_dict["loc"] = f"in the place of {', '.join(loc_col)}"
                    special_token_dict["loc"].add(f"in the place of {', '.join(loc_col)}")

            # Interactions
            net_years = artist_interact_dict.get(artist, {})
            if real_year in net_years:
                net_col = []
                for net in list(net_years[real_year]):
                    if net:
                        net_col.append(net.replace('.', '').replace(',', '').strip())
                
                if net_col:
                    phrases.append(f"knew {', '.join(net_col)}")
                    token_dict["net"] = f"knew {', '.join(net_col)}"
                    special_token_dict["net"].add(f"knew {', '.join(net_col)}")

            # Compose prompt
            caption_content = f"A painting of {content_clean[str(int(image_n))]}"
            full_prompt = f"{caption_content} by the artist {' '.join(phrases)}."

            all_entries_full.append({
                "file_name": str(image_path_full),
                "content": caption_content,
                "metadata": token_dict,
            })

            # Token pair combinations
            phrase_pairs = list(combinations(phrases, 3))
            shuffle(phrase_pairs)

            for pair in phrase_pairs[:5]:
                prompt = f"{caption_content} by the artist {' '.join(pair)}."
                all_entries.append({
                    "file_name": str(image_path),
                    "text": prompt
                })

# Save output
jsonl_out.parent.mkdir(parents=True, exist_ok=True)
with jsonl_out.open("w", encoding="utf-8") as f:
    for entry in all_entries:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

shuffle(all_entries_full)
with jsonl_out_full.open("w", encoding="utf-8") as f:
    for entry in all_entries_full[:2000]:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"✅ Saved {len(all_entries)} training prompts and {len(all_entries_full[:2000])} full prompts for {n} images.")
print(f"📁 Output directory: {jsonl_out.parent}")

# Build special_token_dict
# Convert sets to sorted lists
special_token_dict = {k: sorted(list(v)) for k, v in special_token_dict.items()}

# Save to JSON file
output_path = Path("/raven/u/wangyd/mpib/chm-artistic-social-determinism/Data/natural_prompt_dataset/special_token_dict.json")
with output_path.open("w", encoding="utf-8") as f:
    json.dump(special_token_dict, f, ensure_ascii=False, indent=2)

print("✅ special_token_dict saved to:")
print(output_path)

