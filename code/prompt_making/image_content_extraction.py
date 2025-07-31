
#!/usr/bin/env python3
"""
Offline image-to-text captioner (LLaVA-NeXT).
Writes JSON-Lines: {"file_stem": "description …"}
"""

import argparse, json, os, torch, warnings, re
from pathlib import Path 
from PIL import Image
from tqdm import tqdm
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from collections import defaultdict


def main(img_dir: str,
         model_dir: str,
         out_path: str,
         length_list: list[int],
         fp16: bool = True):

    already_done = set()
    out_path = Path(out_path)

    if out_path.is_file():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        obj = json.loads(line)
                        already_done.update(obj.keys())   # each line has one key = file stem
                    except json.JSONDecodeError:
                        warnings.warn(f"Bad JSONL line skipped: {line[:80]}…")

    print(f"Found {len(already_done):,} captions in {out_path.name} – those images will be skipped.")

    # ---- 100 % offline -------------------------------------------------------
    os.environ["HF_HUB_OFFLINE"]       = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    dtype  = torch.float16 if fp16 and torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model from:", model_dir)
    processor = LlavaNextProcessor.from_pretrained(model_dir)
    model     = LlavaNextForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    ).eval()

    import logging
    logging.getLogger("transformers.generation_utils").setLevel(logging.ERROR)

    print("device:", next(model.parameters()).device, 
        "| cuda_available:", torch.cuda.is_available(),
        "| gpu_name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    # ---- collect images ------------------------------------------------------
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    all_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(exts)])

    if "SLURM_ARRAY_TASK_ID" in os.environ:
        rank = int(os.environ["SLURM_ARRAY_TASK_ID"])
        world_size = 8  # or pass as an arg if needed
        chunk_size = (len(all_files) + world_size - 1) // world_size
        all_files = all_files[rank * chunk_size : (rank + 1) * chunk_size]

    files = [f for f in all_files if os.path.splitext(f)[0] not in already_done and "metadata.jsonl" not in f.lower()]


    if not files:
        print("Nothing new to caption – exiting.")
        return

    with out_path.open("a", encoding="utf-8") as fp:
        BATCH = 16  # or larger if memory allows

        for i in tqdm(range(0, len(files), BATCH), desc="Describing"):
            batch_paths = files[i: i+BATCH]
            good_paths, images = [], []

            for p in batch_paths:
                try:
                    with Image.open(os.path.join(img_dir, p)) as im:
                        images.append(im.convert("RGB"))
                        good_paths.append(p)
                except OSError as e:
                    print("⚠️ skipped", p, "→", e)

            if not images:
                continue

            captions_per_image = defaultdict(list)

            for length in length_list:
                dynamic_prompt = (
                    f"In no more than {int(length)/4} words, describe the contents in the painting "
                    f"regardless of its style, in the format of 'This painting depicts ...'."
                )

                chats = [[{"role": "user",
                        "content": [{"type": "image"},
                                    {"type": "text", "text": dynamic_prompt}]}] for _ in images]

                prompts = processor.apply_chat_template(
                    chats, add_generation_prompt=True, tokenize=False)

                inputs = processor(images, prompts, return_tensors="pt", padding=True).to(device)
                if fp16 and device == "cuda":
                    inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
                    model = model.half()

                with torch.inference_mode():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=length + 20,
                        eos_token_id=processor.tokenizer.eos_token_id,
                        do_sample=True,
                        temperature=0.5,
                    )

                prompt_len = inputs["input_ids"].shape[1]

                for path, out_ids in zip(good_paths, out):
                    gen_only = out_ids[prompt_len:]  # define this properly
                    desc_full = processor.tokenizer.decode(gen_only, skip_special_tokens=True).strip()

                    match = re.search(r"This painting depicts(.*?)(?:[.。]|$)", desc_full, re.IGNORECASE)
                    if match:
                        desc = match.group(1).strip()
                        stem = os.path.splitext(os.path.basename(path))[0]
                        captions_per_image[stem].append(desc)

            for stem, desc_list in captions_per_image.items():
                fp.write(json.dumps({stem: desc_list}, ensure_ascii=False) + "\n")
            fp.flush()



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder",     help="directory with images")
    ap.add_argument("model_dir",  help="path to local LLaVA-NeXT checkpoint")
    ap.add_argument("--out",      default="descriptions.jsonl",
                    help="output JSONL file")
    ap.add_argument("--lengths", type=int, nargs="+", default=[32],
                help="List of max token lengths to generate multiple captions, e.g., --lengths 16 32 64 96 128")
    ap.add_argument("--no-fp16",  action="store_true",
                    help="disable half-precision (use full FP32)")
    args = ap.parse_args()

    main(args.folder, args.model_dir,
        args.out, args.lengths, not args.no_fp16)

