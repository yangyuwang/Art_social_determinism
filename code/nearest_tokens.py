#!/usr/bin/env python
"""
nearest_tokens.py  –  Inspect SD‑3/LoRA token embeddings

Example:
$ python nearest_tokens.py \
      --base_model  /u/wangyd/sd3-custom \
      --lora_dir    /u/wangyd/models/prodigy_running/checkpoint-38500 \
      --token_dict  /u/wangyd/mpib/chm-artistic-social-determinism/Data/special_token_dict.json \
      --category    style \
      --target      "<style_impressionism>" \
      --topk        15
"""

import argparse, json, torch, torch.nn.functional as F
from diffusers import StableDiffusion3Pipeline

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--lora_dir",  required=True)
    ap.add_argument("--token_dict", required=True)
    ap.add_argument("--category",   required=True, help="e.g. style, year, gender …")
    ap.add_argument("--target",     required=True, help="token to inspect, in <> brackets")
    ap.add_argument("--topk",       type=int, default=10)
    return ap.parse_args()

@torch.inference_mode()
def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 1) ─── load pipeline and extract tokenizer/text encoder ─────────────
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.base_model, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(device)
    pipe.load_lora_weights(args.lora_dir)

    tok = pipe.tokenizer
    txt = pipe.text_encoder
    txt.eval()

    # 2) ─── load token dict and pick category list ───────────────────────
    with open(args.token_dict, "r", encoding="utf-8") as f:
        token_dict = json.load(f)

    if args.category not in token_dict:
        raise KeyError(f"{args.category!r} not in token dict keys: {list(token_dict)}")
    cat_tokens = token_dict[args.category]

    # 3) ─── map tokens → ids → embeddings ────────────────────────────────
    emb_table = txt.get_input_embeddings().weight.detach().cpu()  # (vocab, dim)

    target_id = tok.encode(args.target, add_special_tokens=False)[0]
    target_emb = emb_table[target_id]

    cat_ids   = [tok.encode(t, add_special_tokens=False)[0] for t in cat_tokens]
    cat_embs  = emb_table[cat_ids]

    # 4) ─── similarity & ranking ─────────────────────────────────────────
    sims = F.cosine_similarity(target_emb.unsqueeze(0), cat_embs)  # (N,)
    topk = torch.topk(sims, k=min(args.topk, len(cat_tokens)))

    print(f"\nNearest to {args.target}:")
    for rank, (idx, score) in enumerate(zip(topk.indices, topk.values), 1):
        print(f"{rank:2d}. {cat_tokens[idx]}  ↔  {score:.4f}")

if __name__ == "__main__":
    main()
