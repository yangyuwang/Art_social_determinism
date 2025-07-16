import argparse, os, pathlib, torch
from diffusers import StableDiffusion3Pipeline
from PIL import Image

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--lora_dir",  required=True)
    p.add_argument("--prompts",   required=True)
    p.add_argument("--out_dir",   required=True)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--seed",  type=int, default=42)
    args = p.parse_args()

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model {args.base_model} and LoRA from {args.lora_dir}")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.base_model, torch_dtype=torch.float16
    ).to("cuda")
    pipe.load_lora_weights(args.lora_dir)
    pipe.set_progress_bar_config(disable=False)

    with open(args.prompts, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

    for idx, prompt in enumerate(prompts, 1):
        gen = torch.Generator(device="cuda").manual_seed(args.seed + idx)
        img  = pipe(prompt, num_inference_steps=args.steps, generator=gen).images[0]
        img.save(out / f"{idx:03d}.png")
        print(f"{idx:03d}.png  --  {prompt[:70]}")

    print("Images saved to", out)

if __name__ == "__main__":
    main()
