"""Probe whether JevK5 text weights can make image decisions with Qwen3.5's vision tower.

This is an exploratory weight transplant, not a trained multimodal checkpoint.
Run on a CUDA GPU with transformers >=5.17, Pillow, and torchvision.
Use --width 350 for a faster, lower-resolution image check.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw, ImageFont
from safetensors import safe_open
from transformers import AutoModelForMultimodalLM, AutoProcessor

BASE = "Qwen/Qwen3.5-4B"
JEVK5 = "alibiserikbay/JevK5"
LETTERS = "ABCDEFGHIJKLMNOP"
SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)
QUESTIONS = [
    ("total", "What is the final total due on this invoice?", ["$80.00", "$8.00", "$88.00"], 2),
    ("vendor", "Which company issued this invoice?", ["Cedar Labs", "Northwind Supplies"], 1),
    ("due_date", "What is the payment due date?", ["2026-09-03", "2026-10-03"], 1),
]


def make_invoice(path: Path, width: int) -> Image.Image:
    image = Image.new("RGB", (1000, 690), "white")
    draw = ImageDraw.Draw(image)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 30)
    bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
    draw.text((60, 40), "INVOICE", fill="black", font=bold)
    lines = [
        "Vendor: Northwind Supplies",
        "Bill to: Cedar Labs",
        "Invoice date: 2026-09-03",
        "Payment due: 2026-10-03",
        "Subtotal: $80.00",
        "Tax: $8.00",
        "TOTAL DUE: $88.00",
    ]
    for index, line in enumerate(lines):
        draw.text((60, 130 + index * 72), line, fill="black", font=font)
    if width != image.width:
        image = image.resize((width, round(image.height * width / image.width)), Image.Resampling.LANCZOS)
    image.save(path)
    return image


def decide(model, processor, image, question, with_image: bool, temperature: float):
    name, criterion, options, gold = question
    payload = json.dumps(
        {
            "evidence": "Read the attached invoice image.",
            "criterion": criterion,
            "options": [
                {"letter": LETTERS[i], "description": f"{i}: {option}"}
                for i, option in enumerate(options)
            ],
        },
        ensure_ascii=False,
    )
    content = ([{"type": "image", "image": image}] if with_image else []) + [
        {"type": "text", "text": payload}
    ]
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": content},
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    slots = [processor.tokenizer.encode(LETTERS[i], add_special_tokens=False) for i in range(len(options))]
    if any(len(slot) != 1 for slot in slots):
        raise ValueError("Answer letters must each have one token")
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        logits = model(**inputs, use_cache=False).logits[0, -1, [slot[0] for slot in slots]]
        probabilities = torch.softmax(logits.float() / temperature, dim=-1).cpu().tolist()
    torch.cuda.synchronize()
    result = {
        "case": name,
        "with_image": with_image,
        "gold": options[gold],
        "prediction": options[max(range(len(options)), key=probabilities.__getitem__)],
        "probabilities": dict(zip(options, probabilities, strict=True)),
        "tokens": int(inputs["input_ids"].shape[-1]),
        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
    }
    print(json.dumps(result), flush=True)
    return result


def transplant_jevk5_text(model):
    path = hf_hub_download(JEVK5, "model.safetensors")
    weights = model.state_dict()
    copied = 0
    with safe_open(path, framework="pt", device="cpu") as checkpoint, torch.no_grad():
        for key in checkpoint.keys():
            if key not in weights:
                raise KeyError(f"Missing target weight: {key}")
            source = checkpoint.get_tensor(key)
            target = weights[key]
            if source.shape != target.shape:
                raise ValueError(f"Shape mismatch at {key}: {source.shape} != {target.shape}")
            target.copy_(source.to(device=target.device, dtype=target.dtype))
            copied += 1
    print(f"Copied {copied} JevK5 text tensors into the full vision model", flush=True)
    return copied


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=500)
    args = parser.parse_args()
    output = Path(f"vision-probe-{args.width}px-20260924.json")
    image_path = Path(f"vision-probe-invoice-{args.width}px.png")
    image = make_invoice(image_path, args.width)
    processor = AutoProcessor.from_pretrained(BASE)
    model = AutoModelForMultimodalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, device_map="auto"
    ).eval()
    print(f"Loaded {BASE}; GPU allocated {torch.cuda.memory_allocated() / 1024**3:.2f} GiB", flush=True)
    rows = []
    for question in QUESTIONS:
        for with_image in (False, True):
            rows.append({"model": "Qwen base", **decide(model, processor, image, question, with_image, 1.0)})
    copied = transplant_jevk5_text(model)
    temperature = 1.532
    for question in QUESTIONS:
        for with_image in (False, True):
            rows.append({"model": "JevK5 text + Qwen vision", **decide(model, processor, image, question, with_image, temperature)})
    output.write_text(
        json.dumps(
            {"base": BASE, "text_weights": JEVK5, "gpu": torch.cuda.get_device_name(0),
             "copied_text_tensors": copied, "image": str(image_path),
             "image_width": args.width, "cases": rows},
            indent=2,
        ) + "\n"
    )
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
