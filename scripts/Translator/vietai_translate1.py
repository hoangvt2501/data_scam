import json
import os
from datetime import datetime
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# =========================
# CONFIG
# =========================
MODEL_NAME = "VietAI/envit5-translation"

INPUT_FILE = "data/dialogues.json"
OUTPUT_FILE = "data/dialogues_vi.json"
CHECKPOINT_FILE = "data/translation_checkpoint.json"

MAX_LENGTH = 512
SAVE_EVERY = 10


# =========================
# MODEL SETUP
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
model.eval()


# =========================
# TRANSLATION
# =========================
def translate_text(text: str) -> str:
    input_text = f"en: {text}"

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=MAX_LENGTH,
            num_beams=5,
            early_stopping=True
        )

    translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated.replace("vi:", "").strip()


def translate_sample(sample: dict) -> dict:
    new_sample = dict(sample)
    new_turns = []

    for turn in sample["turns"]:
        new_turn = dict(turn)
        new_turn["utterance"] = translate_text(turn["utterance"])
        new_turns.append(new_turn)

    new_sample["turns"] = new_turns
    return new_sample


# =========================
# CHECKPOINT
# =========================
def load_checkpoint(path):
    if not os.path.exists(path):
        return set()

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return set(data.get("processed_ids", []))


def save_checkpoint(path, processed_ids):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "processed_ids": list(processed_ids),
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            f,
            ensure_ascii=False,
            indent=2
        )


# =========================
# MAIN PROCESS
# =========================
def process_dataset(input_file, output_file, checkpoint_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    processed_ids = load_checkpoint(checkpoint_file)

    translated_data = []
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            translated_data = json.load(f)

    start_time = datetime.now()

    for sample in tqdm(data, total=total, desc="Translating"):
        dialogue_id = sample["dialogue_id"]

        if dialogue_id in processed_ids:
            continue

        translated_sample = translate_sample(sample)
        translated_data.append(translated_sample)
        processed_ids.add(dialogue_id)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(translated_data, f, ensure_ascii=False, indent=2)

        if len(processed_ids) % SAVE_EVERY == 0:
            save_checkpoint(checkpoint_file, processed_ids)

    save_checkpoint(checkpoint_file, processed_ids)

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"Completed: {len(processed_ids)}/{total}")
    print(f"Total time: {elapsed / 60:.2f} minutes")
    print(f"Output saved to: {output_file}")


# =========================
# ENTRY
# =========================
if __name__ == "__main__":
    process_dataset(INPUT_FILE, OUTPUT_FILE, CHECKPOINT_FILE)