import pandas as pd
import json
import time
import os
from tqdm import tqdm
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re

load_dotenv()

REQUEST_DELAY = 0.5
MAX_RETRIES = 3

MODEL_NAME = "VietAI/envit5-translation"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model: {MODEL_NAME}")
print(f"Device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

print("Model loaded successfully!\n")


def translate_single(dialogue: str) -> str:
    """
    Dịch dialogue - tự động chia nhỏ nếu quá dài
    """
    try:
        # Kiểm tra độ dài
        test_tokens = tokenizer.encode(f"en: {dialogue}", add_special_tokens=True)
        
        # Nếu quá dài (>400 tokens input), chia nhỏ theo câu
        if len(test_tokens) > 400:
            return translate_long_dialogue(dialogue)
        
        # Nếu ngắn, dịch bình thường
        input_text = f"en: {dialogue}"
        
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            max_length=512,
            truncation=True
        ).input_ids.to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=1024,  # TĂNG lên 1024 cho output
                num_beams=5,
                early_stopping=True
            )
        
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        translation = post_process_translation(translation)
        
        return translation.strip()
        
    except Exception as e:
        raise e


def translate_long_dialogue(dialogue: str) -> str:
    """
    Dịch dialogue dài bằng cách chia nhỏ theo từng lượt hội thoại
    """
    # Tách theo pattern "Innocent:" và "Suspect:"
    parts = re.split(r'(Innocent:|Suspect:)', dialogue)
    
    translated_parts = []
    current_speaker = None
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        # Nếu là label speaker
        if part in ['Innocent:', 'Suspect:']:
            current_speaker = part
            continue
        
        # Dịch từng đoạn text
        if current_speaker:
            input_text = f"en: {part}"
            
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                max_length=512,
                truncation=True
            ).input_ids.to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=1024,
                    num_beams=5,
                    early_stopping=True
                )
            
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Ghép với speaker label
            translated_parts.append(f"{current_speaker} {translation}")
            current_speaker = None
    
    # Ghép tất cả lại
    full_translation = ' '.join(translated_parts)
    full_translation = post_process_translation(full_translation)
    
    return full_translation.strip()


def post_process_translation(translation: str) -> str:
    
    translation = re.sub(r'^vi:\s*', '', translation, flags=re.IGNORECASE)
    translation = re.sub(r'\nvi:\s*', '\n', translation, flags=re.IGNORECASE)
    
    label_replacements = [
        ("Vô tội:", "Người nhận cuộc gọi:"),
        ("Nghi ngờ:", "Kẻ lừa đảo:"),
        ("Nghi phạm:", "Kẻ lừa đảo:"),
        ("Nghi can:", "Kẻ lừa đảo:"),
        ("Người bị nghi ngờ:", "Kẻ lừa đảo:"),
        ("Innocent:", "Người nhận cuộc gọi:"),
        ("Suspect:", "Kẻ lừa đảo:"),
    ]
    
    for old, new in label_replacements:
        translation = translation.replace(old, new)
    
    replacements = {
        r"Cục An sinh xã hội": "Bộ Công an",
        r"Cơ quan An sinh Xã hội": "Bộ Công an",
        r"Bảo hiểm Xã hội": "Bộ Công an",
        r"Social Security Administration": "Bộ Công an",
        r"Ủy ban Thương mại Liên bang": "Ủy ban Thương mại",
        r"Federal Trade Commission": "Ủy ban Thương mại",
        r"Dịch vụ Tín dụng Allied": "Dịch vụ Tín dụng Allied",
    
        r"số an sinh xã hội": "số CCCD",
        r"social security number": "số CCCD",
        r"\bSSN\b": "CCCD",

        r"Sĩ quan": "Cán bộ",
        r"Officer": "Cán bộ",
        r"Thanh tra viên": "Thanh tra",
        
        r"thưa bà": "chị",
        r"thưa ông": "anh",
        r"ma'am": "chị",
        r"sir": "anh",
        r"\bcô\b(?! ấy)": "chị", 
    }
    
    for pattern, replacement in replacements.items():
        translation = re.sub(pattern, replacement, translation, flags=re.IGNORECASE)
 
    translation = re.sub(
        r'(?<!^)(\s*)(Người nhận cuộc gọi:|Kẻ lừa đảo:)', 
        r'\n\2', 
        translation
    )
    
    translation = re.sub(r'\n\s*\n+', '\n', translation) 
    translation = re.sub(r' +', ' ', translation)          
    translation = re.sub(r'^\s+', '', translation)         
    
    return translation.strip()


def load_progress(path):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"📂 Resume: {len(data)} dòng đã dịch")
            return data
    except:
        return []


def save_one(path, record):
    """Lưu từng kết quả vào file JSON"""
    data = load_progress(path)
    data.append(record)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def process_csv(csv_file, output_json, test_mode=False):
    print("🎯 Model:", MODEL_NAME)
    print(f"🖥️ Device: {device}")
    print("✅ Sẵn sàng dịch\n")

    df = pd.read_csv(csv_file, encoding="utf-8-sig")
    print(f"📖 CSV rows: {len(df)}\n")

    if test_mode:
        df = df.head(10)
        print("⚠️ TEST MODE (10 dòng)\n")

    existing = load_progress(output_json)
    done_ids = {x["id"] for x in existing}

    to_translate = []
    errors = []

    for idx, row in df.iterrows():
        line_id = idx + 1

        if line_id in done_ids:
            continue

        dialogue = str(row.get("dialogue", "")).strip()

        if len(dialogue) < 50:
            errors.append({
                "line": line_id,
                "reason": "Dialogue quá ngắn"
            })
            continue

        to_translate.append({
            "id": line_id,
            "dialogue": dialogue,
            "personality": row.get("personality"),
            "type": row.get("type"),
            "labels": row.get("labels")
        })

    if not to_translate:
        print("✅ TẤT CẢ ĐÃ DỊCH XONG!")
        return existing, errors

    print(f"🚀 Cần dịch: {len(to_translate)} mẫu\n")

    success = 0

    for idx, item in enumerate(tqdm(to_translate, desc="Translating"), 1):
        retry = 0
        
        while retry < MAX_RETRIES:
            try:
                vi_text = translate_single(item["dialogue"])

                if len(vi_text) < 30:
                    raise Exception("Translation too short")

                result = {
                    "id": item["id"],
                    "dialogue_original": item["dialogue"],
                    "dialogue_vietnamese": vi_text,
                    "personality": item["personality"],
                    "type": item["type"],
                    "labels": item["labels"]
                }

                save_one(output_json, result)
                success += 1

                if idx == 1:
                    print("\n" + "="*70)
                    print("🧪 SAMPLE TRANSLATION:")
                    print("="*70)
                    print("📌 ORIGINAL:")
                    print(item["dialogue"][:500])
                    print("\n📌 VIETNAMESE:")
                    print(vi_text[:500])
                    print("="*70 + "\n")

                break  

            except Exception as e:
                retry += 1
                print(f"\n❌ Lỗi dòng {item['id']}: {e}")
                
                if retry < MAX_RETRIES:
                    print(f"🔄 Retry {retry}/{MAX_RETRIES}...")
                    time.sleep(2)
                else:
                    errors.append({
                        "line": item["id"],
                        "error": str(e),
                        "preview": item["dialogue"][:100]
                    })
                    break

        if idx < len(to_translate):
            time.sleep(REQUEST_DELAY)

    final = load_progress(output_json)

    if errors:
        error_file = output_json.replace('.json', '_errors.json')
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)
        print(f"⚠️ Đã lưu errors vào: {error_file}")

    print("\n" + "="*70)
    print("🎉 HOÀN TẤT!")
    print(f"✅ Thành công: {success}/{len(to_translate)} mẫu mới")
    print(f"📊 Tổng trong file: {len(final)} dòng")
    print(f"❌ Lỗi: {len(errors)} dòng")
    print(f"💾 Output: {output_json}")
    print("="*70)

    return final, errors


if __name__ == "__main__":
    input_csv = r"C:\Users\admin\Desktop\Hoangvt\data_scam\raw\BothBosu\agent_conversation_all.csv"
    output_json = r"C:\Users\admin\Desktop\Hoangvt\data_scam\processed\agent_conversation_all.json"

    print("\n" + "="*70)
    print("CHỌN CHẾ ĐỘ:")
    print("1. TEST MODE - Dịch 10 dòng đầu")
    print("2. FULL MODE - Dịch tất cả")
    print("="*70)
    
    mode = input("Chọn 1 hoặc 2: ").strip()

    process_csv(
        input_csv,
        output_json,
        test_mode=(mode == "1")
    )