import pandas as pd
import json
from google import genai
from tqdm import tqdm
import time
import os
from dotenv import load_dotenv

load_dotenv()

REQUEST_DELAY = 1 
RETRY_DELAY = 30  
MAX_RETRIES = 3 


def translate_single(dialogue, client):
    prompt = f"""Dịch đoạn hội thoại này sang tiếng Việt tự nhiên:

RULES:
- "Innocent:" → "Người nhận cuộc gọi:"
- "Suspect:" → "Kẻ lừa đảo:"
- Social Security Administration → Cơ quan Bảo hiểm Xã hội
- social security number → mã số BHXH/CCCD
- Officer → Cán bộ
- ma'am/sir → chị/anh
- Federal Trade Commission → Ủy ban Thương mại Liên bang
- Xưng hô tự nhiên theo văn hóa VN

CHỈ TRẢ VỀ BẢN DỊCH, KHÔNG THÊM GÌ:

{dialogue}"""
    
    try:
        response = client.models.generate_content(
            model='models/gemini-3-flash',
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        error_msg = str(e).lower()
        if any(x in error_msg for x in ["quota", "rate", "429", "resource_exhausted"]):
            raise Exception("RATE_LIMIT")
        raise e


def load_progress(output_file):
    if not os.path.exists(output_file):
        return []
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"📂 Đã load {len(data)} dòng đã dịch")
            return data
    except:
        return []


def save_one_result(output_file, result):
    results = load_progress(output_file)
    results.append(result)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def process_csv(csv_file, output_json, api_key, test_mode=False):

    client = genai.Client(api_key=api_key)
    print("🎯 Model: gemini-pro-latest")
    print("✅ Đã kết nối Gemini API\n")
    
    print("📖 Đang đọc CSV...")
    df = pd.read_csv(csv_file, encoding='utf-8-sig')
    print(f"✅ Đã đọc {len(df)} dòng\n")
    
    if test_mode:
        print("⚠️ TEST MODE: Chỉ dịch 10 dòng\n")
        df = df.head(10)
    
    existing = load_progress(output_json)
    translated_ids = {r['id'] for r in existing}
    
    if translated_ids:
        print(f"♻️ Resume: Đã có {len(translated_ids)} dòng, tiếp tục...\n")

    to_translate = []
    errors = []
    
    for idx, row in df.iterrows():
        line_id = idx + 1
        
        if line_id in translated_ids:
            continue
        
        dialogue = str(row['dialogue']).strip() if pd.notna(row['dialogue']) else ""
        personality = str(row['personality']).strip() if pd.notna(row['personality']) else ""
        type_field = str(row['type']).strip() if pd.notna(row['type']) else ""
        labels = int(row['labels']) if pd.notna(row['labels']) else None
        
        if not dialogue or len(dialogue) < 50:
            errors.append({"line": line_id, "reason": "Dialogue quá ngắn"})
            continue
        
        if not personality or not type_field or labels is None:
            errors.append({"line": line_id, "reason": "Thiếu metadata"})
            continue
        
        to_translate.append({
            'id': line_id,
            'dialogue': dialogue,
            'personality': personality,
            'type': type_field,
            'labels': labels
        })
    
    if not to_translate:
        print("✅ TẤT CẢ ĐÃ DỊCH XONG!")
        return existing, errors
    
    print(f"🚀 Cần dịch: {len(to_translate)} mẫu")
    print(f"⏱️ Delay: {REQUEST_DELAY}s/mẫu\n")

    success = 0
    
    for idx, item in enumerate(tqdm(to_translate, desc="Đang dịch"), 1):
        retry = 0
        
        while retry < MAX_RETRIES:
            try:
                translation = translate_single(item['dialogue'], client)
                
                if not translation or len(translation) < 50:
                    raise Exception("Dịch quá ngắn")

                result = {
                    "id": item['id'],
                    "dialogue_original": item['dialogue'],
                    "dialogue_vietnamese": translation,
                    "personality": item['personality'],
                    "type": item['type'],
                    "labels": item['labels']
                }
                
                save_one_result(output_json, result)
                success += 1
                
                if idx == 1:
                    print(f"\n{'='*70}")
                    print("✅ MẪU ĐẦU TIÊN!")
                    print(f"{'='*70}")
                    print("EN:", item['dialogue'][:120], "...")
                    print("\nVI:", translation[:120], "...")
                    print(f"{'='*70}\n")
                
                break 
                
            except Exception as e:
                if "RATE_LIMIT" in str(e):
                    retry += 1
                    wait = RETRY_DELAY * retry
                    print(f"\n⏸️ Rate limit! Chờ {wait}s... ({retry}/{MAX_RETRIES})")
                    time.sleep(wait)
                    continue
                
                retry += 1
                print(f"\n❌ Mẫu {item['id']} lỗi: {e}")
                
                if retry < MAX_RETRIES:
                    print(f"🔄 Retry {retry}/{MAX_RETRIES}...")
                    time.sleep(5)
                else:
                    errors.append({
                        "line": item['id'],
                        "error": str(e),
                        "preview": item['dialogue'][:100]
                    })
                    break
        
        if idx < len(to_translate):
            time.sleep(REQUEST_DELAY)
    
    final = load_progress(output_json)
    
    if errors:
        error_file = output_json.replace('.json', '_errors.json')
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*70}")
    print(f"🎉 HOÀN TẤT!")
    print(f"✅ Dịch thành công: {success}/{len(to_translate)} mẫu mới")
    print(f"📊 Tổng trong file: {len(final)} dòng")
    print(f"❌ Lỗi: {len(errors)} dòng")
    print(f"💾 File: {output_json}")
    print(f"{'='*70}")
    
    return final, errors


# ===== MAIN =====
if __name__ == "__main__":
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    if not GEMINI_API_KEY:
        print("❌ Chưa set GEMINI_API_KEY trong .env!")
        exit(1)
    
    input_csv = r"F:\Projetcs\data_scam\raw\BothBosu\agent_conversation_all.csv"
    output_json = r"F:\Projetcs\data_scam\processed\agent_conversation_all.json"
    
    print("\n" + "="*70)
    print("CHỌN CHẾ ĐỘ:")
    print("1. TEST MODE - Dịch 10 dòng đầu")
    print("2. FULL MODE - Dịch tất cả")
    print("="*70)
    
    mode = input("Nhập 1 hoặc 2: ").strip()
    test_mode = (mode == "1")
    
    print(f"\n🚀 BẮT ĐẦU {'TEST' if test_mode else 'FULL'} MODE\n")
    
    results, errors = process_csv(
        input_csv,
        output_json,
        GEMINI_API_KEY,
        test_mode=test_mode
    )
    
    if results:
        print("\n📄 MẪU KẾT QUẢ:")
        print(json.dumps(results[0], ensure_ascii=False, indent=2)[:300])