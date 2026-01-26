import pandas as pd
import json
import time
import os
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

# ================== LOAD ENV ==================
load_dotenv()

# ================== CONFIG ==================
REQUEST_DELAY = 1        # nghỉ giữa các request
RETRY_DELAY = 30         # nghỉ khi rate limit
MAX_RETRIES = 3

MODEL_NAME = "openai/gpt-4o-mini"
BASE_URL = "https://openrouter.ai/api/v1"


# ================== PROMPT ==================
def build_translation_prompt(dialogue: str) -> str:
    return f"""
Bạn là một biên dịch viên chuyên nghiệp, chuyên dịch các đoạn hội thoại lừa đảo qua điện thoại
(phone scam / fraud call) từ tiếng Anh sang tiếng Việt.

MỤC TIÊU:
- Bản dịch tự nhiên như người Việt nói
- Giữ đúng ngữ cảnh lừa đảo

QUY TẮC:
- "Innocent:" → "Người nhận cuộc gọi:"
- "Suspect:" → "Kẻ lừa đảo:"
- Officer → Cán bộ
- ma'am / sir → chị / anh
- Social Security Administration → Cơ quan Bảo hiểm Xã hội
- social security number → mã số BHXH / CCCD
- Federal Trade Commission → Ủy ban Thương mại Liên bang

YÊU CẦU:
- Giữ cấu trúc hội thoại
- Thể hiện rõ giọng đe dọa / thao túng nếu có
- KHÔNG giải thích
- KHÔNG thêm nội dung
- CHỈ TRẢ VỀ BẢN DỊCH

HỘI THOẠI:
{dialogue}
""".strip()


# ================== TRANSLATE ONE ==================
def translate_single(dialogue: str, client: OpenAI) -> str:
    prompt = build_translation_prompt(dialogue)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        msg = str(e).lower()
        if any(x in msg for x in ["429", "rate", "quota"]):
            raise Exception("RATE_LIMIT")
        raise e


# ================== IO HELPERS ==================
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
    data = load_progress(path)
    data.append(record)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ================== MAIN PROCESS ==================
def process_csv(csv_file, output_json, api_key, test_mode=False):
    client = OpenAI(api_key=api_key, base_url=BASE_URL)

    print("🎯 OpenRouter model:", MODEL_NAME)
    print("✅ Kết nối OpenRouter OK\n")

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
            continue

        to_translate.append({
            "id": line_id,
            "dialogue": dialogue,
            "personality": row.get("personality"),
            "type": row.get("type"),
            "labels": row.get("labels")
        })

    print(f"🚀 Cần dịch: {len(to_translate)} mẫu\n")

    success = 0

    for idx, item in enumerate(tqdm(to_translate, desc="Translating"), 1):
        retry = 0
        while retry < MAX_RETRIES:
            try:
                vi_text = translate_single(item["dialogue"], client)

                if len(vi_text) < 50:
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
                    print("\n🧪 SAMPLE:")
                    print("EN:", item["dialogue"][:120], "...")
                    print("VI:", vi_text[:120], "...\n")

                break

            except Exception as e:
                if "RATE_LIMIT" in str(e):
                    retry += 1
                    wait = RETRY_DELAY * retry
                    print(f"\n⏸️ Rate limit → chờ {wait}s ({retry}/{MAX_RETRIES})")
                    time.sleep(wait)
                else:
                    retry += 1
                    print(f"\n❌ Lỗi dòng {item['id']}: {e}")
                    time.sleep(5)

        time.sleep(REQUEST_DELAY)

    print("\n🎉 HOÀN TẤT")
    print(f"✅ Thành công: {success}")
    print(f"💾 Output: {output_json}")

    return load_progress(output_json), errors


# ================== ENTRY ==================
if __name__ == "__main__":
    API_KEY = os.getenv("OPENROUTER_API_KEY")
    if not API_KEY:
        raise RuntimeError("❌ Chưa set OPENROUTER_API_KEY")

    input_csv = r"F:\Projetcs\data_scam\raw\BothBosu\agent_conversation_all.csv"
    output_json = r"F:\Projetcs\data_scam\processed\agent_conversation_all.json"

    print("1. TEST MODE (10 dòng)")
    print("2. FULL MODE")
    mode = input("Chọn 1 hoặc 2: ").strip()

    process_csv(
        input_csv,
        output_json,
        API_KEY,
        test_mode=(mode == "1")
    )
