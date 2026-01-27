import pandas as pd
import os
import time
import json
from openai import OpenAI
from tqdm import tqdm
from scam_config import SCAM_GROUPS

INPUT_FILE = r"f:\Projetcs\data_scam\raw\BothBosu\agent_conversation_all.csv"
OUTPUT_FILE = r"f:\Projetcs\data_scam\processed\agent_conversation_vietnamese.csv"

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def get_add_info_prompt():
    scam_list_text = "\n".join([f"{k}. {v}" for k, v in SCAM_GROUPS.items()])
    return f"""
Bạn là một chuyên gia phân tích an ninh mạng và lừa đảo trực tuyến. 
Nhiệm vụ của bạn là:
1. Dịch đoạn hội thoại sau sang tiếng Việt tự nhiên, giữ nguyên ngữ cảnh.
2. Phân loại đoạn hội thoại vào MỘT trong 29 nhóm kịch bản lừa đảo dưới đây dựa trên nội dung chính.

Danh sách 29 nhóm kịch bản:
{scam_list_text}

Định dạng trả về (JSON):
{{
    "vietnamese_dialogue": "Nội dung hội thoại tiếng Việt...",
    "scam_type_id": ID_của_nhóm (số nguyên từ 1-29),
    "scam_type_name": "Tên nhóm kịch bản tương ứng",
    "explanation": "Giải thích ngắn gọn tại sao chọn nhóm này"
}}

Chỉ trả về JSON, không thêm văn bản nào khác.
"""

def process_dialogue(dialogue):
    prompt = get_add_info_prompt()
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Hội thoại cần xử lý:\n{dialogue}"}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error processing dialogue: {e}")
        return None

def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    if not os.path.exists(INPUT_FILE):
        print(f"File không tồn tại: {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"Đã tải {len(df)} dòng dữ liệu.")

   
    if os.path.exists(OUTPUT_FILE):
        df_existing = pd.read_csv(OUTPUT_FILE)
        start_idx = len(df_existing)
        print(f"Tiếp tục từ dòng {start_idx}...")
    else:
        pd.DataFrame(columns=['original_dialogue', 'vietnamese_dialogue', 'scam_type_id', 'scam_type_name', 'explanation', 'original_personality', 'original_type', 'original_labels']).to_csv(OUTPUT_FILE, index=False)

    for i in tqdm(range(start_idx, len(df))):
        row = df.iloc[i]
        original_dialogue = row['dialogue']
        
        result = process_dialogue(original_dialogue)
        
        if result:
            new_row = {
                'original_dialogue': original_dialogue,
                'vietnamese_dialogue': result.get('vietnamese_dialogue', ''),
                'scam_type_id': result.get('scam_type_id', ''),
                'scam_type_name': result.get('scam_type_name', ''),
                'explanation': result.get('explanation', ''),
                'original_personality': row.get('personality', ''),
                'original_type': row.get('type', ''),
                'original_labels': row.get('labels', '')
            }
            pd.DataFrame([new_row]).to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
        else:

            pass
        
        time.sleep(0.5)

if __name__ == "__main__":
    main()
