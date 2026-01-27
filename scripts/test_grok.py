import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def translate_single(dialogue, client):
    prompt = f"""Bạn là một chuyên gia bản địa hóa ngôn ngữ. 
Nhiệm vụ: Dịch đoạn hội thoại lừa đảo qua điện thoại (scam call) sau sang tiếng Việt.

YÊU CẦU PHONG CÁCH (QUAN TRỌNG):
1. VĂN NÓI TỰ NHIÊN: 
   - Dùng ngữ điệu hội thoại đời thường của người Việt. 
   - Sử dụng từ đệm phù hợp: "ạ, vâng, dạ, à, ừ, nhé, nha, hả, chứ...".
   - Câu văn có thể rút gọn chủ ngữ nếu ngữ cảnh cho phép.
   
2. NGỮ KHÍ NHÂN VẬT:
   - Suspect: Giọng điệu nghiêm trọng, đe dọa, gấp gáp hoặc dụ dỗ chuyên nghiệp.
   - Innocent: Giọng điệu lo lắng, bối rối, ngây thơ hoặc nghi ngờ.

QUY TẮC THUẬT NGỮ (LOCALIZATION):
- Giữ cấu trúc hội thoại
- "Innocent:" → "Người nhận cuộc gọi:" 
- "Suspect:" → "Kẻ lừa đảo:"
- "Social Security Administration" → "Cơ quan Bảo hiểm Xã hội" hoặc "Bộ Công an" (tùy ngữ cảnh dọa nạt)
- "social security number" → "số Căn cước công dân (CCCD)" hoặc "mã số định danh"
- "Officer" → "Cán bộ" hoặc "Thanh tra"
- "ma'am/sir" → "anh/chị" (xưng hô linh hoạt theo ngữ cảnh, không cứng nhắc)
- "Federal Trade Commission" → "Ủy ban Thương mại" hoặc "Cục Quản lý"

INPUT DIALOGUE:
{dialogue}

OUTPUT (Chỉ trả về nội dung dịch, không giải thích):"""
    
    try:
        response = client.chat.completions.create(
            model="grok-4-latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        print("❌ Chưa set XAI_API_KEY trong .env!")
        # api_key = input("Nhập XAI API Key: ")
    else:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
        )

        # Sample dialogue for testing
        sample_dialogue = """
Suspect: Hello, this is Officer John Smith from the Social Security Administration. Can you hear me?
Innocent: Yes, who is this?
Suspect: I'm calling to inform you that your social security number has been suspended due to suspicious activity.
Innocent: Suspended? Oh my god, what should I do?
Suspect: Don't worry, ma'am. To resolve this, you need to confirm your details immediately.
        """

        print("-" * 50)
        print("INPUT:\n", sample_dialogue)
        print("-" * 50)
        print("TRANSLATING WITH GROK...")
        
        result = translate_single(sample_dialogue, client)
        
        print("-" * 50)
        print("OUTPUT:\n", result)
        print("-" * 50)
