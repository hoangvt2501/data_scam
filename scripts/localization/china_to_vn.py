import google.generativeai as genai
import json
import os
from datetime import datetime
from typing import List, Dict
import time
from dotenv import load_dotenv

# Load biến môi trường từ file .env
load_dotenv()

class VietnameseContextConverter:
    def __init__(self, input_file: str, output_file: str, log_file: str):

        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("Không tìm thấy GEMINI_API_KEY trong file .env")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-3-flash-preview')
        self.input_file = input_file
        self.output_file = output_file
        self.log_file = log_file

        self._init_files()
    
    def _init_files(self):
        """Khởi tạo các file"""

        if not os.path.exists(self.output_file):
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"Bắt đầu chuyển đổi: {datetime.now()}\n")
            f.write(f"{'='*50}\n")
    
    def _log(self, message: str, print_console: bool = True):
        """Ghi log"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
        
        if print_console:
            print(log_message)
    
    def _create_prompt(self, dialogue: List[Dict]) -> str:
        """Tạo prompt cho Gemini"""
        dialogue_text = "\n".join([
            f"{turn['role']}: {turn['content']}" 
            for turn in dialogue
        ])
        
        prompt = f"""Bạn là chuyên gia ngôn ngữ tiếng Việt. Nhiệm vụ của bạn là chuyển đổi đoạn hội thoại sau sang ngữ cảnh tiếng Việt TỰ NHIÊN, GẦN GŨI hơn.

YÊU CẦU QUAN TRỌNG:
1. GIỮ NGUYÊN HOÀN TOÀN NỘI DUNG và Ý NGHĨA của cuộc hội thoại
2. Xưng hô phải THỐNG NHẤT và TỰ NHIÊN xuyên suốt (anh/em, bạn, mày/tao, chị/em, etc.)
3. Phân tích ngữ cảnh để chọn cách xưng hô phù hợp:
   - Nếu là cuộc gọi chính thức/công việc → dùng "anh/chị/em" hoặc "bạn"
   - Nếu là bạn bè thân thiết → có thể dùng "mày/tao", "bạn", "cậu/tớ"
   - Nếu là người lạ lịch sự → dùng "bạn" hoặc "anh/chị"
4. Chuyển các tên riêng Trung Quốc sang tên Việt Nam phù hợp (Tiểu Lý → Linh, Thiên Hà Ngân hàng → Ngân hàng Vietcombank, etc.)
5. Điều chỉnh ngữ cảnh, địa danh, đơn vị tiền tệ sang Việt Nam
6. Giữ nguyên cấu trúc role: "người gọi" và "người nghe"
7. Ngôn ngữ phải tự nhiên, gần gũi như người Việt nói chuyện thật

Đoạn hội thoại cần chuyển đổi:
{dialogue_text}

Hãy trả về kết quả dưới dạng JSON với format:
{{
    "dialogue": [
        {{"role": "người gọi", "content": "..."}},
        {{"role": "người nghe", "content": "..."}}
    ]
}}

CHỈ TRẢ VỀ JSON, KHÔNG THÊM BẤT KỲ TEXT NÀO KHÁC."""

        return prompt
    
    def _convert_dialogue(self, dialogue: List[Dict], retry_count: int = 3) -> List[Dict]:
        """Chuyển đổi một đoạn hội thoại"""
        prompt = self._create_prompt(dialogue)
        
        for attempt in range(retry_count):
            try:
                response = self.model.generate_content(prompt)
                result_text = response.text.strip()
    
                if result_text.startswith('```'):
                    result_text = result_text.split('```')[1]
                    if result_text.startswith('json'):
                        result_text = result_text[4:]
                    result_text = result_text.strip()
                
                result = json.loads(result_text)
                return result['dialogue']
                
            except json.JSONDecodeError as e:
                self._log(f"Lỗi parse JSON (lần {attempt + 1}/{retry_count}): {e}")
                if attempt == retry_count - 1:
                    raise
                time.sleep(2)
                
            except Exception as e:
                self._log(f"Lỗi API (lần {attempt + 1}/{retry_count}): {e}")
                if attempt == retry_count - 1:
                    raise
                time.sleep(2)
    
    def _save_single_result(self, item: Dict):
        """Lưu kết quả từng sample vào file"""

        with open(self.output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        data.append(item)
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def convert(self, start_id: int = None, end_id: int = None):
        """
        Chuyển đổi dữ liệu
        
        Args:
            start_id: ID bắt đầu (None = từ đầu)
            end_id: ID kết thúc (None = đến cuối)
        """
        with open(self.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if start_id is not None or end_id is not None:
            data = [
                item for item in data 
                if (start_id is None or item['_id'] >= start_id) and 
                   (end_id is None or item['_id'] <= end_id)
            ]
        
        total = len(data)
        self._log(f"Tổng số mẫu cần xử lý: {total}")
        
        success_count = 0
        error_count = 0
        
        for idx, item in enumerate(data, 1):
            item_id = item['_id']
            try:
                self._log(f"Đang xử lý [{idx}/{total}] - ID: {item_id}")
                
                # Chuyển đổi dialogue
                converted_dialogue = self._convert_dialogue(item['dialogue'])
                
                new_item = {
                    "_id": item_id,
                    "dialogue": converted_dialogue
                }
                
                # Lưu kết quả
                self._save_single_result(new_item)
                
                success_count += 1
                self._log(f"✓ Hoàn thành ID: {item_id} [{success_count}/{total}]")
                
                time.sleep(1)
                
            except Exception as e:
                error_count += 1
                self._log(f"✗ Lỗi xử lý ID {item_id}: {str(e)}")
                continue
        
        self._log(f"\n{'='*50}")
        self._log(f"KẾT THÚC CHUYỂN ĐỔI")
        self._log(f"Tổng số: {total}")
        self._log(f"Thành công: {success_count}")
        self._log(f"Lỗi: {error_count}")
        self._log(f"{'='*50}\n")


def main():
    # Cấu hình
    INPUT_FILE = r"C:\Users\admin\Desktop\Hoangvt\data_scam\translate\tele28k_harmless_translate.json"   # File dữ liệu đầu vào
    OUTPUT_FILE = r"C:\Users\admin\Desktop\Hoangvt\data_scam\localization\tele28k_harmless.json" # File kết quả
    LOG_FILE = r"C:\Users\admin\Desktop\Hoangvt\data_scam\conversion2_log.txt"  # File log
    
    print("="*60)
    print("CHƯƠNG TRÌNH CHUYỂN ĐỔI NGỮ CẢNH TIẾNG VIỆT")
    print("="*60)
    
    try:
        # Khởi tạo converter
        converter = VietnameseContextConverter(
            input_file=INPUT_FILE,
            output_file=OUTPUT_FILE,
            log_file=LOG_FILE
        )
        
        print("\nChọn cách xử lý:")
        print("1. Xử lý tất cả")
        print("2. Xử lý theo ID range")
        
        choice = input("\nNhập lựa chọn (1/2): ").strip()
        
        if choice == "2":
            start_id = input("Nhập ID bắt đầu (Enter để bỏ qua): ").strip()
            end_id = input("Nhập ID kết thúc (Enter để bỏ qua): ").strip()
            
            start_id = int(start_id) if start_id else None
            end_id = int(end_id) if end_id else None
            
            converter.convert(start_id=start_id, end_id=end_id)
        else:
            converter.convert()
        
        print(f"\n{'='*60}")
        print(f"HOÀN TẤT!")
        print(f"Kết quả: {OUTPUT_FILE}")
        print(f"Log: {LOG_FILE}")
        print(f"{'='*60}")
        
    except ValueError as e:
        print(f"\n❌ LỖI: {e}")
        print("Hãy kiểm tra file .env và đảm bảo có GEMINI_API_KEY")
    except FileNotFoundError as e:
        print(f"\n❌ LỖI: Không tìm thấy file - {e}")
    except Exception as e:
        print(f"\n❌ LỖI: {e}")


if __name__ == "__main__":
    main()