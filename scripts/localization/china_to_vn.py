import google.generativeai as genai
import json
import os
from datetime import datetime
from typing import List, Dict
import time
from dotenv import load_dotenv

load_dotenv()


class VietnameseContextConverter:
    def __init__(self, input_file: str, output_file: str, log_file: str, api_keys: List[str] = None):
        if api_keys is None:
            api_keys_str = os.getenv('GEMINI_API_KEYS')
            if not api_keys_str:
                raise ValueError("Không tìm thấy GEMINI_API_KEYS trong file .env")
            self.api_keys = [key.strip() for key in api_keys_str.split(',')]
        else:
            self.api_keys = api_keys
            
        if not self.api_keys:
            raise ValueError("Danh sách API keys trống!")
        
        self.current_key_index = 0
        self.input_file = input_file
        self.output_file = output_file
        self.log_file = log_file
        
        self.api_stats = {i: {'success': 0, 'errors': 0, 'key': key[:10] + '...'} 
                         for i, key in enumerate(self.api_keys)}
        
        self._init_files()
        self._init_model()
        
    def _init_model(self):
        """Khởi tạo model với API key hiện tại"""
        try:
            genai.configure(api_key=self.api_keys[self.current_key_index])
            self.model = genai.GenerativeModel('gemini-3-flash-preview')
            self._log(f"✓ Đã khởi tạo model với API key #{self.current_key_index + 1}")
        except Exception as e:
            self._log(f"✗ Lỗi khởi tạo model với API key #{self.current_key_index + 1}: {e}")
            raise

    def _switch_api_key(self) -> bool:
        """Chuyển sang API key tiếp theo"""
        old_index = self.current_key_index
        self.current_key_index += 1
        
        if self.current_key_index >= len(self.api_keys):
            self._log("✗ ĐÃ HẾT TẤT CẢ API KEYS!")
            return False
        
        try:
            self._log(f"→ Chuyển từ API key #{old_index + 1} sang API key #{self.current_key_index + 1}")
            self._init_model()
            return True
        except Exception as e:
            self._log(f"✗ Lỗi chuyển API key: {e}")
            return self._switch_api_key()

    def _init_files(self):
        """Khởi tạo các file"""
        if not os.path.exists(self.output_file):
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"Bắt đầu chuyển đổi: {datetime.now()}\n")
            f.write(f"Số lượng API keys: {len(self.api_keys)}\n")
            f.write(f"{'='*50}\n")

    def _get_last_processed_id(self) -> int:
        """Lấy ID lớn nhất đã xử lý"""
        try:
            if os.path.exists(self.output_file):
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if data:
                    max_id = max(item['_id'] for item in data)
                    self._log(f"ID lớn nhất đã xử lý: {max_id}")
                    return max_id
        except Exception as e:
            self._log(f"Lỗi khi đọc last processed ID: {e}")
        return 0

    def _log(self, message: str, print_console: bool = True):
        """Ghi log"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
        
        if print_console:
            print(log_message)

    def _log_api_stats(self):
        """Ghi thống kê sử dụng API keys"""
        self._log("\n" + "="*60)
        self._log("THỐNG KÊ SỬ DỤNG API KEYS:")
        self._log("="*60)
        
        for idx, stats in self.api_stats.items():
            if stats['success'] > 0 or stats['errors'] > 0:
                self._log(f"API Key #{idx + 1} ({stats['key']}): "
                         f"Thành công: {stats['success']} | Lỗi: {stats['errors']}")
        
        self._log("="*60 + "\n")

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
4. Chuyển các tên riêng Trung Quốc sang tên Việt Nam phù hợp
5. Điều chỉnh ngữ cảnh, địa danh, đơn vị tiền tệ sang Việt Nam
6. Giữ nguyên cấu trúc role: "người gọi" và "người nghe"
7. Ngôn ngữ phải tự nhiên, gần gũi như người Việt nói chuyện thật

Đoạn hội thoại cần chuyển đổi:
{dialogue_text}

Hãy trả về kết quả dưới dạng JSON:
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
                self.api_stats[self.current_key_index]['success'] += 1
                
                return result['dialogue']
                
            except json.JSONDecodeError as e:
                self._log(f"Lỗi parse JSON (lần {attempt + 1}/{retry_count}): {e}")
                if attempt == retry_count - 1:
                    self.api_stats[self.current_key_index]['errors'] += 1
                    raise
                time.sleep(2)
                
            except Exception as e:
                error_msg = str(e).lower()
                
                if any(keyword in error_msg for keyword in ['quota', 'rate limit', 'api key', 'permission', 'invalid']):
                    self._log(f"✗ API key #{self.current_key_index + 1} gặp lỗi: {e}")
                    self.api_stats[self.current_key_index]['errors'] += 1
                    
                    if self._switch_api_key():
                        self._log(f"→ Thử lại với API key mới...")
                        return self._convert_dialogue(dialogue, retry_count)
                    else:
                        raise Exception("Đã hết tất cả API keys!")
                else:
                    self._log(f"Lỗi API (lần {attempt + 1}/{retry_count}): {e}")
                    if attempt == retry_count - 1:
                        self.api_stats[self.current_key_index]['errors'] += 1
                        raise
                    time.sleep(2)

    def _save_single_result(self, item: Dict):
        """Lưu kết quả từng sample"""
        with open(self.output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        data.append(item)
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def convert(self):
        """Chuyển đổi dữ liệu"""
        with open(self.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        last_processed_id = self._get_last_processed_id()
        data = [item for item in data if item['_id'] > last_processed_id]
        
        total = len(data)
        if total == 0:
            self._log("Không còn dữ liệu nào cần xử lý!")
            return
        
        self._log(f"Tiếp tục từ ID: {last_processed_id + 1}")
        self._log(f"Tổng số mẫu cần xử lý: {total}")
        
        success_count = 0
        error_count = 0
        
        for idx, item in enumerate(data, 1):
            item_id = item['_id']
            
            try:
                self._log(f"Đang xử lý [{idx}/{total}] - ID: {item_id} - API Key #{self.current_key_index + 1}")
                
                converted_dialogue = self._convert_dialogue(item['dialogue'])
                
                new_item = {
                    "_id": item_id,
                    "dialogue": converted_dialogue
                }
                
                self._save_single_result(new_item)
                success_count += 1
                
                self._log(f"✓ Hoàn thành ID: {item_id} [{success_count}/{total}] - API Key #{self.current_key_index + 1}")
                
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
        
        self._log_api_stats()


def main():
    INPUT_FILE = r"F:\Projetcs\data_scam\translate\tele28k_harmless_translate.json"
    OUTPUT_FILE = r"F:\Projetcs\data_scam\localization\tele28k_harmless.json"
    LOG_FILE = r"F:\Projetcs\data_scam\localization\conversion_log.txt"
    
    print("="*60)
    print("CHƯƠNG TRÌNH CHUYỂN ĐỔI NGỮ CẢNH TIẾNG VIỆT")
    print("HỖ TRỢ NHIỀU API KEYS")
    print("="*60)
    
    try:
        converter = VietnameseContextConverter(
            input_file=INPUT_FILE,
            output_file=OUTPUT_FILE,
            log_file=LOG_FILE
        )
        
        print(f"\n✓ Đã load {len(converter.api_keys)} API keys")
        print("\nBắt đầu chuyển đổi (tự động tiếp tục từ ID đã xử lý)...")
        
        converter.convert()
        
        print(f"\n{'='*60}")
        print(f"HOÀN TẤT!")
        print(f"Kết quả: {OUTPUT_FILE}")
        print(f"Log: {LOG_FILE}")
        print(f"{'='*60}")
        
    except ValueError as e:
        print(f"\n❌ LỖI: {e}")
        print("Hãy kiểm tra file .env và đảm bảo có GEMINI_API_KEYS")
        print("Format: GEMINI_API_KEYS=key1,key2,key3")
    except FileNotFoundError as e:
        print(f"\n❌ LỖI: Không tìm thấy file - {e}")
    except Exception as e:
        print(f"\n❌ LỖI: {e}")


if __name__ == "__main__":
    main()