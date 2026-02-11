# convert_vietnamese_context_llama.py

import json
import os
from datetime import datetime
from typing import List, Dict
import time
from dotenv import load_dotenv
import requests

# Load biến môi trường từ file .env
load_dotenv()

class VietnameseContextConverter:
    
    def __init__(self, input_file: str, output_file: str, log_file: str):
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError("Không tìm thấy OPENROUTER_API_KEY trong file .env")
        
        self.api_key = api_key
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Sử dụng Llama 3.3 70B Instruct FREE
        # self.model = "google/gemma-3-4b-it:free"
        # self.model = "openrouter/pony-alpha"
        # self.model = "nvidia/nemotron-3-nano-30b-a3b:free"
        self.model = "stepfun/step-3.5-flash:nitro"
        
        self.input_file = input_file
        self.output_file = output_file
        self.log_file = log_file

        self._init_files()
        self._log(f"🤖 Sử dụng model: {self.model}")
    
    def _init_files(self):
        """Khởi tạo các file"""
        if not os.path.exists(self.output_file):
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"Bắt đầu chuyển đổi: {datetime.now()}\n")
            f.write(f"Model: {self.model}\n")
            f.write(f"{'='*50}\n")
    
    def _log(self, message: str, print_console: bool = True):
        """Ghi log"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
        
        if print_console:
            print(log_message)
    
    def _get_last_processed_id(self) -> int:
        """Đọc ID cuối cùng đã xử lý từ output file"""
        try:
            if os.path.exists(self.output_file):
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data:
                        last_id = max([item['_id'] for item in data])
                        self._log(f"ID cuối cùng đã xử lý: {last_id}")
                        return last_id
        except Exception as e:
            self._log(f"Lỗi khi đọc file output: {e}")
        
        return None

    def _create_prompt(self, dialogue: List[Dict]) -> str:
        """Tạo prompt cho Llama 3.3"""
        dialogue_text = "\n".join([
            f"{turn['role']}: {turn['content']}" 
            for turn in dialogue
        ])
        
        prompt = f"""Bạn là chuyên gia bản địa hóa tiếng Việt. Đoạn hội thoại dưới đây ĐÃ ĐƯỢC DỊCH sang tiếng Việt nhưng còn mang dấu ấn Trung Quốc. Hãy BẢN ĐỊA HÓA cho tự nhiên như người Việt nói chuyện thật.

    NHIỆM VỤ:

    1. XƯNG HÔ - Chọn 1 cặp và giữ THỐNG NHẤT xuyên suốt:
    • Công việc/chính thức: anh/em, chị/em
    • Bạn bè thân: mày/tao, cậu/tớ, bạn  
    • Người lạ lịch sự: anh/chị, bạn
    ⚠️ TUYỆT ĐỐI không đổi xưng hô giữa chừng!

    2. NGÔN NGỮ TỰ NHIÊN như người Việt nói thật:
    ✓ Thêm: ừ, à, ơ, nhé, nha, hả, vậy à, thế à
    ✓ Cảm thán: ối, trời ơi, ủa, hả
    ✓ Câu ngắn, súc tích
    ✗ Tránh văn viết cứng nhắc

    3. BẢN ĐỊA HÓA - Chuyển ngữ cảnh Việt Nam:
    
    TÊN NGƯỜI:
    • Tiểu Lý, Lý Na → Linh, Mai, Hương
    • Vương Minh, Trương Vỹ → Minh, Tuấn, Dũng
    • A Cường, Tiểu Hồng → Cường, Hồng, Lan
    
    ĐỊA DANH:
    • Bắc Kinh → Hà Nội
    • Thượng Hải → TP.HCM  
    • Quảng Châu → Đà Nẵng
    • Thâm Quyến → Hải Phòng
    
    NGÂN HÀNG/CÔNG TY:
    • Ngân hàng Công Thương TQ → Vietcombank
    • Ngân hàng Nông nghiệp → Agribank
    • Alipay/WeChat Pay → MoMo/ZaloPay
    • Taobao/Tmall → Shopee/Lazada
    
    TIỀN TỆ (nhân ~3500):
    • 100 nhân dân tệ → 350,000đ
    • 1000 tệ → 3,500,000đ
    • 5000 tệ → 17,500,000đ
    
    SỐ ĐIỆN THOẠI:
    • Format Việt: 09xx-xxx-xxx hoặc 03xx-xxx-xxx

    4. GIỮ NGUYÊN:
    ✓ Nội dung và ý nghĩa chính xác
    ✓ Role: "người gọi" và "người nghe"  
    ✓ Cảm xúc, giọng điệu

    VÍ DỤ:

    Input (chưa localize):
    người gọi: Xin chào, tôi là Tiểu Lý từ Ngân hàng Công Thương Trung Quốc
    người nghe: Xin chào, anh cần gì?

    Output (đã localize):
    {{
    "dialogue": [
        {{"role": "người gọi", "content": "Alo, chào anh. Em là Linh, bên Vietcombank ạ"}},
        {{"role": "người nghe", "content": "Ừ chào em, em cần gì?"}}
    ]
    }}

    Input:
    người gọi: Anh ơi, tài khoản anh có 500 nhân dân tệ bị đóng băng
    người nghe: Tại sao lại như vậy?

    Output:
    {{
    "dialogue": [
        {{"role": "người gọi", "content": "Anh ơi, tài khoản anh có 1,750,000đ bị phong tỏa"}},
        {{"role": "người nghe", "content": "Hả? Sao lại thế?"}}
    ]
    }}

    ĐOẠN HỘI THOẠI CẦN BẢN ĐỊA HÓA:
    {dialogue_text}

    TRẢ VỀ CHỈ JSON, KHÔNG GIẢI THÍCH:
    {{
    "dialogue": [
        {{"role": "người gọi", "content": "..."}},
        {{"role": "người nghe", "content": "..."}}
    ]
    }}"""

        return prompt
    
    def _convert_dialogue(self, dialogue: List[Dict], retry_count: int = 3) -> List[Dict]:
        """Chuyển đổi một đoạn hội thoại"""
        prompt = self._create_prompt(dialogue)
        
        for attempt in range(retry_count):
            try:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/vietnamese-converter",
                    "X-Title": "Vietnamese Context Converter"
                }
                
                payload = {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.3,  # Giảm nhiệt độ cho kết quả ổn định
                    "max_tokens": 5000   # Tăng token limit
                }
                
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=90  # Tăng timeout vì model lớn
                )
                
                if response.status_code != 200:
                    self._log(f"HTTP {response.status_code}: {response.text}")
                    response.raise_for_status()
                
                result_data = response.json()
                result_text = result_data['choices'][0]['message']['content'].strip()
                
                # Loại bỏ markdown code block
                if result_text.startswith('```'):
                    result_text = result_text.split('```')[1]
                    if result_text.startswith('json'):
                        result_text = result_text[4:]
                    result_text = result_text.strip()
                
                # Loại bỏ phần text thừa trước/sau JSON
                if '{' in result_text and '}' in result_text:
                    start = result_text.find('{')
                    end = result_text.rfind('}') + 1
                    result_text = result_text[start:end]
                
                result = json.loads(result_text)
                return result['dialogue']
                
            except json.JSONDecodeError as e:
                self._log(f"Lỗi parse JSON (lần {attempt + 1}/{retry_count}): {e}")
                self._log(f"Response text: {result_text[:300]}...")
                if attempt == retry_count - 1:
                    raise
                time.sleep(3)
                
            except requests.exceptions.RequestException as e:
                self._log(f"Lỗi HTTP (lần {attempt + 1}/{retry_count}): {e}")
                if attempt == retry_count - 1:
                    raise
                time.sleep(5)
                
            except Exception as e:
                self._log(f"Lỗi khác (lần {attempt + 1}/{retry_count}): {e}")
                if attempt == retry_count - 1:
                    raise
                time.sleep(3)
    
    def _save_single_result(self, item: Dict):
        """Lưu kết quả từng sample vào file"""
        with open(self.output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        data.append(item)
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def convert(self, start_id: int = None, end_id: int = None, auto_resume: bool = True):
        """
        Chuyển đổi dữ liệu
        
        Args:
            start_id: ID bắt đầu (None = từ đầu hoặc auto-resume)
            end_id: ID kết thúc (None = đến cuối)
            auto_resume: Tự động tiếp tục từ ID cuối cùng
        """
        with open(self.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Tự động resume từ ID cuối cùng đã xử lý
        if auto_resume and start_id is None:
            last_id = self._get_last_processed_id()
            if last_id is not None:
                start_id = last_id + 1
                self._log(f"🔄 Tự động tiếp tục từ ID: {start_id}")
        
        # Lọc theo ID nếu có
        if start_id is not None or end_id is not None:
            data = [
                item for item in data 
                if (start_id is None or item['_id'] >= start_id) and 
                   (end_id is None or item['_id'] <= end_id)
            ]
        
        total = len(data)
        
        if total == 0:
            self._log("✅ Không có dữ liệu mới cần xử lý!")
            return
        
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
                
                # Delay 2 giây để tránh rate limit (Llama 3.3 70B free có rate limit)
                time.sleep(2)
                
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
    INPUT_FILE = r"F:\Projetcs\data_scam\translate\tele28k_harmless_translate.json"
    OUTPUT_FILE = r"F:\Projetcs\data_scam\localization\tele28k_harmless_llama.json"
    LOG_FILE = r"F:\Projetcs\data_scam\conversion_llama_log.txt"
    
    print("="*60)
    print("CHƯƠNG TRÌNH CHUYỂN ĐỔI NGỮ CẢNH TIẾNG VIỆT")
    print("Model: Llama 3.3 70B Instruct (FREE)")
    print("="*60)
    
    try:
        # Khởi tạo converter
        converter = VietnameseContextConverter(
            input_file=INPUT_FILE,
            output_file=OUTPUT_FILE,
            log_file=LOG_FILE
        )
        
        print("\nChọn cách xử lý:")
        print("1. Tự động tiếp tục từ ID cuối cùng (khuyến nghị)")
        print("2. Chạy lại từ đầu")
        print("3. Chọn ID range tùy chỉnh")
        print("4. Test 3 mẫu đầu tiên")
        
        choice = input("\nNhập lựa chọn (1/2/3/4): ").strip()
        
        if choice == "1":
            converter.convert(auto_resume=True)
            
        elif choice == "2":
            confirm = input("⚠️  Bạn chắc chắn muốn chạy lại từ đầu? (y/n): ").strip().lower()
            if confirm == 'y':
                with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                    json.dump([], f, ensure_ascii=False)
                converter.convert(auto_resume=False)
            else:
                print("Đã hủy!")
                
        elif choice == "3":
            start_id = input("Nhập ID bắt đầu (Enter để bỏ qua): ").strip()
            end_id = input("Nhập ID kết thúc (Enter để bỏ qua): ").strip()
            
            start_id = int(start_id) if start_id else None
            end_id = int(end_id) if end_id else None
            
            converter.convert(start_id=start_id, end_id=end_id, auto_resume=False)
            
        elif choice == "4":
            print("\n🔍 TEST: Chạy thử 3 mẫu đầu tiên...")
            converter.convert(start_id=1, end_id=3, auto_resume=False)
        else:
            print("Lựa chọn không hợp lệ!")
        
        print(f"\n{'='*60}")
        print(f"HOÀN TẤT!")
        print(f"Kết quả: {OUTPUT_FILE}")
        print(f"Log: {LOG_FILE}")
        print(f"{'='*60}")
        
    except ValueError as e:
        print(f"\n❌ LỖI: {e}")
        print("Hãy kiểm tra file .env và đảm bảo có OPENROUTER_API_KEY")
    except FileNotFoundError as e:
        print(f"\n❌ LỖI: Không tìm thấy file - {e}")
    except Exception as e:
        print(f"\n❌ LỖI: {e}")


if __name__ == "__main__":
    main()