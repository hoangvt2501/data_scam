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
    
    def __init__(self, input_file: str, output_file: str, log_file: str, model_name: str = None):

        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError("Không tìm thấy OPENROUTER_API_KEY trong file .env")
        
        self.api_key = api_key
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Sử dụng model được chỉ định hoặc default
        self.model = model_name or "google/gemini-flash-1.5-8b-exp-0827"
        
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
        """Tạo prompt cho AI"""
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
                # Headers đầy đủ cho OpenRouter
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
                    ]
                }
                
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                
                # Log chi tiết lỗi nếu có
                if response.status_code != 200:
                    self._log(f"HTTP {response.status_code}: {response.text}")
                    response.raise_for_status()
                
                result_data = response.json()
                result_text = result_data['choices'][0]['message']['content'].strip()
                
                # Loại bỏ markdown code block nếu có
                if result_text.startswith('```'):
                    result_text = result_text.split('```')[1]
                    if result_text.startswith('json'):
                        result_text = result_text[4:]
                    result_text = result_text.strip()
                
                result = json.loads(result_text)
                return result['dialogue']
                
            except json.JSONDecodeError as e:
                self._log(f"Lỗi parse JSON (lần {attempt + 1}/{retry_count}): {e}")
                self._log(f"Response text: {result_text[:200]}...")
                if attempt == retry_count - 1:
                    raise
                time.sleep(2)
                
            except requests.exceptions.RequestException as e:
                self._log(f"Lỗi HTTP (lần {attempt + 1}/{retry_count}): {e}")
                if attempt == retry_count - 1:
                    raise
                time.sleep(3)
                
            except Exception as e:
                self._log(f"Lỗi khác (lần {attempt + 1}/{retry_count}): {e}")
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
                
                time.sleep(1.5)
                
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


def list_available_models():
    """Liệt kê các model có sẵn trên OpenRouter"""
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("❌ Không tìm thấy OPENROUTER_API_KEY")
        return []
    
    try:
        print("\n🔍 Đang tải danh sách models từ OpenRouter...")
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {api_key}"}
        )
        
        if response.status_code == 200:
            models = response.json()['data']
            
            print("\n" + "="*80)
            print("CÁC MODEL KHẢ DỤNG TRÊN OPENROUTER")
            print("="*80)
            
            # Lọc models miễn phí và Gemini
            free_models = []
            gemini_models = []
            
            for model in models:
                model_id = model['id']
                model_name = model.get('name', model_id)
                pricing = model.get('pricing', {})
                
                # Kiểm tra miễn phí
                is_free = (
                    pricing.get('prompt') == '0' or 
                    ':free' in model_id.lower()
                )
                
                if is_free:
                    free_models.append({
                        'id': model_id,
                        'name': model_name
                    })
                
                if 'gemini' in model_id.lower():
                    gemini_models.append({
                        'id': model_id,
                        'name': model_name,
                        'free': is_free
                    })
            
            # Hiển thị models miễn phí
            print("\n📌 MODELS MIỄN PHÍ (FREE):")
            print("-" * 80)
            for idx, model in enumerate(free_models[:10], 1):
                print(f"{idx}. {model['id']}")
                print(f"   {model['name']}")
            
            # Hiển thị Gemini models
            print("\n📌 TẤT CẢ GEMINI MODELS:")
            print("-" * 80)
            for idx, model in enumerate(gemini_models, 1):
                free_tag = "✅ FREE" if model['free'] else "💰 PAID"
                print(f"{idx}. {free_tag} - {model['id']}")
            
            print("\n" + "="*80)
            
            return free_models
            
        else:
            print(f"❌ Lỗi {response.status_code}: {response.text}")
            return []
            
    except Exception as e:
        print(f"❌ Lỗi khi lấy danh sách models: {e}")
        return []


def main():
    # Cấu hình
    INPUT_FILE = r"F:\Projetcs\data_scam\translate\tele28k_harmless_translate.json"
    OUTPUT_FILE = r"F:\Projetcs\data_scam\localization\tele28k_harmless.json"
    LOG_FILE = r"F:\Projetcs\data_scam\conversion1_log.txt"
    
    print("="*60)
    print("CHƯƠNG TRÌNH CHUYỂN ĐỔI NGỮ CẢNH TIẾNG VIỆT")
    print("="*60)
    
    try:
        # Tùy chọn xem models
        view_models = input("\nBạn có muốn xem danh sách models? (y/n): ").strip().lower()
        
        if view_models == 'y':
            list_available_models()
        
        # Nhập tên model
        print("\n📝 Một số model phổ biến:")
        print("1. google/gemini-flash-1.5-8b-exp-0827 (FREE)")
        print("2. google/gemini-pro-1.5-exp (FREE thử nghiệm)")
        print("3. meta-llama/llama-3.1-8b-instruct:free (FREE)")
        print("4. Nhập tên model khác")
        
        choice = input("\nNhập lựa chọn (1-4, Enter = 1): ").strip()
        
        if choice == "2":
            model_name = "google/gemini-pro-1.5-exp"
        elif choice == "3":
            model_name = "meta-llama/llama-3.1-8b-instruct:free"
        elif choice == "4":
            model_name = input("Nhập tên model: ").strip()
        else:
            model_name = "google/gemini-3-flash-preview"
        
        # Khởi tạo converter
        converter = VietnameseContextConverter(
            input_file=INPUT_FILE,
            output_file=OUTPUT_FILE,
            log_file=LOG_FILE,
            model_name=model_name
        )
        
        print("\nChọn cách xử lý:")
        print("1. Tự động tiếp tục từ ID cuối cùng (khuyến nghị)")
        print("2. Chạy lại từ đầu")
        print("3. Chọn ID range tùy chỉnh")
        
        choice = input("\nNhập lựa chọn (1/2/3): ").strip()
        
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