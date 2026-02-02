import json
from pathlib import Path
from typing import List, Dict, Any

# Configuration
INPUT_PATH = Path(r"C:\Users\admin\Desktop\Hoangvt\data_scam\raw\TeleAntiFraud-28k\TeleAntiFraud-28k.harmless.json")
OUTPUT_ORIGIN_PATH = Path(r"C:\Users\admin\Desktop\Hoangvt\data_scam\raw\TeleAntiFraud-28k\harmless.json")
OUTPUT_VI_PATH = Path(r"C:\Users\admin\Desktop\Hoangvt\data_scam\processed\tele28k_harmless_translate.json")


def process_single_record(record: Dict[str, Any], record_id: int) -> tuple[Dict[str, Any], Dict[str, Any], bool]:
    """
    Process a single record and return both original and translated versions
    
    Args:
        record: Input record with original and vi_translation data
        record_id: Sequential ID for this record
        
    Returns:
        Tuple of (original_data, vietnamese_data, has_translation)
    """
    # Create original Chinese version
    original_data = {
        "_id": record_id,
        "dialogue": record.get("dialogue", []),
        "thinking": record.get("thinking", ""),
        "result": record.get("result", ""),
        "label": record.get("label", 0)
    }
    
    # Create Vietnamese translation version
    # Check if vi_translation exists and has required fields
    if "vi_translation" in record and record["vi_translation"].get("status") == "done":
        vi_trans = record["vi_translation"]
        vietnamese_data = {
            "_id": record_id,
            "dialogue": vi_trans.get("dialogue", []),
            "thinking": vi_trans.get("thinking", ""),
            "result": vi_trans.get("result", ""),
            "label": record.get("label", 0)
        }
        has_translation = True
    else:
        # If no translation, keep same ID but with original Chinese data
        # This maintains ID alignment between files
        vietnamese_data = {
            "_id": record_id,
            "dialogue": record.get("dialogue", []),  # Keep original Chinese
            "thinking": record.get("thinking", ""),
            "result": record.get("result", ""),
            "label": record.get("label", 0)
        }
        has_translation = False
    
    return original_data, vietnamese_data, has_translation


def process_dataset(input_path: Path, output_origin_path: Path, output_vi_path: Path):
    """
    Process the entire dataset and create two separate JSON files
    
    Args:
        input_path: Path to input JSON file
        output_origin_path: Path to output original (Chinese) JSON file
        output_vi_path: Path to output Vietnamese translation JSON file
    """
    print(f"📂 Đang đọc dữ liệu từ: {input_path}")
    
    # Read input file
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Lỗi: Không tìm thấy file {input_path}")
        return
    except json.JSONDecodeError as e:
        print(f"❌ Lỗi: File JSON không hợp lệ - {e}")
        return
    
    # Ensure data is a list
    if isinstance(data, dict):
        data = [data]
    
    print(f"📊 Tổng số records: {len(data)}")
    
    # Process all records
    original_records = []
    vietnamese_records = []
    translated_count = 0
    not_translated_count = 0
    error_count = 0
    
    for idx, record in enumerate(data, start=1):
        try:
            original, vietnamese, has_translation = process_single_record(record, idx)
            
            # Always add both records to maintain ID alignment
            original_records.append(original)
            vietnamese_records.append(vietnamese)
            
            if has_translation:
                translated_count += 1
            else:
                not_translated_count += 1
                if not_translated_count <= 10:  # Only show first 10 warnings
                    print(f"⚠️  Record {idx}: Chưa có bản dịch - giữ nguyên ID và dữ liệu gốc")
                
        except Exception as e:
            error_count += 1
            if error_count <= 10:  # Only show first 10 errors
                print(f"❌ Lỗi xử lý record {idx}: {str(e)}")
            continue
    
    # Show summary if there are more errors
    if error_count > 10:
        print(f"... và {error_count - 10} lỗi khác")
    
    # Create output directories if they don't exist
    output_origin_path.parent.mkdir(parents=True, exist_ok=True)
    output_vi_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save original (Chinese) data
    print(f"\n💾 Đang lưu dữ liệu gốc (Tiếng Trung)...")
    with open(output_origin_path, 'w', encoding='utf-8') as f:
        json.dump(original_records, f, ensure_ascii=False, indent=2)
    print(f"✅ Đã lưu {len(original_records)} records vào: {output_origin_path}")
    
    # Save Vietnamese translation data
    print(f"\n💾 Đang lưu dữ liệu dịch (Tiếng Việt)...")
    with open(output_vi_path, 'w', encoding='utf-8') as f:
        json.dump(vietnamese_records, f, ensure_ascii=False, indent=2)
    print(f"✅ Đã lưu {len(vietnamese_records)} records vào: {output_vi_path}")
    
    # Summary
    print("\n" + "="*60)
    print("📈 TỔNG KẾT:")
    print("="*60)
    print(f"📥 Tổng số records đầu vào:        {len(data)}")
    print(f"✅ Records xử lý thành công:       {len(original_records)}")
    print(f"❌ Records bị lỗi (bỏ qua):        {error_count}")
    print(f"📤 Records gốc (Tiếng Trung):      {len(original_records)}")
    print(f"📤 Records file Tiếng Việt:        {len(vietnamese_records)}")
    print(f"✅ Records đã dịch sang Tiếng Việt: {translated_count}")
    print(f"⚠️  Records chưa dịch (giữ gốc):    {not_translated_count}")
    print("="*60)
    print(f"💡 Lưu ý: Cả 2 file đều có {len(original_records)} records với ID tương ứng")
    if error_count > 0:
        print(f"⚠️  Cảnh báo: {error_count} records bị lỗi và không được lưu")
    print("="*60)
    
    # Sample validation
    if original_records and vietnamese_records:
        print("\n🔍 KIỂM TRA MẪU (Record đầu tiên):")
        print(f"   ID gốc:  {original_records[0]['_id']}")
        print(f"   ID dịch: {vietnamese_records[0]['_id']}")
        print(f"   Label:   {original_records[0]['label']}")
        print(f"   Số câu hội thoại gốc:  {len(original_records[0]['dialogue'])}")
        print(f"   Số câu hội thoại dịch: {len(vietnamese_records[0]['dialogue'])}")


def main():
    """Main execution function"""
    print("🚀 BẮT ĐẦU XỬ LÝ DỮ LIỆU")
    print("="*60)
    
    # Validate input file exists
    if not INPUT_PATH.exists():
        print(f"❌ Lỗi: File input không tồn tại: {INPUT_PATH}")
        print("👉 Vui lòng kiểm tra đường dẫn và thử lại!")
        return
    
    # Process the dataset
    process_dataset(INPUT_PATH, OUTPUT_ORIGIN_PATH, OUTPUT_VI_PATH)
    
    print("\n✅ HOÀN THÀNH!")


if __name__ == "__main__":
    main()