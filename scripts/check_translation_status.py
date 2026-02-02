"""
Script to analyze and display translation status of records
Shows which records are translated and which are still in Chinese
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict


# Configuration
INPUT_ORIGINAL_PATH = Path(r"C:\Users\admin\Desktop\Hoangvt\data_scam\processed\tele28k_scam_translate.json")
INPUT_VI_PATH = Path(r"C:\Users\admin\Desktop\Hoangvt\data_scam\processed\tele28k_scam_translate_complete.json")
OUTPUT_REPORT_PATH = Path(r"C:\Users\admin\Desktop\Hoangvt\data_scam\translation_status_report.txt")
OUTPUT_JSON_PATH = Path(r"C:\Users\admin\Desktop\Hoangvt\data_scam\translation_status.json")


def is_chinese(text: str) -> bool:
    """Check if text contains Chinese characters"""
    if not text:
        return False
    chinese_count = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
    # If more than 20% is Chinese, consider it Chinese
    return chinese_count > len(text) * 0.2


def is_vietnamese(text: str) -> bool:
    """Check if text contains Vietnamese characters with diacritics"""
    if not text:
        return False
    vietnamese_chars = set('àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịĩỉòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ')
    vietnamese_chars.update('ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊĨỈÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ')
    
    vietnamese_count = sum(1 for char in text if char in vietnamese_chars)
    # If has Vietnamese diacritics, consider it Vietnamese
    return vietnamese_count > 0


def check_translation_status(original_record: Dict, vi_record: Dict) -> Dict[str, Any]:
    """
    Check if a record is translated or not
    Returns detailed status information
    """
    status = {
        'dialogue_translated': False,
        'thinking_translated': False,
        'result_translated': False,
        'fully_translated': False,
        'partially_translated': False,
        'not_translated': False
    }
    
    # Check dialogue
    if original_record.get("dialogue") and vi_record.get("dialogue"):
        # Compare first dialogue turn
        orig_content = original_record["dialogue"][0]["content"] if original_record["dialogue"] else ""
        vi_content = vi_record["dialogue"][0]["content"] if vi_record["dialogue"] else ""
        
        if orig_content != vi_content and is_vietnamese(vi_content):
            status['dialogue_translated'] = True
    
    # Check thinking
    orig_thinking = original_record.get("thinking", "")
    vi_thinking = vi_record.get("thinking", "")
    if orig_thinking != vi_thinking and is_vietnamese(vi_thinking):
        status['thinking_translated'] = True
    
    # Check result
    orig_result = original_record.get("result", "")
    vi_result = vi_record.get("result", "")
    if orig_result != vi_result and is_vietnamese(vi_result):
        status['result_translated'] = True
    
    # Determine overall status
    translated_count = sum([
        status['dialogue_translated'],
        status['thinking_translated'],
        status['result_translated']
    ])
    
    if translated_count == 3:
        status['fully_translated'] = True
    elif translated_count > 0:
        status['partially_translated'] = True
    else:
        status['not_translated'] = True
    
    return status


def analyze_translation_status(original_path: Path, vi_path: Path) -> Dict[str, Any]:
    """
    Analyze all records and return detailed status
    """
    print("📂 Đang đọc dữ liệu...")
    
    # Load data
    with open(original_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    with open(vi_path, 'r', encoding='utf-8') as f:
        vi_data = json.load(f)
    
    print(f"✅ Đã load {len(original_data)} records gốc")
    print(f"✅ Đã load {len(vi_data)} records tiếng Việt")
    print()
    
    # Analyze each record
    analysis = {
        'total': len(original_data),
        'fully_translated': [],
        'partially_translated': [],
        'not_translated': [],
        'records_detail': []
    }
    
    print("🔍 Đang phân tích từng record...")
    for i, (orig, vi) in enumerate(zip(original_data, vi_data)):
        record_id = orig.get('_id', i + 1)
        status = check_translation_status(orig, vi)
        
        record_info = {
            'id': record_id,
            'status': status,
            'label': orig.get('label', 0),
            'dialogue_turns': len(orig.get('dialogue', []))
        }
        
        analysis['records_detail'].append(record_info)
        
        if status['fully_translated']:
            analysis['fully_translated'].append(record_id)
        elif status['partially_translated']:
            analysis['partially_translated'].append(record_id)
        else:
            analysis['not_translated'].append(record_id)
    
    return analysis


def generate_report(analysis: Dict[str, Any], output_path: Path):
    """
    Generate human-readable report
    """
    report_lines = []
    
    # Header
    report_lines.append("=" * 80)
    report_lines.append("BÁO CÁO TÌNH TRẠNG DỊCH DATASET".center(80))
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Summary statistics
    report_lines.append("📊 TỔNG QUAN:")
    report_lines.append("-" * 80)
    report_lines.append(f"Tổng số records:              {analysis['total']}")
    report_lines.append(f"✅ Đã dịch đầy đủ:             {len(analysis['fully_translated'])} records")
    report_lines.append(f"⚠️  Dịch một phần:              {len(analysis['partially_translated'])} records")
    report_lines.append(f"❌ Chưa dịch:                  {len(analysis['not_translated'])} records")
    report_lines.append("")
    
    # Percentage
    total = analysis['total']
    if total > 0:
        fully_pct = len(analysis['fully_translated']) / total * 100
        partial_pct = len(analysis['partially_translated']) / total * 100
        not_pct = len(analysis['not_translated']) / total * 100
        
        report_lines.append("📈 TỶ LỆ:")
        report_lines.append("-" * 80)
        report_lines.append(f"Đã dịch đầy đủ:  {fully_pct:.2f}%")
        report_lines.append(f"Dịch một phần:   {partial_pct:.2f}%")
        report_lines.append(f"Chưa dịch:       {not_pct:.2f}%")
        report_lines.append("")
    
    # List of fully translated records
    report_lines.append("=" * 80)
    report_lines.append(f"✅ DANH SÁCH {len(analysis['fully_translated'])} RECORDS ĐÃ DỊCH ĐẦY ĐỦ:")
    report_lines.append("=" * 80)
    
    if analysis['fully_translated']:
        # Group into ranges for readability
        ranges = []
        start = analysis['fully_translated'][0]
        end = start
        
        for id in analysis['fully_translated'][1:]:
            if id == end + 1:
                end = id
            else:
                if start == end:
                    ranges.append(f"{start}")
                else:
                    ranges.append(f"{start}-{end}")
                start = id
                end = id
        
        # Add last range
        if start == end:
            ranges.append(f"{start}")
        else:
            ranges.append(f"{start}-{end}")
        
        # Format in columns
        for i in range(0, len(ranges), 10):
            report_lines.append("  " + ", ".join(ranges[i:i+10]))
    else:
        report_lines.append("  (Không có)")
    
    report_lines.append("")
    
    # List of not translated records
    report_lines.append("=" * 80)
    report_lines.append(f"❌ DANH SÁCH {len(analysis['not_translated'])} RECORDS CHƯA DỊCH:")
    report_lines.append("=" * 80)
    
    if analysis['not_translated']:
        # Show detailed list
        for id in analysis['not_translated']:
            record_detail = next((r for r in analysis['records_detail'] if r['id'] == id), None)
            if record_detail:
                report_lines.append(f"  ID {id:4d} | Label: {record_detail['label']} | "
                                  f"Số câu hội thoại: {record_detail['dialogue_turns']}")
    else:
        report_lines.append("  (Không có)")
    
    report_lines.append("")
    
    # List of partially translated records
    if analysis['partially_translated']:
        report_lines.append("=" * 80)
        report_lines.append(f"⚠️  DANH SÁCH {len(analysis['partially_translated'])} RECORDS DỊCH MỘT PHẦN:")
        report_lines.append("=" * 80)
        
        for id in analysis['partially_translated']:
            record_detail = next((r for r in analysis['records_detail'] if r['id'] == id), None)
            if record_detail:
                status = record_detail['status']
                parts = []
                if status['dialogue_translated']:
                    parts.append("dialogue✅")
                else:
                    parts.append("dialogue❌")
                
                if status['thinking_translated']:
                    parts.append("thinking✅")
                else:
                    parts.append("thinking❌")
                
                if status['result_translated']:
                    parts.append("result✅")
                else:
                    parts.append("result❌")
                
                report_lines.append(f"  ID {id:4d} | {' | '.join(parts)}")
        
        report_lines.append("")
    
    # Footer
    report_lines.append("=" * 80)
    report_lines.append("KẾT THÚC BÁO CÁO".center(80))
    report_lines.append("=" * 80)
    
    # Write to file
    report_text = "\n".join(report_lines)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    # Also print to console
    print(report_text)
    
    return report_text


def save_json_status(analysis: Dict[str, Any], output_path: Path):
    """
    Save analysis results as JSON for programmatic access
    """
    json_data = {
        'summary': {
            'total': analysis['total'],
            'fully_translated': len(analysis['fully_translated']),
            'partially_translated': len(analysis['partially_translated']),
            'not_translated': len(analysis['not_translated'])
        },
        'fully_translated_ids': analysis['fully_translated'],
        'partially_translated_ids': analysis['partially_translated'],
        'not_translated_ids': analysis['not_translated'],
        'records_detail': analysis['records_detail']
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Đã lưu JSON status vào: {output_path}")


def main():
    """Main execution function"""
    print("=" * 80)
    print("🔍 PHÂN TÍCH TÌNH TRẠNG DỊCH DATASET".center(80))
    print("=" * 80)
    print()
    
    # Check if files exist
    if not INPUT_ORIGINAL_PATH.exists():
        print(f"❌ File gốc không tồn tại: {INPUT_ORIGINAL_PATH}")
        return
    
    if not INPUT_VI_PATH.exists():
        print(f"❌ File tiếng Việt không tồn tại: {INPUT_VI_PATH}")
        return
    
    # Analyze
    analysis = analyze_translation_status(INPUT_ORIGINAL_PATH, INPUT_VI_PATH)
    
    # Generate report
    print("\n📝 Đang tạo báo cáo...")
    generate_report(analysis, OUTPUT_REPORT_PATH)
    
    # Save JSON
    save_json_status(analysis, OUTPUT_JSON_PATH)
    
    print(f"\n✅ Đã lưu báo cáo text vào: {OUTPUT_REPORT_PATH}")
    print("\n✅ HOÀN THÀNH!")


if __name__ == "__main__":
    main()