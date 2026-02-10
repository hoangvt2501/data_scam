import json
from collections import Counter

def analyze(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    total = len(data)
    label_counts = Counter()
    missing_label = 0
    
    for item in data:
        lbl = item.get('label')
        if lbl is None:
            missing_label += 1
        else:
            label_counts[lbl] += 1

    print(f"Total samples: {total}")
    print(f"Missing label: {missing_label}")
    print("Label distribution:")
    for lbl, count in sorted(label_counts.items(), key=lambda x: str(x[0])):
        print(f"  Label {lbl}: {count}")

if __name__ == "__main__":
    analyze(r'c:\Users\admin\Desktop\Hoangvt\data_scam\localization\tele28k_scam.json')
