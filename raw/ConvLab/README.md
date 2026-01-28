# DailyDialog Dataset (ConvLab/dailydialog)

## 1. Giới thiệu

**DailyDialog** là bộ dữ liệu hội thoại đa lượt (multi-turn dialogue) chất lượng cao, được xây dựng từ các cuộc hội thoại đời thường bằng tiếng Anh. Dataset được gán nhãn thủ công cho nhiều tác vụ NLP quan trọng như **dialogue act recognition**, **emotion recognition**, **dialogue modeling** và **dialogue generation**.

Dataset này thường được sử dụng trong nghiên cứu và huấn luyện các mô hình:

- Conversational AI
- Dialogue Understanding & Generation
- Emotion-aware Dialogue Systems
- Multi-turn NLU / NLG

---

## 2. Thông tin chung

## Quy mô (tổng quan)

- Train: 11,118 hội thoại
- Validation: 1,000 hội thoại
- Test: 1,000 hội thoại
- Tổng: 13,118 hội thoại (khoảng ~103k utterances)

## Cấu trúc dữ liệu (mẫu)

Mỗi phần tử trong `dialogues.json` đại diện cho một cuộc hội thoại, ví dụ:

```json
{
  "dataset": "dailydialog",
  "data_split": "train",
  "dialogue_id": "dailydialog-train-0",
  "original_id": "train-0",
  "domains": ["Attitude & Emotion"],
  "turns": [
    {
      "speaker": "user",
      "utterance": "Say, Jim, how about going for a few beers after dinner?",
      "utt_idx": 0,
      "dialogue_acts": {
        "binary": [{ "intent": "directive", "domain": "", "slot": "" }],
        "categorical": [],
        "non-categorical": []
      },
      "emotion": "no emotion"
    }
  ]
}
```

Mỗi `turn` thường có các trường chính: `speaker`, `utterance`, `utt_idx`, `dialogue_acts`, `emotion`.

## Nhãn (annotation)

- Dialogue acts (intent): ví dụ `inform`, `question`, `directive`, `commissive`, ...
- Emotion: ví dụ `no emotion`, `happiness`, `sadness`, `anger`, `surprise`, ...

Những nhãn này thích hợp cho bài toán phân loại intent, nhận diện cảm xúc, hoặc làm nguồn huấn luyện cho các mô hình dialogue-aware.

## Nguồn & Liên kết

- Bài báo gốc / nguồn: IJCNLP 2017 — DailyDialog
- Hugging Face: <https://huggingface.co/datasets/ConvLab/dailydialog>
