"""
Script to translate missing Vietnamese records using multiple translation models
and select the best quality translation.

Models:
1. Helsinki-NLP/opus-mt-zh-vi
2. erax-ai/EraX-Translator-V1.0
3. chi-vi/hirashiba-mt-tiny-zh-vi
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# Configuration
INPUT_ORIGINAL_PATH = Path(r"C:\Users\admin\Desktop\Hoangvt\data_scam\raw\TeleAntiFraud-28k\harmless.json")
INPUT_VI_PATH = Path(r"C:\Users\admin\Desktop\Hoangvt\data_scam\processed\tele28k_harmless_translate.json")
OUTPUT_VI_PATH = Path(r"C:\Users\admin\Desktop\Hoangvt\data_scam\processed\tele28k_harmless_translate_complete.json")

# Translation models to use
MODELS = [
    "Helsinki-NLP/opus-mt-zh-vi",
    "erax-ai/EraX-Translator-V1.0",
    "chi-vi/hirashiba-mt-tiny-zh-vi"
]


class TranslationEvaluator:
    """Evaluate translation quality based on multiple criteria"""
    
    @staticmethod
    def calculate_length_ratio(source: str, translation: str) -> float:
        """
        Calculate length ratio between source and translation.
        Closer to 1.0-1.5 is better for Chinese to Vietnamese.
        """
        if not source or not translation:
            return 0.0
        ratio = len(translation) / len(source)
        # Penalize too short or too long translations
        if ratio < 0.8 or ratio > 2.5:
            return 0.0
        return 1.0 - abs(ratio - 1.3) / 1.3
    
    @staticmethod
    def check_special_chars(translation: str) -> float:
        """Check if translation has too many special characters (indicates poor quality)"""
        if not translation:
            return 0.0
        special_count = sum(1 for c in translation if not c.isalnum() and c not in ' ,.:;!?-""''()[]{}')
        ratio = special_count / len(translation)
        return 1.0 - min(ratio * 5, 1.0)  # Penalize high special char ratio
    
    @staticmethod
    def check_repetition(translation: str) -> float:
        """Check for excessive repetition"""
        if not translation or len(translation) < 10:
            return 0.0
        words = translation.split()
        if len(words) < 2:
            return 0.5
        unique_ratio = len(set(words)) / len(words)
        return unique_ratio
    
    @staticmethod
    def check_vietnamese_chars(translation: str) -> float:
        """Check if translation contains Vietnamese characters"""
        vietnamese_chars = set('àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịĩỉòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ')
        vietnamese_chars.update('ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊĨỈÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ')
        
        if not translation:
            return 0.0
        
        vietnamese_count = sum(1 for c in translation if c in vietnamese_chars)
        # At least 5% of characters should be Vietnamese-specific
        return min(vietnamese_count / max(len(translation) * 0.05, 1), 1.0)
    
    @classmethod
    def evaluate(cls, source: str, translation: str) -> float:
        """
        Evaluate overall translation quality (0-1 score)
        """
        if not translation or not translation.strip():
            return 0.0
        
        scores = {
            'length_ratio': cls.calculate_length_ratio(source, translation),
            'special_chars': cls.check_special_chars(translation),
            'repetition': cls.check_repetition(translation),
            'vietnamese_chars': cls.check_vietnamese_chars(translation)
        }
        
        # Weighted average
        weights = {
            'length_ratio': 0.2,
            'special_chars': 0.2,
            'repetition': 0.3,
            'vietnamese_chars': 0.3
        }
        
        total_score = sum(scores[k] * weights[k] for k in scores)
        return total_score


class MultiModelTranslator:
    """Translate using multiple models and select the best result"""
    
    def __init__(self, model_names: List[str]):
        self.models = {}
        self.tokenizers = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Using device: {self.device}")
        
        for model_name in model_names:
            try:
                print(f"📥 Loading model: {model_name}...")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                model.to(self.device)
                model.eval()
                
                self.models[model_name] = model
                self.tokenizers[model_name] = tokenizer
                print(f"✅ Successfully loaded: {model_name}")
            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")
                continue
        
        if not self.models:
            raise ValueError("No models loaded successfully!")
        
        print(f"\n✅ Successfully loaded {len(self.models)} models\n")
    
    def translate_with_model(self, text: str, model_name: str) -> str:
        """Translate text using a specific model"""
        try:
            tokenizer = self.tokenizers[model_name]
            model = self.models[model_name]
            
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate translation
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3
                )
            
            # Decode
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translation.strip()
            
        except Exception as e:
            print(f"  ⚠️  Error translating with {model_name}: {e}")
            return ""
    
    def translate_best(self, text: str) -> Dict[str, Any]:
        """
        Translate using all models and return the best translation
        """
        if not text or not text.strip():
            return {"translation": "", "model": None, "score": 0.0}
        
        translations = {}
        
        # Get translations from all models
        for model_name in self.models:
            translation = self.translate_with_model(text, model_name)
            if translation:
                score = TranslationEvaluator.evaluate(text, translation)
                translations[model_name] = {
                    "translation": translation,
                    "score": score
                }
        
        if not translations:
            return {"translation": text, "model": None, "score": 0.0}
        
        # Select best translation
        best_model = max(translations.items(), key=lambda x: x[1]["score"])
        
        return {
            "translation": best_model[1]["translation"],
            "model": best_model[0].split('/')[-1],  # Short name
            "score": best_model[1]["score"]
        }


def translate_missing_records(
    original_path: Path,
    vi_path: Path,
    output_path: Path,
    translator: MultiModelTranslator
):
    """
    Translate missing Vietnamese records
    Save after each record and log detailed progress
    """
    print("📂 Loading data...")
    
    # Load original Chinese data
    with open(original_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    # Load Vietnamese data
    with open(vi_path, 'r', encoding='utf-8') as f:
        vi_data = json.load(f)
    
    print(f"✅ Loaded {len(original_data)} original records")
    print(f"✅ Loaded {len(vi_data)} Vietnamese records")
    
    # Find records that need translation
    needs_translation = []
    for i, (orig, vi) in enumerate(zip(original_data, vi_data)):
        # Check if dialogue is the same (not translated)
        if orig["dialogue"] == vi["dialogue"]:
            needs_translation.append(i)
    
    print(f"\n🔍 Found {len(needs_translation)} records that need translation")
    print(f"📝 Record IDs to translate: {[original_data[i]['_id'] for i in needs_translation]}")
    
    if not needs_translation:
        print("✅ All records already translated!")
        return
    
    # Create log file
    log_path = output_path.parent / "translation_log.txt"
    with open(log_path, 'w', encoding='utf-8') as log_file:
        log_file.write("=" * 80 + "\n")
        log_file.write("TRANSLATION LOG\n")
        log_file.write("=" * 80 + "\n\n")
    
    # Translate missing records
    print("\n🚀 Starting translation...\n")
    
    translation_stats = {
        "total": len(needs_translation),
        "success": 0,
        "failed": 0,
        "model_usage": {},
        "translated_ids": [],
        "failed_ids": []
    }
    
    for idx in needs_translation:
        record_id = original_data[idx]['_id']
        orig_record = original_data[idx]
        
        print(f"\n{'='*60}")
        print(f"🔄 Translating Record ID: {record_id} (Index: {idx})")
        print(f"{'='*60}")
        
        try:
            # Translate dialogue
            print(f"  📝 Translating dialogue ({len(orig_record['dialogue'])} turns)...")
            translated_dialogue = []
            dialogue_models = []
            
            for turn_idx, turn in enumerate(orig_record["dialogue"]):
                result = translator.translate_best(turn["content"])
                translated_dialogue.append({
                    "role": turn["role"],
                    "content": result["translation"]
                })
                
                # Track model usage
                if result["model"]:
                    translation_stats["model_usage"][result["model"]] = \
                        translation_stats["model_usage"].get(result["model"], 0) + 1
                    dialogue_models.append(result["model"])
                
                print(f"    Turn {turn_idx + 1}: {result['model']} (score: {result['score']:.3f})")
            
            # Translate thinking
            print(f"  🧠 Translating thinking...")
            thinking_result = translator.translate_best(orig_record["thinking"])
            print(f"    Model: {thinking_result['model']} (score: {thinking_result['score']:.3f})")
            
            # Translate result
            print(f"  📊 Translating result...")
            result_result = translator.translate_best(orig_record["result"])
            print(f"    Model: {result_result['model']} (score: {result_result['score']:.3f})")
            
            # Update Vietnamese record
            vi_data[idx]["dialogue"] = translated_dialogue
            vi_data[idx]["thinking"] = thinking_result["translation"]
            vi_data[idx]["result"] = result_result["translation"]
            
            # Save immediately after translating this record
            print(f"  💾 Saving progress...")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(vi_data, f, ensure_ascii=False, indent=2)
            
            translation_stats["success"] += 1
            translation_stats["translated_ids"].append(record_id)
            
            # Log to file
            with open(log_path, 'a', encoding='utf-8') as log_file:
                log_file.write(f"✅ Record ID {record_id} (Index {idx}) - SUCCESS\n")
                log_file.write(f"   Dialogue models: {', '.join(dialogue_models)}\n")
                log_file.write(f"   Thinking model: {thinking_result['model']}\n")
                log_file.write(f"   Result model: {result_result['model']}\n")
                log_file.write(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            print(f"  ✅ Record ID {record_id} translated successfully!")
            print(f"  📊 Progress: {translation_stats['success']}/{translation_stats['total']}")
            
        except Exception as e:
            print(f"  ❌ Error translating record ID {record_id}: {e}")
            translation_stats["failed"] += 1
            translation_stats["failed_ids"].append(record_id)
            
            # Log error to file
            with open(log_path, 'a', encoding='utf-8') as log_file:
                log_file.write(f"❌ Record ID {record_id} (Index {idx}) - FAILED\n")
                log_file.write(f"   Error: {str(e)}\n")
                log_file.write(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            continue
    
    # Save final data
    print(f"\n💾 Saving final data to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vi_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Saved {len(vi_data)} records")
    
    # Print final statistics
    print("\n" + "="*80)
    print("📈 TRANSLATION STATISTICS:")
    print("="*80)
    print(f"Total records to translate:  {translation_stats['total']}")
    print(f"Successfully translated:      {translation_stats['success']}")
    print(f"Failed:                       {translation_stats['failed']}")
    print(f"\n✅ Translated Record IDs: {translation_stats['translated_ids']}")
    if translation_stats['failed_ids']:
        print(f"❌ Failed Record IDs:     {translation_stats['failed_ids']}")
    print("\n📊 Model usage statistics:")
    for model, count in sorted(translation_stats["model_usage"].items(), key=lambda x: -x[1]):
        percentage = (count / sum(translation_stats["model_usage"].values())) * 100
        print(f"  {model:30s}: {count:4d} times ({percentage:5.1f}%)")
    print("="*80)
    print(f"\n📝 Detailed log saved to: {log_path}")
    
    # Save statistics to JSON
    stats_path = output_path.parent / "translation_statistics.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(translation_stats, f, ensure_ascii=False, indent=2)
    print(f"📊 Statistics saved to: {stats_path}")


def main():
    """Main execution function"""
    print("="*60)
    print("🌐 MULTI-MODEL TRANSLATION SYSTEM")
    print("="*60)
    print()
    
    # Check if files exist
    if not INPUT_ORIGINAL_PATH.exists():
        print(f"❌ Original file not found: {INPUT_ORIGINAL_PATH}")
        return
    
    if not INPUT_VI_PATH.exists():
        print(f"❌ Vietnamese file not found: {INPUT_VI_PATH}")
        return
    
    # Initialize translator
    try:
        translator = MultiModelTranslator(MODELS)
    except Exception as e:
        print(f"❌ Failed to initialize translator: {e}")
        return
    
    # Translate missing records
    translate_missing_records(
        INPUT_ORIGINAL_PATH,
        INPUT_VI_PATH,
        OUTPUT_VI_PATH,
        translator
    )
    
    print("\n✅ TRANSLATION COMPLETED!")


if __name__ == "__main__":
    main()