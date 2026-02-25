# tele28k_report.py
# -*- coding: utf-8 -*-

import json
import re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# CONFIG
# =========================================================
INPUT_PATH = r"F:\Projetcs\data_scam\localization\tele28k_scam.json"
OUTPUT_DIR = r"F:\Projetcs\data_scam\localization\report_outputs"

# Từ khóa phân tích (có thể chỉnh thêm)
SCAM_KEYWORDS = [
    "hoàn tiền", "lỗi kỹ thuật", "thu hồi", "xác minh", "đường link", "link",
    "website", "trang web", "hotline", "tài khoản", "số tài khoản",
    "ngân hàng", "otp", "mã otp", "chuyển tiền", "chuyển khoản",
    "phí", "bộ phận kế toán", "khẩn", "gấp", "miễn phí", "an toàn tuyệt đối"
]


# =========================================================
# UTILS
# =========================================================
def safe_text(x):
    return "" if x is None else str(x)


def normalize_text(s: str) -> str:
    s = safe_text(s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_fig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def print_section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# =========================================================
# LOAD + FLATTEN
# =========================================================
def load_data(input_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def flatten_data(data):
    sample_rows = []
    turn_rows = []

    for item in data:
        _id = item.get("_id")
        label = item.get("label")
        label_name = item.get("label_name")
        dialogue = item.get("dialogue", [])

        sample_rows.append({
            "_id": _id,
            "label": label,
            "label_name": label_name,
            "num_turns": len(dialogue),
        })

        for turn_idx, turn in enumerate(dialogue):
            role = safe_text(turn.get("role"))
            content = safe_text(turn.get("content"))

            turn_rows.append({
                "_id": _id,
                "label": label,
                "label_name": label_name,
                "turn_idx": turn_idx,
                "role": role,
                "content": content,
                "char_len": len(content),
                "word_len": len(content.split()),
            })

    df_samples = pd.DataFrame(sample_rows)
    df_turns = pd.DataFrame(turn_rows)

    # dialogue-level length
    if not df_turns.empty:
        dialog_char = df_turns.groupby("_id")["char_len"].sum().reset_index(name="dialogue_char_len")
        dialog_word = df_turns.groupby("_id")["word_len"].sum().reset_index(name="dialogue_word_len")
        df_samples = df_samples.merge(dialog_char, on="_id", how="left")
        df_samples = df_samples.merge(dialog_word, on="_id", how="left")
        df_samples["dialogue_char_len"] = df_samples["dialogue_char_len"].fillna(0).astype(int)
        df_samples["dialogue_word_len"] = df_samples["dialogue_word_len"].fillna(0).astype(int)

    return df_samples, df_turns


# =========================================================
# ANALYSIS
# =========================================================
def dataset_overview(df_samples: pd.DataFrame, df_turns: pd.DataFrame):
    stats = {
        "total_samples": int(len(df_samples)),
        "total_turns": int(len(df_turns)),
        "unique_labels": int(df_samples["label"].nunique()) if not df_samples.empty else 0,
        "avg_turns_per_dialogue": float(df_samples["num_turns"].mean()) if not df_samples.empty else 0.0,
        "median_turns_per_dialogue": float(df_samples["num_turns"].median()) if not df_samples.empty else 0.0,
        "avg_turn_char_len": float(df_turns["char_len"].mean()) if not df_turns.empty else 0.0,
        "avg_turn_word_len": float(df_turns["word_len"].mean()) if not df_turns.empty else 0.0,
        "avg_dialogue_char_len": float(df_samples["dialogue_char_len"].mean()) if "dialogue_char_len" in df_samples else 0.0,
        "avg_dialogue_word_len": float(df_samples["dialogue_word_len"].mean()) if "dialogue_word_len" in df_samples else 0.0,
    }
    return pd.DataFrame([stats])


def label_distribution(df_samples: pd.DataFrame):
    if df_samples.empty:
        return pd.DataFrame(columns=["label", "label_name", "count", "ratio"])

    out = (
        df_samples.groupby(["label", "label_name"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    out["ratio"] = out["count"] / out["count"].sum()
    return out


def role_statistics(df_turns: pd.DataFrame):
    if df_turns.empty:
        return pd.DataFrame()

    role_counts = df_turns["role"].value_counts().rename_axis("role").reset_index(name="turn_count")
    role_counts["turn_ratio"] = role_counts["turn_count"] / role_counts["turn_count"].sum()

    role_len = (
        df_turns.groupby("role")
        .agg(
            avg_char_len=("char_len", "mean"),
            median_char_len=("char_len", "median"),
            avg_word_len=("word_len", "mean"),
            median_word_len=("word_len", "median"),
        )
        .reset_index()
    )

    return role_counts.merge(role_len, on="role", how="left")


def conversation_structure(df_samples: pd.DataFrame, raw_data: list):
    rows = []
    for item in raw_data:
        _id = item.get("_id")
        label = item.get("label")
        label_name = item.get("label_name")
        dialogue = item.get("dialogue", [])
        roles = [safe_text(t.get("role")) for t in dialogue]

        alternating_ratio = np.nan
        if len(roles) > 1:
            alternating_ratio = sum(1 for i in range(1, len(roles)) if roles[i] != roles[i-1]) / (len(roles) - 1)

        rows.append({
            "_id": _id,
            "label": label,
            "label_name": label_name,
            "num_turns": len(roles),
            "first_role": roles[0] if roles else None,
            "last_role": roles[-1] if roles else None,
            "alternating_ratio": alternating_ratio,
        })

    return pd.DataFrame(rows)


def keyword_analysis(df_turns: pd.DataFrame, keywords: list):
    rows = []
    if df_turns.empty:
        return pd.DataFrame(columns=["keyword", "turn_count", "sample_count"])

    content_lower = df_turns["content"].astype(str).str.lower()

    for kw in keywords:
        mask = content_lower.str.contains(re.escape(kw.lower()), na=False)
        rows.append({
            "keyword": kw,
            "turn_count": int(mask.sum()),
            "sample_count": int(df_turns.loc[mask, "_id"].nunique())
        })

    df_kw = pd.DataFrame(rows).sort_values(["sample_count", "turn_count"], ascending=False)
    return df_kw


def keyword_by_role(df_turns: pd.DataFrame, keywords: list):
    rows = []
    if df_turns.empty:
        return pd.DataFrame(columns=["keyword", "role", "count"])

    content_lower = df_turns["content"].astype(str).str.lower()

    for kw in keywords:
        mask = content_lower.str.contains(re.escape(kw.lower()), na=False)
        tmp = df_turns.loc[mask].groupby("role").size().reset_index(name="count")
        for _, r in tmp.iterrows():
            rows.append({
                "keyword": kw,
                "role": r["role"],
                "count": int(r["count"])
            })

    return pd.DataFrame(rows)


def signal_position_analysis(df_turns: pd.DataFrame, df_samples: pd.DataFrame):
    # Regex tổng hợp tín hiệu lừa đảo
    pattern = r"(hoàn tiền|lỗi kỹ thuật|link|đường link|tài khoản|otp|chuyển khoản|xác minh|website|trang web)"
    if df_turns.empty:
        return pd.DataFrame()

    mask = df_turns["content"].astype(str).str.lower().str.contains(pattern, regex=True, na=False)
    df_pos = df_turns.loc[mask].copy()
    if df_pos.empty:
        return df_pos

    turn_count_map = df_samples.set_index("_id")["num_turns"].to_dict()
    df_pos["dialogue_num_turns"] = df_pos["_id"].map(turn_count_map)
    df_pos["turn_pos_norm"] = df_pos["turn_idx"] / df_pos["dialogue_num_turns"].replace(0, np.nan)
    return df_pos


def duplicate_dialogues_exact(raw_data: list):
    rows = []

    def dialogue_signature(dialogue):
        parts = []
        for t in dialogue:
            role = normalize_text(t.get("role"))
            content = normalize_text(t.get("content"))
            parts.append(f"{role}:{content}")
        return " ||| ".join(parts)

    for item in raw_data:
        rows.append({
            "_id": item.get("_id"),
            "label": item.get("label"),
            "label_name": item.get("label_name"),
            "dialogue_signature": dialogue_signature(item.get("dialogue", []))
        })

    df_sig = pd.DataFrame(rows)
    dup_mask = df_sig["dialogue_signature"].duplicated(keep=False)
    dup_df = df_sig.loc[dup_mask].sort_values("dialogue_signature").copy()
    return dup_df


# =========================================================
# PLOTS
# =========================================================
def plot_label_distribution(label_df: pd.DataFrame, out_dir: Path, topn: int = 20):
    if label_df.empty:
        return

    top = label_df.head(topn).copy()
    plt.figure(figsize=(13, 6))
    plt.bar(top["label_name"].astype(str), top["count"])
    plt.xticks(rotation=75, ha="right")
    plt.title(f"Top {len(top)} nhãn theo số mẫu")
    plt.xlabel("label_name")
    plt.ylabel("Số mẫu")
    save_fig(out_dir / "01_label_distribution_top.png")


def plot_num_turns(df_samples: pd.DataFrame, out_dir: Path):
    if df_samples.empty:
        return
    plt.figure(figsize=(10, 5))
    plt.hist(df_samples["num_turns"], bins=40)
    plt.title("Phân bố số lượt thoại mỗi hội thoại")
    plt.xlabel("num_turns")
    plt.ylabel("Tần suất")
    save_fig(out_dir / "02_num_turns_hist.png")


def plot_turn_char_len(df_turns: pd.DataFrame, out_dir: Path):
    if df_turns.empty:
        return
    plt.figure(figsize=(10, 5))
    plt.hist(df_turns["char_len"], bins=50)
    plt.title("Phân bố độ dài lượt thoại (ký tự)")
    plt.xlabel("char_len")
    plt.ylabel("Tần suất")
    save_fig(out_dir / "03_turn_char_len_hist.png")


def plot_dialogue_char_len(df_samples: pd.DataFrame, out_dir: Path):
    if df_samples.empty or "dialogue_char_len" not in df_samples.columns:
        return
    plt.figure(figsize=(10, 5))
    plt.hist(df_samples["dialogue_char_len"], bins=50)
    plt.title("Phân bố độ dài hội thoại (tổng ký tự)")
    plt.xlabel("dialogue_char_len")
    plt.ylabel("Tần suất")
    save_fig(out_dir / "04_dialogue_char_len_hist.png")


def plot_keyword_coverage(df_kw: pd.DataFrame, out_dir: Path, topn: int = 20):
    if df_kw.empty:
        return

    top = df_kw.head(topn).copy()
    plt.figure(figsize=(13, 6))
    plt.bar(top["keyword"], top["sample_count"])
    plt.xticks(rotation=75, ha="right")
    plt.title(f"Top {len(top)} từ khóa lừa đảo theo số hội thoại chứa từ khóa")
    plt.xlabel("keyword")
    plt.ylabel("Số hội thoại")
    save_fig(out_dir / "05_keyword_coverage.png")


def plot_signal_position(df_pos: pd.DataFrame, out_dir: Path):
    if df_pos.empty or "turn_pos_norm" not in df_pos.columns:
        return
    plt.figure(figsize=(10, 5))
    plt.hist(df_pos["turn_pos_norm"].dropna(), bins=20)
    plt.title("Vị trí xuất hiện tín hiệu lừa đảo trong hội thoại (chuẩn hóa)")
    plt.xlabel("turn_pos_norm (0 = đầu hội thoại, 1 = cuối hội thoại)")
    plt.ylabel("Tần suất")
    save_fig(out_dir / "06_signal_position_hist.png")


# =========================================================
# REPORT EXPORT
# =========================================================
def export_csvs(
    out_dir: Path,
    overview_df: pd.DataFrame,
    df_samples: pd.DataFrame,
    df_turns: pd.DataFrame,
    label_df: pd.DataFrame,
    role_df: pd.DataFrame,
    struct_df: pd.DataFrame,
    kw_df: pd.DataFrame,
    kw_role_df: pd.DataFrame,
    df_pos: pd.DataFrame,
    dup_df: pd.DataFrame,
):
    overview_df.to_csv(out_dir / "overview_summary.csv", index=False, encoding="utf-8-sig")
    df_samples.to_csv(out_dir / "samples_flat.csv", index=False, encoding="utf-8-sig")
    df_turns.to_csv(out_dir / "turns_flat.csv", index=False, encoding="utf-8-sig")
    label_df.to_csv(out_dir / "label_distribution.csv", index=False, encoding="utf-8-sig")
    role_df.to_csv(out_dir / "role_statistics.csv", index=False, encoding="utf-8-sig")
    struct_df.to_csv(out_dir / "conversation_structure.csv", index=False, encoding="utf-8-sig")
    kw_df.to_csv(out_dir / "keyword_stats.csv", index=False, encoding="utf-8-sig")
    kw_role_df.to_csv(out_dir / "keyword_by_role.csv", index=False, encoding="utf-8-sig")
    if not df_pos.empty:
        df_pos.to_csv(out_dir / "signal_positions.csv", index=False, encoding="utf-8-sig")
    if not dup_df.empty:
        dup_df.to_csv(out_dir / "duplicate_dialogues_exact.csv", index=False, encoding="utf-8-sig")


def print_console_summary(
    overview_df: pd.DataFrame,
    label_df: pd.DataFrame,
    role_df: pd.DataFrame,
    struct_df: pd.DataFrame,
    kw_df: pd.DataFrame,
    dup_df: pd.DataFrame,
):
    print_section("TONG QUAN DATASET")
    print(overview_df.to_string(index=False))

    print_section("TOP NHAN")
    print(label_df.head(15).to_string(index=False))

    print_section("THONG KE ROLE")
    if not role_df.empty:
        print(role_df.to_string(index=False))
    else:
        print("Khong co du lieu role")

    print_section("CAU TRUC HOI THOAI")
    if not struct_df.empty:
        print("First role distribution:")
        print(struct_df["first_role"].value_counts(dropna=False).to_string())
        print("\nLast role distribution:")
        print(struct_df["last_role"].value_counts(dropna=False).to_string())
        print("\nAlternating ratio stats:")
        print(struct_df["alternating_ratio"].describe().to_string())
    else:
        print("Khong co du lieu cau truc")

    print_section("TOP TU KHOA LUA DAO")
    if not kw_df.empty:
        print(kw_df.head(20).to_string(index=False))
    else:
        print("Khong co du lieu tu khoa")

    print_section("TRUNG LAP HOI THOAI (EXACT)")
    print(f"So dong trung lap (tinh ca ban sao lap): {len(dup_df)}")
    if not dup_df.empty:
        print(dup_df.head(10).to_string(index=False))


# =========================================================
# MAIN
# =========================================================
def main():
    ensure_dir(OUTPUT_DIR)
    out_dir = Path(OUTPUT_DIR)

    print_section("LOAD DATA")
    data = load_data(INPUT_PATH)
    print(f"Loaded {len(data):,} samples from: {INPUT_PATH}")

    print_section("FLATTEN DATA")
    df_samples, df_turns = flatten_data(data)
    print("df_samples:", df_samples.shape)
    print("df_turns:", df_turns.shape)

    print_section("RUN ANALYSIS")
    overview_df = dataset_overview(df_samples, df_turns)
    label_df = label_distribution(df_samples)
    role_df = role_statistics(df_turns)
    struct_df = conversation_structure(df_samples, data)
    kw_df = keyword_analysis(df_turns, SCAM_KEYWORDS)
    kw_role_df = keyword_by_role(df_turns, SCAM_KEYWORDS)
    df_pos = signal_position_analysis(df_turns, df_samples)
    dup_df = duplicate_dialogues_exact(data)

    # Plot
    print_section("SAVE PLOTS")
    plot_label_distribution(label_df, out_dir, topn=20)
    plot_num_turns(df_samples, out_dir)
    plot_turn_char_len(df_turns, out_dir)
    plot_dialogue_char_len(df_samples, out_dir)
    plot_keyword_coverage(kw_df, out_dir, topn=20)
    plot_signal_position(df_pos, out_dir)
    print(f"Saved plots to: {out_dir}")

    # Export CSV
    print_section("EXPORT CSV")
    export_csvs(
        out_dir=out_dir,
        overview_df=overview_df,
        df_samples=df_samples,
        df_turns=df_turns,
        label_df=label_df,
        role_df=role_df,
        struct_df=struct_df,
        kw_df=kw_df,
        kw_role_df=kw_role_df,
        df_pos=df_pos,
        dup_df=dup_df,
    )
    print(f"Saved CSVs to: {out_dir}")

    # Console summary
    print_console_summary(overview_df, label_df, role_df, struct_df, kw_df, dup_df)

    print_section("DONE")
    print("Hoan tat phan tich bao cao dataset tele28k_scam.json")


if __name__ == "__main__":
    main()