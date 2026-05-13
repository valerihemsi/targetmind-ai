"""
TargetMind AI — CLI giriş noktası (LLM-driven crew yolu)
Çalıştırma: python main.py

`.env` içinde ANTHROPIC_API_KEY tanımlı olmalı.
"""
import os
import json
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

# .env yükle
_env = Path(__file__).parent / ".env"
if _env.exists():
    with open(_env) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main():
    Path("data").mkdir(exist_ok=True)

    # 1. Demo veriyi oluştur
    print("\n" + "=" * 60)
    print("ADIM 0 — Örnek müşteri verisi oluşturuluyor...")
    print("=" * 60)
    from data.generate_data import build_row, inject_issues
    import pandas as pd
    rows = [build_row(i + 1) for i in range(300)]
    df = pd.DataFrame(rows)
    df = inject_issues(df)
    df = df.sample(frac=1, random_state=99).reset_index(drop=True)
    df.to_csv("data/customers.csv", index=False)
    print(f"✓ {len(df)} satır oluşturuldu → data/customers.csv")

    # 2. column_mapping.json yaz (tool'ların okuduğu metadata)
    Path("data/column_mapping.json").write_text(json.dumps({
        "id_col": "musteri_id",
        "demographic_cols": ["cinsiyet", "gelir_seviyesi", "yas"],
        "metric_cols": [
            "haftalik_oyun_saati",
            "aylik_ortalama_harcama",
            "aylik_oturum_sayisi",
            "kampanya_tiklanma_orani",
            "arkadaslardan_referans",
        ],
        "segment_col": "tercih_edilen_tur",
    }, ensure_ascii=False, indent=2))

    # 3. Pipeline log'unu sıfırla (her çalıştırma temiz başlasın)
    Path("data/pipeline_log.json").write_text("{}")

    # 4. LLM-driven crew'u başlat
    print("\n" + "=" * 60)
    print("ADIM 1-7 — 7 LLM-driven ajan devreye giriyor...")
    print("=" * 60 + "\n")

    from crew import build_crew
    crew = build_crew()
    result = crew.kickoff()

    # 5. Özet
    print("\n" + "=" * 60)
    print("PIPELINE TAMAMLANDI")
    print("=" * 60)

    output_files = [
        ("data/customers.csv",            "Ham veri"),
        ("data/customers_cleaned.csv",    "Temizlenmiş veri"),
        ("data/customers_scored.csv",     "Skorlanmış veri (düzeltilmiş)"),
        ("data/final_targets.csv",        "Final hedef kitle"),
        ("data/optimal_targets.csv",      "Optimal 20"),
        ("data/pipeline_log.json",        "Ajan log'u"),
    ]
    print("\nOluşturulan dosyalar:")
    for path, label in output_files:
        if Path(path).exists():
            if path.endswith(".csv"):
                n = len(pd.read_csv(path))
                print(f"  ✓ {path:35s} → {n:4d} satır  ({label})")
            else:
                size = Path(path).stat().st_size
                print(f"  ✓ {path:35s} → {size:>5d} B    ({label})")

    print(f"\nFinal ajan çıktısı:\n{result}")


if __name__ == "__main__":
    main()
