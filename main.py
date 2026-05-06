"""
Bias-Aware Multi-Agent Data Optimization System
Çalıştırma: python main.py
"""
import os
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
    # 1. Veriyi oluştur
    print("\n" + "="*60)
    print("ADIM 0 — Örnek müşteri verisi oluşturuluyor...")
    print("="*60)
    from data.generate_data import build_row, inject_issues
    import pandas as pd
    rows = [build_row(i + 1) for i in range(300)]
    df = pd.DataFrame(rows)
    df = inject_issues(df)
    df = df.sample(frac=1, random_state=99).reset_index(drop=True)
    df.to_csv("data/customers.csv", index=False)
    print(f"✓ {len(df)} satır oluşturuldu → data/customers.csv")

    # 2. Crew'u başlat
    print("\n" + "="*60)
    print("ADIM 1-7 — 7 ajan devreye giriyor...")
    print("="*60 + "\n")

    from crew import build_crew
    crew = build_crew()
    result = crew.kickoff()

    # 3. Özet
    print("\n" + "="*60)
    print("SİSTEM TAMAMLANDI")
    print("="*60)

    output_files = [
        ("data/customers.csv",         "Ham veri"),
        ("data/customers_cleaned.csv", "Temizlenmiş veri"),
        ("data/customers_scored.csv",  "Skorlanmış veri"),
        ("data/final_targets.csv",     "Final hedef kitle"),
    ]
    print("\nOluşturulan dosyalar:")
    for path, label in output_files:
        if Path(path).exists():
            import pandas as pd
            n = len(pd.read_csv(path))
            print(f"  ✓ {path:35s} → {n:4d} satır  ({label})")

    print(f"\nFinal sonuç:\n{result}")


if __name__ == "__main__":
    main()
