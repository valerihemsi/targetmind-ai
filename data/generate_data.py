"""
Oyun şirketi için sahte müşteri verisi üretici.
Kasıtlı: eksik değerler, format tutarsızlıkları, demografik dengesizlikler.
"""
import random
import pandas as pd
import numpy as np
from faker import Faker

fake = Faker("tr_TR")
random.seed(42)
np.random.seed(42)

CITIES = [
    "İstanbul", "Ankara", "İzmir", "Bursa", "Antalya",
    "Adana", "Konya", "Gaziantep", "Mersin", "Kayseri",
]
GENDERS = ["Erkek", "Kadın", "Diğer"]          # kasıtlı: bazıları "E", "K", "male" olacak
DEVICES = ["Mobil", "PC", "Konsol", "Tablet"]
PLATFORMS = {
    "Mobil":  ["iOS", "Android"],
    "PC":     ["Steam", "Epic Games"],
    "Konsol": ["PlayStation", "Xbox"],
    "Tablet": ["iOS", "Android"],
}
GENRES = ["Strateji", "RPG", "Aksiyon", "Casual", "Spor", "Bulmaca"]
INCOME = ["Düşük", "Orta", "Yüksek"]
SUB_TYPE = ["Ücretsiz", "Premium"]


def random_gender():
    """Bazı satırlar tutarsız format döndürür (bias testi için)."""
    g = random.choices(GENDERS, weights=[0.62, 0.30, 0.08])[0]
    if random.random() < 0.12:
        g = {"Erkek": "E", "Kadın": "K", "Diğer": "male"}.get(g, g)
    return g


def maybe_null(val, prob=0.12):
    """Belirli olasılıkla None döndür (eksik veri simülasyonu)."""
    return None if random.random() < prob else val


def build_row(i: int) -> dict:
    device = random.choices(DEVICES, weights=[0.55, 0.28, 0.12, 0.05])[0]
    platform = random.choice(PLATFORMS[device])
    age = int(np.clip(np.random.normal(27, 10), 10, 70))
    income = random.choices(INCOME, weights=[0.40, 0.42, 0.18])[0]

    # Gelire göre harcama dağılımı
    base_spend = {"Düşük": 30, "Orta": 120, "Yüksek": 380}[income]
    spend = round(max(0, np.random.normal(base_spend, base_spend * 0.5)), 2)

    # Oyun saatine göre satın alma eğilimi
    gaming_h = round(max(0, np.random.normal(12, 8)), 1)
    in_app = "Evet" if (gaming_h > 8 and spend > 50 and random.random() > 0.3) else "Hayır"

    session = int(max(0, np.random.normal(18, 10)))
    campaign_click = round(random.uniform(0, 1), 2)
    referrals = int(max(0, np.random.poisson(1.2)))
    last_active = int(max(0, np.random.exponential(30)))
    completion = round(random.uniform(0.1, 1.0), 2)

    genre = random.choices(
        GENRES,
        weights=[0.22, 0.20, 0.20, 0.15, 0.13, 0.10]
    )[0]

    return {
        "musteri_id":           f"C{i:04d}",
        "yas":                  maybe_null(age, 0.05),
        "cinsiyet":             maybe_null(random_gender(), 0.08),
        "sehir":                maybe_null(random.choice(CITIES), 0.06),
        "gelir_seviyesi":       maybe_null(income, 0.10),
        "cihaz_turu":           maybe_null(device, 0.04),
        "platform":             maybe_null(platform, 0.04),
        "haftalik_oyun_saati":  maybe_null(gaming_h, 0.09),
        "tercih_edilen_tur":    maybe_null(genre, 0.07),
        "uygulama_ici_satin_alma": maybe_null(in_app, 0.08),
        "aylik_ortalama_harcama":  maybe_null(spend, 0.11),
        "son_aktivite_gun":     maybe_null(last_active, 0.07),
        "abonelik_turu":        maybe_null(random.choice(SUB_TYPE), 0.06),
        "kampanya_tiklanma_orani": maybe_null(campaign_click, 0.10),
        "arkadaslardan_referans":  maybe_null(referrals, 0.09),
        "aylik_oturum_sayisi":  maybe_null(session, 0.08),
        "tamamlanma_orani":     maybe_null(completion, 0.07),
    }


def inject_issues(df: pd.DataFrame) -> pd.DataFrame:
    """Kasıtlı kalite sorunları ekle."""
    # Yaş aykırı değerler
    df.loc[5,  "yas"] = 7
    df.loc[11, "yas"] = 142
    df.loc[78, "yas"] = 4

    # Negatif harcama
    df.loc[22, "aylik_ortalama_harcama"] = -50

    # Yinelenen satırlar (3 çift)
    dupes = df.sample(3, random_state=1)
    df = pd.concat([df, dupes], ignore_index=True)

    # Harcama formatı tutarsızlığı — object sütununa dönüştürerek yaz
    df["aylik_ortalama_harcama"] = df["aylik_ortalama_harcama"].astype(object)
    df.loc[33, "aylik_ortalama_harcama"] = "yüksek"

    return df


if __name__ == "__main__":
    rows = [build_row(i + 1) for i in range(300)]
    df = pd.DataFrame(rows)
    df = inject_issues(df)
    df = df.sample(frac=1, random_state=99).reset_index(drop=True)

    path = "data/customers.csv"
    df.to_csv(path, index=False)
    print(f"✓ {len(df)} satır oluşturuldu → {path}")
    print(f"\nSütunlar: {list(df.columns)}")
    print(f"\nEksik değer sayıları:\n{df.isnull().sum().to_string()}")
    print(f"\nİlk 5 satır:\n{df.head().to_string()}")
