"""
Test Senaryo 3: Neredeyse Eşit İki Müşteri
-------------------------------------------
Müşteri A (Ayşe): Orta gelir, daha az oynuyor, daha çok harcıyor
Müşteri B (Barış): Düşük gelir, daha çok oynuyor, daha az harcıyor

Soru: Kim daha yüksek skor alır? Fark adil mi?
"""
import sys, json
import pandas as pd
import numpy as np

sys.path.insert(0, ".")

# ── Test profilleri ──────────────────────────────────────────────────────────

PROFILLER = [
    {
        "musteri_id":              "TEST_A",
        "isim":                    "Ayşe (Orta Gelir, Çok Harcıyor)",
        "yas":                     27,
        "cinsiyet":                "Kadın",
        "sehir":                   "İstanbul",
        "gelir_seviyesi":          "Orta",
        "cihaz_turu":              "Mobil",
        "platform":                "Android",
        "haftalik_oyun_saati":     14.0,
        "tercih_edilen_tur":       "Strateji",
        "uygulama_ici_satin_alma": "Evet",
        "aylik_ortalama_harcama":  210.0,
        "son_aktivite_gun":        8,
        "abonelik_turu":           "Premium",
        "kampanya_tiklanma_orani": 0.62,
        "arkadaslardan_referans":  2,
        "aylik_oturum_sayisi":     18,
        "tamamlanma_orani":        0.71,
    },
    {
        "musteri_id":              "TEST_B",
        "isim":                    "Barış (Düşük Gelir, Çok Oynuyor)",
        "yas":                     24,
        "cinsiyet":                "Erkek",
        "sehir":                   "Ankara",
        "gelir_seviyesi":          "Düşük",
        "cihaz_turu":              "Mobil",
        "platform":                "Android",
        "haftalik_oyun_saati":     31.0,
        "tercih_edilen_tur":       "Strateji",
        "uygulama_ici_satin_alma": "Evet",
        "aylik_ortalama_harcama":  55.0,
        "son_aktivite_gun":        3,
        "abonelik_turu":           "Premium",
        "kampanya_tiklanma_orani": 0.58,
        "arkadaslardan_referans":  3,
        "aylik_oturum_sayisi":     32,
        "tamamlanma_orani":        0.88,
    },
]


# ── Normalize + skor hesapla ─────────────────────────────────────────────────

def hesapla_skor(df_ref: pd.DataFrame, profil: dict, agirliklar: dict) -> dict:
    """
    Referans veri setinin min/max değerlerini kullanarak profili normalize et
    ve skoru bileşen bileşen hesapla.
    """
    W = agirliklar

    def norm(val, col):
        mn = df_ref[col].min()
        mx = df_ref[col].max()
        return (val - mn) / (mx - mn + 1e-9)

    tur_bonus_map = {
        "Strateji": 1.0, "RPG": 0.9, "Aksiyon": 0.6,
        "Casual": 0.3, "Spor": 0.3, "Bulmaca": 0.4
    }

    bileskenler = {
        "oyun_saati":  norm(profil["haftalik_oyun_saati"], "haftalik_oyun_saati") * W["oyun_saati"],
        "harcama":     norm(profil["aylik_ortalama_harcama"], "aylik_ortalama_harcama") * W["harcama"],
        "oturum":      norm(profil["aylik_oturum_sayisi"], "aylik_oturum_sayisi") * W["oturum"],
        "kampanya":    norm(profil["kampanya_tiklanma_orani"], "kampanya_tiklanma_orani") * W["kampanya"],
        "referans":    norm(profil["arkadaslardan_referans"], "arkadaslardan_referans") * W["referans"],
        "tur_bonus":   tur_bonus_map.get(profil["tercih_edilen_tur"], 0.3) * W["tur_bonus"],
    }
    toplam = sum(bileskenler.values()) * 100
    return {"bileskenler": bileskenler, "toplam_skor": round(toplam, 2)}


# ── Ana test ─────────────────────────────────────────────────────────────────

def main():
    scored_ref = pd.read_csv("data/customers_scored.csv")

    MEVCUT_AGIRLIKLAR = {
        "oyun_saati": 0.25, "harcama": 0.30, "oturum": 0.15,
        "kampanya": 0.10, "referans": 0.08, "tur_bonus": 0.12
    }
    DUZELTILMIS_AGIRLIKLAR = {
        "oyun_saati": 0.32, "harcama": 0.15, "oturum": 0.22,
        "kampanya": 0.12, "referans": 0.10, "tur_bonus": 0.09
    }

    senaryolar = [
        ("Mevcut model  (harcama %30)", MEVCUT_AGIRLIKLAR),
        ("Düzeltilmiş   (harcama %15)", DUZELTILMIS_AGIRLIKLAR),
    ]

    print("=" * 68)
    print("TEST SENARYO 3 — NEREDEYSE EŞİT İKİ MÜŞTERİ")
    print("=" * 68)

    # Profil karşılaştırması
    print("\n── PROFİL KARŞILAŞTIRMASI ───────────────────────────────────────")
    satirlar = [
        ("Yaş",                 "yas"),
        ("Cinsiyet",            "cinsiyet"),
        ("Gelir seviyesi",      "gelir_seviyesi"),
        ("Oyun türü",           "tercih_edilen_tur"),
        ("Platform",            "platform"),
        ("Haftalık oyun saati", "haftalik_oyun_saati"),
        ("Aylık harcama",       "aylik_ortalama_harcama"),
        ("Aylık oturum",        "aylik_oturum_sayisi"),
        ("Son aktivite (gün)",  "son_aktivite_gun"),
        ("Tamamlanma oranı",    "tamamlanma_orani"),
        ("Kampanya tıklanma",   "kampanya_tiklanma_orani"),
        ("Referans",            "arkadaslardan_referans"),
    ]
    a, b = PROFILLER[0], PROFILLER[1]
    print(f"{'Özellik':25s}  {'Ayşe (A)':>18}  {'Barış (B)':>18}  Üstün")
    print("-" * 68)
    for label, key in satirlar:
        va, vb = a[key], b[key]
        try:
            if float(va) > float(vb):
                ustun = "A ▲"
            elif float(vb) > float(va):
                ustun = "B ▲"
            else:
                ustun = "Eşit"
        except (ValueError, TypeError):
            ustun = "—"
        print(f"{label:25s}  {str(va):>18}  {str(vb):>18}  {ustun}")

    # Skor senaryoları
    for senaryo_adi, agirliklar in senaryolar:
        print(f"\n── SKOR SONUÇLARI: {senaryo_adi.upper()} ──────────────────────────")
        print(f"{'Bileşen':15s}  {'Ağırlık':>8}  {'Ayşe (A)':>10}  {'Barış (B)':>10}  Üstün")
        print("-" * 60)

        skor_a = hesapla_skor(scored_ref, a, agirliklar)
        skor_b = hesapla_skor(scored_ref, b, agirliklar)
        bil_a  = skor_a["bileskenler"]
        bil_b  = skor_b["bileskenler"]

        for bil_adi, agirlik in agirliklar.items():
            va = bil_a[bil_adi] * 100
            vb = bil_b[bil_adi] * 100
            ustun = "A" if va > vb else ("B" if vb > va else "=")
            print(f"{bil_adi:15s}  {agirlik:>8.2f}  {va:>10.2f}  {vb:>10.2f}  {ustun}")

        ta = skor_a["toplam_skor"]
        tb = skor_b["toplam_skor"]
        print("-" * 60)
        kazanan = "Ayşe (A)" if ta > tb else ("Barış (B)" if tb > ta else "Eşit")
        print(f"{'TOPLAM SKOR':15s}  {'':>8}  {ta:>10.2f}  {tb:>10.2f}  ← {kazanan} kazandı")
        print(f"Fark: {abs(ta - tb):.2f} puan")

    # Adillik değerlendirmesi
    print("\n── ADİLLİK DEĞERLENDİRMESİ ─────────────────────────────────────")

    s_mevcut_a = hesapla_skor(scored_ref, a, MEVCUT_AGIRLIKLAR)["toplam_skor"]
    s_mevcut_b = hesapla_skor(scored_ref, b, MEVCUT_AGIRLIKLAR)["toplam_skor"]
    s_duz_a    = hesapla_skor(scored_ref, a, DUZELTILMIS_AGIRLIKLAR)["toplam_skor"]
    s_duz_b    = hesapla_skor(scored_ref, b, DUZELTILMIS_AGIRLIKLAR)["toplam_skor"]

    print(f"\nMevcut model   → Ayşe: {s_mevcut_a:.1f} | Barış: {s_mevcut_b:.1f} | Fark: {abs(s_mevcut_a - s_mevcut_b):.1f} puan")
    print(f"Düzeltilmiş    → Ayşe: {s_duz_a:.1f}    | Barış: {s_duz_b:.1f}    | Fark: {abs(s_duz_a - s_duz_b):.1f} puan")

    print("""
Sorular:
  1. Barış haftada 31 saat oynuyor, Ayşe 14 saat.
     Oyun şirketi için hangisi daha değerli bir kullanıcı?

  2. Mevcut modelde Ayşe daha yüksek skor alıyor çünkü harcama ağırlığı %30.
     Ama harcama kapasitesi gelir seviyesiyle kısıtlı — bu Barış'ın hatası mı?

  3. Düzeltilmiş modelde skor farkı değişiyor.
     Bu değişim oyun şirketinin iş hedefleriyle uyumlu mu?
""")

    # Bonus: Barış'ı elenme riskinden kurtaran threshold
    print("── BARIŞIN ELENMESİNİ ÖNLEMEK İÇİN GEREKLİ MİNİMUM HARCAMA ────")
    for h_agirlik in [0.30, 0.20, 0.15, 0.10]:
        ag = {**MEVCUT_AGIRLIKLAR, "harcama": h_agirlik,
              "oyun_saati": MEVCUT_AGIRLIKLAR["oyun_saati"] + (0.30 - h_agirlik) * 0.5,
              "oturum": MEVCUT_AGIRLIKLAR["oturum"] + (0.30 - h_agirlik) * 0.5}
        sa = hesapla_skor(scored_ref, a, ag)["toplam_skor"]
        sb = hesapla_skor(scored_ref, b, ag)["toplam_skor"]
        durum = "Ayşe önde" if sa > sb else "Barış önde" if sb > sa else "Eşit"
        print(f"  Harcama ağırlığı %{int(h_agirlik*100):2d}  →  Ayşe: {sa:.1f}  Barış: {sb:.1f}  → {durum}")


if __name__ == "__main__":
    main()
