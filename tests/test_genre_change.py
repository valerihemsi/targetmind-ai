"""
Test Senaryo 4: Hedef Oyun Türü Değişikliği
--------------------------------------------
Mevcut hedef  : Strateji/RPG — "Chronicles of Aether" (derin, uzun oturumlar)
Yeni hedef    : Casual/Mobil — "Tap & Win" (kısa, sık oturumlar, reklam geliri)

Soru:
  - Ağırlıklar nasıl değişmeli? Neden?
  - Final hedef kitle ne kadar değişiyor?
  - İki kitle arasındaki overlap nedir?
"""
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, ".")


# ── İki oyun profili ─────────────────────────────────────────────────────────

OYUN_PROFILLERI = {

    "Strateji/RPG (Mevcut)": {
        "aciklama": "Derin strateji oyunu — uzun oturumlar, yüksek bağlılık, IAP ağırlıklı gelir",
        "tur_bonuslari": {
            "Strateji": 1.0, "RPG": 0.9, "Aksiyon": 0.6,
            "Casual": 0.2, "Spor": 0.2, "Bulmaca": 0.3,
        },
        "agirliklar": {
            "oyun_saati":  0.28,   # Uzun oturum kritik
            "harcama":     0.25,   # IAP (in-app purchase) önemli
            "oturum":      0.15,   # Oturum sayısı ikincil
            "kampanya":    0.10,
            "referans":    0.08,
            "tur_bonus":   0.14,
        },
        "hedef_turler":   ["Strateji", "RPG", "Aksiyon"],
        "hedef_platform": ["PC", "Android", "iOS", "Steam", "Epic Games"],
        "min_oyun_saati": 8,       # Haftada en az 8 saat
        "max_son_aktif":  60,
        "min_skor":       42,
    },

    "Casual/Mobil (Yeni)": {
        "aciklama": "Hyper-casual mobil oyun — kısa & sık oturumlar, reklam + mikro IAP geliri",
        "tur_bonuslari": {
            "Casual":   1.0, "Bulmaca": 0.9, "Spor": 0.7,
            "Aksiyon":  0.5, "Strateji": 0.2, "RPG": 0.2,
        },
        "agirliklar": {
            "oturum":      0.30,   # Günlük açılma = reklam geliri
            "kampanya":    0.22,   # Reklama tıklayanlar = para
            "oyun_saati":  0.12,   # Kısa oturumlar yeterli
            "harcama":     0.14,   # Mikro IAP, yüksek değil
            "referans":    0.10,   # Viral büyüme önemli
            "tur_bonus":   0.12,
        },
        "hedef_turler":   ["Casual", "Bulmaca", "Spor", "Aksiyon"],
        "hedef_platform": ["Android", "iOS"],   # Sadece mobil
        "min_oyun_saati": 2,        # Düşük — casual oyun az sürer
        "max_son_aktif":  30,       # Daha kısa inaktiflik toleransı
        "min_skor":       38,
    },
}


# ── Skor hesaplama ───────────────────────────────────────────────────────────

def skor_hesapla(df: pd.DataFrame, profil: dict) -> pd.DataFrame:
    df = df.copy()
    w  = profil["agirliklar"]
    tb = profil["tur_bonuslari"]

    def norm(col):
        mn, mx = df[col].min(), df[col].max()
        return (df[col] - mn) / (mx - mn + 1e-9)

    df["_s_oyun"]    = norm("haftalik_oyun_saati")
    df["_s_harca"]   = norm("aylik_ortalama_harcama")
    df["_s_oturum"]  = norm("aylik_oturum_sayisi")
    df["_s_kampanya"]= norm("kampanya_tiklanma_orani")
    df["_s_ref"]     = norm("arkadaslardan_referans")
    df["_s_tur"]     = df["tercih_edilen_tur"].map(tb).fillna(0.2)

    df["skor"] = (
        df["_s_oyun"]     * w["oyun_saati"] +
        df["_s_harca"]    * w["harcama"]    +
        df["_s_oturum"]   * w["oturum"]     +
        df["_s_kampanya"] * w["kampanya"]   +
        df["_s_ref"]      * w["referans"]   +
        df["_s_tur"]      * w["tur_bonus"]
    ) * 100

    df["skor"] = df["skor"].round(2)
    df = df.drop(columns=[c for c in df.columns if c.startswith("_s_")])
    return df


def filtrele(df: pd.DataFrame, profil: dict, profil_adi: str) -> pd.DataFrame:
    return df[
        (df["skor"]                  >= profil["min_skor"])       &
        (df["son_aktivite_gun"]      <= profil["max_son_aktif"])  &
        (df["haftalik_oyun_saati"]   >= profil["min_oyun_saati"]) &
        (df["tercih_edilen_tur"].isin(profil["hedef_turler"]))    &
        (df["platform"].isin(profil["hedef_platform"]))
    ].copy()


# ── Ana test ─────────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv("data/customers_cleaned.csv")

    print("=" * 68)
    print("TEST SENARYO 4 — HEDEF OYUN TÜRÜ DEĞİŞİKLİĞİ")
    print("=" * 68)

    sonuclar = {}
    for ad, profil in OYUN_PROFILLERI.items():
        df_s = skor_hesapla(df, profil)
        df_f = filtrele(df_s, profil, ad)
        sonuclar[ad] = {"skorlu": df_s, "final": df_f}

        print(f"\n── {ad.upper()} ──────────────────────────────────────────")
        print(f"Açıklama  : {profil['aciklama']}")
        print(f"Min skor  : {profil['min_skor']}  |  Max inaktiflik: {profil['max_son_aktif']} gün")
        print(f"Ağırlıklar:")
        for k, v in profil["agirliklar"].items():
            bar = "█" * int(v * 40)
            print(f"  {k:12s}: %{int(v*100):2d}  {bar}")

    # Karşılaştırma
    print("\n\n── HEDEF KİTLE KARŞILAŞTIRMASI ──────────────────────────────────")
    strateji_final = sonuclar["Strateji/RPG (Mevcut)"]["final"]
    casual_final   = sonuclar["Casual/Mobil (Yeni)"]["final"]

    ids_strateji = set(strateji_final["musteri_id"])
    ids_casual   = set(casual_final["musteri_id"])
    overlap      = ids_strateji & ids_casual

    print(f"\n{'Metrik':30s}  {'Strateji/RPG':>14}  {'Casual/Mobil':>14}")
    print("-" * 62)
    print(f"{'Final kitle':30s}  {len(strateji_final):>14}  {len(casual_final):>14}")
    print(f"{'Ortalama skor':30s}  {strateji_final['skor'].mean():>14.1f}  {casual_final['skor'].mean():>14.1f}")
    print(f"{'Ort. oyun saati (saat/hafta)':30s}  {strateji_final['haftalik_oyun_saati'].mean():>14.1f}  {casual_final['haftalik_oyun_saati'].mean():>14.1f}")
    print(f"{'Ort. harcama (₺/ay)':30s}  {strateji_final['aylik_ortalama_harcama'].mean():>14.0f}  {casual_final['aylik_ortalama_harcama'].mean():>14.0f}")
    print(f"{'Ort. oturum (ay)':30s}  {strateji_final['aylik_oturum_sayisi'].mean():>14.1f}  {casual_final['aylik_oturum_sayisi'].mean():>14.1f}")
    print(f"{'Mobil kullanıcı oranı':30s}  {(strateji_final['cihaz_turu']=='Mobil').mean()*100:>13.0f}%  {(casual_final['cihaz_turu']=='Mobil').mean()*100:>13.0f}%")
    print("-" * 62)
    print(f"{'Her iki listede de olan':30s}  {len(overlap):>14}  {'← OVERLAP':>14}")
    print(f"{'Sadece Strateji/RPG listesinde':30s}  {len(ids_strateji - ids_casual):>14}")
    print(f"{'Sadece Casual listesinde':30s}  {len(ids_casual - ids_strateji):>14}")

    # Tür dağılımı
    print(f"\n── TÜR DAĞILIMI ─────────────────────────────────────────────────")
    print(f"{'Tür':12s}  {'Strateji/RPG':>14}  {'Casual/Mobil':>14}")
    print("-" * 42)
    turler = set(strateji_final["tercih_edilen_tur"].tolist() +
                 casual_final["tercih_edilen_tur"].tolist())
    for tur in sorted(turler):
        sa = (strateji_final["tercih_edilen_tur"] == tur).sum()
        sb = (casual_final["tercih_edilen_tur"]   == tur).sum()
        if sa > 0 or sb > 0:
            print(f"{tur:12s}  {sa:>14}  {sb:>14}")

    # Gelir bias karşılaştırması
    print(f"\n── GELİR BİASI KARŞILAŞTIRMASI ─────────────────────────────────")
    print(f"{'Gelir grubu':12s}  {'Strateji skor ort.':>20}  {'Casual skor ort.':>18}")
    print("-" * 55)
    for g in ["Düşük", "Orta", "Yüksek"]:
        sa = sonuclar["Strateji/RPG (Mevcut)"]["skorlu"]
        sb = sonuclar["Casual/Mobil (Yeni)"]["skorlu"]
        va = sa[sa["gelir_seviyesi"] == g]["skor"].mean()
        vb = sb[sb["gelir_seviyesi"] == g]["skor"].mean()
        print(f"{g:12s}  {va:>20.1f}  {vb:>18.1f}")

    # Bias farkı
    sa_yuksek = sonuclar["Strateji/RPG (Mevcut)"]["skorlu"]
    sb_yuksek = sonuclar["Casual/Mobil (Yeni)"]["skorlu"]
    fark_s = sa_yuksek[sa_yuksek["gelir_seviyesi"]=="Yüksek"]["skor"].mean() - \
             sa_yuksek[sa_yuksek["gelir_seviyesi"]=="Düşük"]["skor"].mean()
    fark_c = sb_yuksek[sb_yuksek["gelir_seviyesi"]=="Yüksek"]["skor"].mean() - \
             sb_yuksek[sb_yuksek["gelir_seviyesi"]=="Düşük"]["skor"].mean()
    print(f"\nGelir grubu skor farkı → Strateji/RPG: {fark_s:.1f} puan | Casual/Mobil: {fark_c:.1f} puan")
    print(f"Bias eşiği (5 puan):    {'🔴 AŞILDI' if fark_s > 5 else '🟢 OK'} (Strateji)  |  {'🔴 AŞILDI' if fark_c > 5 else '🟢 OK'} (Casual)")

    # Sonuç
    print(f"""
── SONUÇ ────────────────────────────────────────────────────────

Casual/Mobil modelinde ne değişti:
  • Oturum ağırlığı: %15 → %30  (günlük açılma = reklam geliri)
  • Kampanya ağırlığı: %10 → %22 (reklama duyarlı kullanıcı = para)
  • Oyun saati ağırlığı: %28 → %12 (casual oyun az sürer, sorun değil)
  • Harcama ağırlığı: %25 → %14  (mikro IAP, büyük harcama beklentisi yok)
  • Platform: PC dahil → Sadece Mobil

Overlap = {len(overlap)} müşteri
  Bu {len(overlap)} kişi her iki oyun türü için de potansiyel.
  Şirket her iki oyunu da yayınlıyorsa buradan başlamak mantıklı.
""")


if __name__ == "__main__":
    main()
