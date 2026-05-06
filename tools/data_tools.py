"""
Ajanların CSV üzerinde çalışabilmesi için crewAI tool'ları.
Her tool pandas ile gerçek hesaplamayı yapar, LLM'e özet sunar.
Tüm araçlar column_mapping.json üzerinden herhangi bir CSV ile çalışır.
"""
import json
import pandas as pd
import numpy as np
from crewai.tools import tool

CSV_PATH = "data/customers.csv"
CLEANED_PATH = "data/customers_cleaned.csv"
SCORED_PATH = "data/customers_scored.csv"


def _load(path: str = CSV_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def _load_mapping() -> dict:
    """column_mapping.json varsa okur, yoksa oyun şirketi varsayılanlarını döndürür."""
    import os
    mapping_path = "data/column_mapping.json"
    if os.path.exists(mapping_path):
        with open(mapping_path) as f:
            return json.load(f)
    return {
        "id_col": "musteri_id",
        "demographic_cols": ["cinsiyet", "gelir_seviyesi", "yas"],
        "metric_cols": [
            "haftalik_oyun_saati", "aylik_ortalama_harcama",
            "aylik_oturum_sayisi", "kampanya_tiklanma_orani",
            "arkadaslardan_referans",
        ],
        "segment_col": "tercih_edilen_tur",
    }


# ── 1. Ham veri özeti ────────────────────────────────────────────────────────

@tool("Ham Veri Özeti")
def raw_data_summary(_: str = "") -> str:
    """Ham CSV'nin istatistiksel özetini döndürür: boyut, eksik değerler, veri tipleri."""
    df = _load(CSV_PATH)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(1)

    summary = {
        "toplam_satir": len(df),
        "toplam_sutun": len(df.columns),
        "sutunlar": list(df.columns),
        "eksik_deger_sayisi": missing[missing > 0].to_dict(),
        "eksik_deger_yuzdesi": missing_pct[missing_pct > 0].to_dict(),
        "veri_tipleri": df.dtypes.astype(str).to_dict(),
        "ornek_5_satir": df.head(5).to_dict(orient="records"),
        "numerik_istatistikler": df.describe().round(2).to_dict(),
    }
    return json.dumps(summary, ensure_ascii=False, indent=2)


# ── 2. Temizleme işlemi ──────────────────────────────────────────────────────

@tool("Veri Temizleme İşlemi")
def clean_data(rules_json: str = "") -> str:
    """
    Column mapping'e göre generic temizleme yapar.
    Herhangi bir CSV ile çalışır — oyun şirketi verisine özel değil.
    """
    df = _load(CSV_PATH)
    mapping = _load_mapping()
    report = {"baslangic_satir": len(df), "yapilan_islemler": []}

    try:
        rules = json.loads(rules_json) if rules_json.strip() else {}
    except Exception:
        rules = {}

    # 1. Yinelenen satırları sil
    before = len(df)
    df = df.drop_duplicates()
    report["yapilan_islemler"].append(f"Yinelenen satır silindi: {before - len(df)}")

    id_col = mapping.get("id_col", "")

    # 2. Metrik kolonları sayısala çevir, negatif değerleri NaN yap
    # (id_col'u atlıyoruz — string ID'leri sayısallaştırmak veriyi bozar)
    for col in mapping.get("metric_cols", []):
        if col not in df.columns or col == id_col:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            # Sayısala çevrilebiliyorsa çevir, yoksa atla
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() < len(df) * 0.5:
                continue  # çoğunluk NaN olacaksa string kolon — atla
            df[col] = converted
        neg = (df[col] < 0).sum()
        if neg > 0:
            df.loc[df[col] < 0, col] = np.nan
            report["yapilan_islemler"].append(f"{col}: {neg} negatif değer → NaN")

    # 3. IQR ile aykırı değer tespiti (gerçek sayısal metrik kolonlarda)
    for col in mapping.get("metric_cols", []):
        if col not in df.columns or col == id_col:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if df[col].dropna().nunique() < 4:
            continue  # çok az unique değer varsa IQR uygulanmaz
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        if outliers > 0:
            df.loc[(df[col] < lower) | (df[col] > upper), col] = np.nan
            report["yapilan_islemler"].append(f"{col}: {outliers} aykırı değer → NaN (3×IQR)")

    # 4. ID kolonu boş satırları sil
    if id_col and id_col in df.columns:
        before = len(df)
        df = df.dropna(subset=[id_col])
        report["yapilan_islemler"].append(f"ID kolonu boş satır silindi: {before - len(df)}")

    # 5. Numerik kolonlarda medyan ile doldur
    num_cols = df.select_dtypes(include="number").columns.tolist()
    for col in num_cols:
        missing = df[col].isna().sum()
        if missing > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            report["yapilan_islemler"].append(f"{col}: {missing} eksik → medyan ({median_val:.2f})")

    # 6. Kategorik kolonlarda mod ile doldur
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        missing = df[col].isna().sum()
        if missing > 0 and not df[col].mode().empty:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            report["yapilan_islemler"].append(f"{col}: {missing} eksik → mod ({mode_val})")

    df = df.reset_index(drop=True)
    df.to_csv(CLEANED_PATH, index=False)

    report["bitis_satir"] = len(df)
    report["elenen_satir"] = report["baslangic_satir"] - report["bitis_satir"]
    report["kalan_eksik_deger"] = df.isnull().sum()[df.isnull().sum() > 0].to_dict()

    # Demografik özet — mapping'teki kolonlarla
    demo_summary = {}
    for col in mapping.get("demographic_cols", []):
        if col in df.columns:
            demo_summary[col] = df[col].value_counts().head(10).to_dict()
    report["demografik_ozet"] = demo_summary

    return json.dumps(report, ensure_ascii=False, indent=2)


# ── 3. Segmentasyon ──────────────────────────────────────────────────────────

@tool("Segmentasyon Analizi")
def segmentation_analysis(_: str = "") -> str:
    """Temizlenmiş verinin segment dağılımını analiz eder. Herhangi bir CSV ile çalışır."""
    df = _load(CLEANED_PATH)
    mapping = _load_mapping()

    analysis = {"toplam_kayit": len(df), "tur_dagilimi": {}, "platform_dagilimi": {}}

    # Demografik/segment kolonların dağılımı
    for col in mapping.get("demographic_cols", []) + [mapping.get("segment_col", "")]:
        if col and col in df.columns:
            analysis[f"{col}_dagilimi"] = df[col].value_counts().head(15).to_dict()

    # Metrik kolonların tanımlayıcı istatistikleri
    metric_stats = {}
    for col in mapping.get("metric_cols", []):
        if col in df.columns:
            metric_stats[col] = {
                "ortalama": round(float(df[col].mean()), 3),
                "medyan":   round(float(df[col].median()), 3),
                "std":      round(float(df[col].std()), 3),
                "min":      round(float(df[col].min()), 3),
                "max":      round(float(df[col].max()), 3),
            }
    analysis["metrik_istatistikler"] = metric_stats

    # Geriye dönük uyumluluk
    seg_col = mapping.get("segment_col", "")
    analysis["tur_dagilimi"]      = analysis.get(f"{seg_col}_dagilimi", {})
    analysis["platform_dagilimi"] = {}

    return json.dumps(analysis, ensure_ascii=False, indent=2)


# ── 4. Skorlama ──────────────────────────────────────────────────────────────

@tool("Müşteri Skorlama")
def score_customers(weights_json: str = "") -> str:
    """
    Her kayda 0-100 arası potansiyel skoru verir.
    Metrik kolonlar mapping'den okunur, eşit ağırlık uygulanır (özel ağırlık verilebilir).
    weights_json: {"kolon_adi": 0.30, ...}  — toplamı 1.0 olmalı
    """
    df = _load(CLEANED_PATH)
    mapping = _load_mapping()
    metric_cols = [c for c in mapping.get("metric_cols", []) if c in df.columns
                   and pd.api.types.is_numeric_dtype(df[c])]

    if not metric_cols:
        return json.dumps({"hata": "Metrik kolon bulunamadı. Lütfen kolon eşlemesini yapın."})

    try:
        w = json.loads(weights_json) if weights_json.strip() else {}
    except Exception:
        w = {}

    # Ağırlıklar: verilmemişse eşit dağılım
    if not w:
        equal = round(1.0 / len(metric_cols), 4)
        w = {col: equal for col in metric_cols}

    def norm(series):
        mn, mx = series.min(), series.max()
        return (series - mn) / (mx - mn + 1e-9)

    skor = pd.Series(0.0, index=df.index)
    kullanilan = {}
    for col in metric_cols:
        agirlik = w.get(col, 1.0 / len(metric_cols))
        skor += norm(df[col]) * agirlik
        kullanilan[col] = round(agirlik, 4)

    df["potansiyel_skor"] = (skor * 100).round(2)

    # Skor segmenti
    df["skor_segmenti"] = pd.cut(
        df["potansiyel_skor"],
        bins=[0, 30, 50, 70, 100],
        labels=["Düşük", "Orta", "Yüksek", "Prime"]
    )

    df = df.drop(columns=[c for c in df.columns if c.startswith("_s_")])
    df.to_csv(SCORED_PATH, index=False)

    # Top-10 profil için generic kolon seçimi
    id_col = mapping.get("id_col", "")
    demo_cols = [c for c in mapping.get("demographic_cols", []) if c in df.columns]
    top_cols = ([id_col] if id_col and id_col in df.columns else []) + demo_cols + metric_cols + ["potansiyel_skor"]
    top_cols = list(dict.fromkeys(top_cols))

    report = {
        "kullanilan_agirliklar": kullanilan,
        "skor_dagilimi": df["skor_segmenti"].value_counts().to_dict(),
        "skor_istatistikleri": df["potansiyel_skor"].describe().round(2).to_dict(),
        "prime_musteri_sayisi": int((df["skor_segmenti"] == "Prime").sum()),
        "yuksek_musteri_sayisi": int((df["skor_segmenti"] == "Yüksek").sum()),
        "top_10_ortalama_profil": df.nlargest(10, "potansiyel_skor")[top_cols].fillna("—").to_dict(orient="records"),
    }
    return json.dumps(report, ensure_ascii=False, indent=2)


# ── 5. Bias tespiti ──────────────────────────────────────────────────────────

@tool("Bias Tespiti")
def detect_bias(_: str = "") -> str:
    """Skorlanmış verideki demografik bias'ı ölçer: seçim oranları, grup farklılıkları."""
    df = _load(SCORED_PATH)
    mapping = _load_mapping()
    demo_cols = [c for c in mapping.get("demographic_cols", []) if c in df.columns]

    high = df[df["skor_segmenti"].isin(["Yüksek", "Prime"])]
    report = {"toplam_musteri": len(df), "yuksek_prime_sayisi": len(high)}

    for col in demo_cols:
        total_dist = df[col].value_counts(normalize=True).round(3) * 100
        high_dist  = high[col].value_counts(normalize=True).round(3) * 100
        gap        = (high_dist - total_dist).round(1)

        report[col] = {
            "genel_dagilim_yuzde":  total_dist.to_dict(),
            "yuksek_segment_yuzde": high_dist.to_dict(),
            "fark_puan":            gap.dropna().to_dict(),
            "max_fark":             round(float(gap.abs().max()), 1) if not gap.empty else 0,
        }

    # Ortalama skor karşılaştırması — her demografik kolon için
    for col in demo_cols:
        try:
            report[f"{col}_ortalama_skor"] = df.groupby(col)["potansiyel_skor"].mean().round(2).to_dict()
        except Exception:
            pass

    return json.dumps(report, ensure_ascii=False, indent=2)


# ── 6. Proxy değişken tespiti ────────────────────────────────────────────────

@tool("Proxy Değişken Tespiti")
def detect_proxy_variables(_: str = "") -> str:
    """Dolaylı bias üretebilecek proxy değişkenleri tespit eder."""
    df = _load(SCORED_PATH)
    mapping = _load_mapping()

    sensitive = [c for c in mapping.get("demographic_cols", []) if c in df.columns]
    id_col = mapping.get("id_col", "")
    skip = set(sensitive + [id_col, "skor_segmenti", "potansiyel_skor"])
    # Proxy adayları: kategorik kolonlar (demografik + id dışı)
    potential_proxies = [
        c for c in df.select_dtypes(include=["object", "category"]).columns
        if c not in skip
    ]

    report = {"proxy_analizi": []}

    for proxy in potential_proxies:
        for sens in sensitive:
            try:
                ct = pd.crosstab(df[proxy], df[sens])
                chi2 = 0.0
                for col in ct.columns:
                    expected = ct[col].sum() * ct.sum(axis=1) / len(df)
                    chi2 += ((ct[col] - expected) ** 2 / (expected + 1e-9)).sum()
                n = len(df)
                k = min(ct.shape)
                cramer_v = round(float(np.sqrt(chi2 / (n * (k - 1) + 1e-9))), 3) if k > 1 else 0

                if cramer_v > 0.15:
                    risk = "YÜKSEK" if cramer_v > 0.35 else "ORTA"
                    report["proxy_analizi"].append({
                        "degisken": proxy,
                        "hassas_degisken": sens,
                        "iliski_gucü_cramerV": cramer_v,
                        "risk_seviyesi": risk,
                        "aciklama": f"'{proxy}' değişkeni '{sens}' ile güçlü ilişki taşıyor — dolaylı bias riski."
                    })
            except Exception:
                continue

    report["proxy_analizi"].sort(key=lambda x: x["iliski_gucü_cramerV"], reverse=True)
    report["yuksek_riskli_proxy_sayisi"] = sum(1 for p in report["proxy_analizi"] if p["risk_seviyesi"] == "YÜKSEK")
    return json.dumps(report, ensure_ascii=False, indent=2)


# ── 6b. Temizleme kararı denetimi ────────────────────────────────────────────

@tool("Temizleme Kararı Denetimi")
def audit_cleaning_decisions(_: str = "") -> str:
    """
    Temizleme ajanının dolgu kararlarının bias etkisini ölçer.
    Ham veri ile temizlenmiş veriyi karşılaştırarak demografik ve metrik
    dağılımların nasıl değiştiğini gösterir.
    """
    raw     = _load(CSV_PATH)
    cleaned = _load(CLEANED_PATH)
    mapping = _load_mapping()
    report  = {"denetlenen_kararlar": []}

    # Demografik kolon dağılım kaymaları
    demo_cols = [c for c in mapping.get("demographic_cols", [])
                 if c in raw.columns and c in cleaned.columns
                 and raw[c].dtype == object or (c in raw.columns and raw[c].dtype.name == "object")]

    for col in demo_cols:
        try:
            raw_dist     = raw[col].value_counts(normalize=True).round(3) * 100
            cleaned_dist = cleaned[col].value_counts(normalize=True).round(3) * 100
            shift        = (cleaned_dist - raw_dist.reindex(cleaned_dist.index, fill_value=0)).round(1)
            max_shift    = float(shift.abs().max()) if not shift.empty else 0
            bias_risk    = "YÜKSEK" if max_shift > 4 else "ORTA" if max_shift > 2 else "DÜŞÜK"

            report["denetlenen_kararlar"].append({
                "karar": f"'{col}' kolonu eksik değerleri dolduruldu / temizlendi",
                "ham_dagilim_yuzde":         raw_dist.to_dict(),
                "temizlenmis_dagilim_yuzde": cleaned_dist.to_dict(),
                "dagilim_kayması_puan":       shift.to_dict(),
                "bias_riski": bias_risk,
                "yorum": (
                    f"'{col}' dağılımında {max_shift:.1f} puanlık kayma. "
                    + ("Mod dolgu belirli grupları fazla temsil ediyor olabilir."
                       if bias_risk != "DÜŞÜK"
                       else "Dağılım değişimi kabul edilebilir düzeyde.")
                ),
            })
        except Exception:
            continue

    # Metrik kolon ortalama kaymaları
    metric_cols = [c for c in mapping.get("metric_cols", [])
                   if c in raw.columns and c in cleaned.columns]
    for col in metric_cols:
        try:
            raw_mean   = float(pd.to_numeric(raw[col], errors="coerce").dropna().mean())
            clean_mean = float(cleaned[col].mean())
            pct_change = abs(clean_mean - raw_mean) / (abs(raw_mean) + 1e-9) * 100
            report["denetlenen_kararlar"].append({
                "karar": f"'{col}' kolonu aykırı değerler ve eksikler temizlendi",
                "ham_ortalama":         round(raw_mean, 2),
                "temizlenmis_ortalama": round(clean_mean, 2),
                "degisim_yuzde":        round(pct_change, 1),
                "bias_riski": "ORTA" if pct_change > 10 else "DÜŞÜK",
                "yorum": f"Ortalama {pct_change:.1f}% değişti.",
            })
        except Exception:
            continue

    yuksek_risk = sum(1 for k in report["denetlenen_kararlar"] if k["bias_riski"] == "YÜKSEK")
    orta_risk   = sum(1 for k in report["denetlenen_kararlar"] if k["bias_riski"] == "ORTA")
    report["ozet"] = {
        "toplam_karar_denetlendi": len(report["denetlenen_kararlar"]),
        "yuksek_riskli_karar":     yuksek_risk,
        "orta_riskli_karar":       orta_risk,
        "kritik_bulgu": (
            "Temizleme kararları demografik dağılımda anlamlı kaymaya yol açıyor."
            if yuksek_risk > 0 else "Temizleme kararları kabul edilebilir risk seviyesinde."
        ),
    }
    return json.dumps(report, ensure_ascii=False, indent=2)


# ── 6c. Bias eşiği kontrolü ──────────────────────────────────────────────────

@tool("Bias Eşiği Kontrolü")
def bias_threshold_check(threshold_json: str = "") -> str:
    """
    Demografik gruplar arası skor farkı belirlenen eşiği aşıyorsa otomatik uyarı üretir.
    threshold_json: JSON — örn. {"esik_puan": 5}
    """
    df = _load(SCORED_PATH)
    mapping = _load_mapping()

    try:
        t = json.loads(threshold_json) if threshold_json.strip() else {}
    except Exception:
        t = {}

    esik = t.get("esik_puan", 5)
    demo_cols  = [c for c in mapping.get("demographic_cols", []) if c in df.columns]
    metric_cols = [c for c in mapping.get("metric_cols", []) if c in df.columns
                   and pd.api.types.is_numeric_dtype(df[c])]

    uyarilar = []
    group_results = {}

    # Demografik grup skor farklarını hesapla
    for col in demo_cols:
        try:
            group_scores = df.groupby(col)["potansiyel_skor"].mean()
            max_fark = round(float(group_scores.max() - group_scores.min()), 2)
            group_results[col] = {
                "grup_ortalamalar": group_scores.round(2).to_dict(),
                "max_fark": max_fark,
            }
            if max_fark > esik:
                uyarilar.append({
                    "uyari_turu": f"{col.upper()} BİASI",
                    "siddet": "KRİTİK" if max_fark > 10 else "YÜKSEK",
                    "olculen_fark_puan": max_fark,
                    "esik_puan": esik,
                    "oneri": f"'{col}' grubundaki fark {max_fark:.1f} puan — ağırlıkları dengeleyin.",
                })
        except Exception:
            continue

    # Metrik-skor korelasyonları
    corr_results = {}
    for col in metric_cols:
        try:
            corr = round(float(df[col].corr(df["potansiyel_skor"])), 3)
            corr_results[col] = corr
            if corr > 0.5:
                uyarilar.append({
                    "uyari_turu": f"{col} BAĞIMLILIĞI",
                    "siddet": "ORTA",
                    "olculen_korelasyon": corr,
                    "oneri": f"'{col}' skoru domine ediyor (r={corr:.2f}) — ağırlığını düşürün.",
                })
        except Exception:
            continue

    # Geriye dönük uyumluluk alanları
    first_demo_fark = list(group_results.values())[0]["max_fark"] if group_results else 0
    return json.dumps({
        "esik_puan": esik,
        "demografik_grup_analizleri": group_results,
        "metrik_skor_korelasyonlari": corr_results,
        "gelir_fark_puan": first_demo_fark,
        "esik_asimi": len(uyarilar) > 0,
        "uyarilar": uyarilar,
        "durum": "🔴 BİAS EŞİĞİ AŞILDI" if uyarilar else "🟢 Bias eşiği dahilinde",
    }, ensure_ascii=False, indent=2)


# ── 6d. Counterfactual test ───────────────────────────────────────────────────

@tool("Counterfactual Test")
def counterfactual_test(scenarios_json: str = "") -> str:
    """
    Farklı metrik ağırlığı senaryolarında skoru yeniden hesaplar ve
    her senaryonun demografik bias profilini karşılaştırır.
    scenarios_json: opsiyonel — boş bırakılırsa eşit ağırlık ve top-metrik senaryoları test edilir.
    """
    df = _load(SCORED_PATH)
    mapping = _load_mapping()
    metric_cols = [c for c in mapping.get("metric_cols", []) if c in df.columns
                   and pd.api.types.is_numeric_dtype(df[c])]
    demo_cols = [c for c in mapping.get("demographic_cols", []) if c in df.columns]

    if not metric_cols:
        return json.dumps({"hata": "Sayısal metrik kolon bulunamadı."})

    def _norm(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-9)

    normed = {col: _norm(df[col]) for col in metric_cols}
    n = len(normed)
    equal_w = {col: round(1.0 / n, 4) for col in normed}

    # Eğer özel senaryo verilmemişse 2 generic senaryo oluştur
    try:
        custom = json.loads(scenarios_json) if scenarios_json.strip() else []
    except Exception:
        custom = []

    if custom:
        scenario_weights = []
        for sc in custom:
            scenario_weights.append({"isim": sc.get("isim", "Senaryo"), "agirliklar": sc})
    else:
        scenario_weights = [{"isim": "Eşit Ağırlık", "agirliklar": equal_w}]
        if n > 1:
            first_col = metric_cols[0]
            rest_w = round(0.5 / (n - 1), 4)
            heavy_w = {col: (0.50 if col == first_col else rest_w) for col in normed}
            scenario_weights.append({"isim": f"{first_col} Ağırlıklı", "agirliklar": heavy_w})
        if n > 2:
            last_col = metric_cols[-1]
            rest_w2 = round(0.5 / (n - 1), 4)
            alt_w = {col: (0.50 if col == last_col else rest_w2) for col in normed}
            scenario_weights.append({"isim": f"{last_col} Ağırlıklı", "agirliklar": alt_w})

    sonuclar = []
    for sc in scenario_weights:
        w = sc["agirliklar"]
        skor = sum(normed[col] * w.get(col, equal_w[col]) for col in normed) * 100
        seg  = pd.cut(skor, bins=[0, 30, 50, 70, 100],
                      labels=["Düşük", "Orta", "Yüksek", "Prime"])
        high_mask = seg.isin(["Yüksek", "Prime"])

        group_gaps = {}
        max_gap = 0.0
        for d_col in demo_cols:
            try:
                g = df.groupby(d_col).apply(lambda g_: skor[g_.index].mean()).round(2)
                gap = round(float(g.max() - g.min()), 2)
                group_gaps[d_col] = {"grup_ortalamalari": g.to_dict(), "max_fark": gap}
                max_gap = max(max_gap, gap)
            except Exception:
                pass

        sonuclar.append({
            "senaryo": sc["isim"],
            "agirliklar": {k: round(v, 4) for k, v in w.items()},
            "skor_segmenti_dagilimi": seg.value_counts().to_dict(),
            "yuksek_prime_sayisi": int(high_mask.sum()),
            "demografik_fark_analizi": group_gaps,
            "max_demografik_fark": max_gap,
            "gelir_grubu_fark_puan": max_gap,
            "bias_durumu": (
                "🔴 Kritik" if max_gap > 10 else
                "🟡 Yüksek" if max_gap > 5 else
                "🟢 Kabul edilebilir"
            ),
        })

    best = min(sonuclar, key=lambda x: x["max_demografik_fark"])

    return json.dumps({
        "test_edilen_senaryo_sayisi": len(sonuclar),
        "senaryolar": sonuclar,
        "tavsiye_edilen_senaryo": best["senaryo"],
        "tavsiye_gerekce": (
            f"'{best['senaryo']}' senaryosunda max demografik fark "
            f"{best['max_demografik_fark']:.1f} puana düşüyor. "
            f"Yüksek+Prime müşteri sayısı: {best['yuksek_prime_sayisi']}."
        ),
    }, ensure_ascii=False, indent=2)


# ── 6e. Gelecek potansiyel ────────────────────────────────────────────────────

@tool("Gelecek Harcama Potansiyeli")
def calculate_future_potential(_: str = "") -> str:
    """
    Mevcut skordan bağımsız, gelecekte değer yaratma olasılığını hesaplar.
    Metrik kolonlar mapping'den okunur, eşit ağırlıklı engagement modeli uygulanır.
    """
    df = _load(SCORED_PATH)
    mapping = _load_mapping()
    metric_cols = [c for c in mapping.get("metric_cols", []) if c in df.columns
                   and pd.api.types.is_numeric_dtype(df[c])]

    if not metric_cols:
        return json.dumps({"hata": "Sayısal metrik kolon bulunamadı."})

    def norm(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-9)

    # Eşit ağırlıklı engagement skoru
    equal_w = 1.0 / len(metric_cols)
    engagement = sum(norm(df[col]) * equal_w for col in metric_cols)

    df["gelecek_potansiyel"] = (engagement * 100).round(2)

    df["potansiyel_kategori"] = pd.cut(
        df["gelecek_potansiyel"],
        bins=[0, 30, 50, 70, 100],
        labels=["Düşük", "Orta", "Yüksek", "Prime"]
    )

    # Optimal kitle: mevcut skor ≥ 38 VE gelecek potansiyel ≥ 45
    optimal = df[
        (df["potansiyel_skor"] >= 38) &
        (df["gelecek_potansiyel"] >= 45)
    ].copy()

    # Display kolonları
    id_col = mapping.get("id_col", "")
    demo_cols = [c for c in mapping.get("demographic_cols", []) if c in df.columns]
    display_cols = ([id_col] if id_col and id_col in df.columns else []) + demo_cols + metric_cols + ["potansiyel_skor", "gelecek_potansiyel"]
    display_cols = list(dict.fromkeys([c for c in display_cols if c in optimal.columns]))

    optimal_sorted = (optimal.nlargest(20, "gelecek_potansiyel")[display_cols]
                      .fillna("—").to_dict(orient="records"))

    df.to_csv(SCORED_PATH, index=False)
    optimal.nlargest(20, "gelecek_potansiyel").to_csv("data/optimal_targets.csv", index=False)

    return json.dumps({
        "gelecek_potansiyel_dagilimi": df["potansiyel_kategori"].value_counts().to_dict(),
        "optimal_kitle_sayisi": len(optimal),
        "optimal_ort_gelecek_skor": round(float(optimal["gelecek_potansiyel"].mean()), 1) if len(optimal) > 0 else 0,
        "optimal_ort_mevcut_skor": round(float(optimal["potansiyel_skor"].mean()), 1) if len(optimal) > 0 else 0,
        "optimal_liste": optimal_sorted,
        "kullanilan_metrikler": metric_cols,
    }, ensure_ascii=False, indent=2)


# ── 7. Final optimizasyon ────────────────────────────────────────────────────

@tool("Final Hedef Kitle Oluştur")
def build_final_targets(criteria_json: str = "") -> str:
    """
    Tüm analizler sonucunda final hedef kitle listesini oluşturur.
    criteria_json: JSON — örn. {"min_skor": 50, "hedef_segmentler": ["A","B"]}
    """
    df = _load(SCORED_PATH)
    mapping = _load_mapping()

    try:
        c = json.loads(criteria_json) if criteria_json.strip() else {}
    except Exception:
        c = {}

    min_skor = c.get("min_skor", 50)
    filtered = df[df["potansiyel_skor"] >= min_skor].copy()

    # Opsiyonel: segment kolonu filtresi
    segment_col = mapping.get("segment_col", "")
    hedef_segmentler = c.get("hedef_segmentler", [])
    if segment_col and segment_col in df.columns and hedef_segmentler:
        filtered = filtered[filtered[segment_col].isin(hedef_segmentler)]

    filtered.to_csv("data/final_targets.csv", index=False)

    report = {
        "uygulanan_kriterler": {"minimum_skor": min_skor},
        "baslangic_havuz": len(df),
        "final_hedef_kitle": len(filtered),
        "elenme_orani_yuzde": round((1 - len(filtered) / len(df)) * 100, 1) if len(df) > 0 else 0,
        "segment_dagilimi": filtered["skor_segmenti"].value_counts().to_dict() if "skor_segmenti" in filtered.columns else {},
        "skor_ortalama": round(float(filtered["potansiyel_skor"].mean()), 2) if len(filtered) > 0 else 0,
        "cikti_dosyasi": "data/final_targets.csv",
    }

    # Demografik dağılımlar
    demo_cols = [col for col in mapping.get("demographic_cols", []) if col in filtered.columns]
    for col in demo_cols:
        report[f"{col}_dagilimi"] = filtered[col].value_counts().to_dict()

    # Metrik ortalamalar
    metric_cols = [col for col in mapping.get("metric_cols", []) if col in filtered.columns]
    for col in metric_cols:
        try:
            report[f"{col}_ortalama"] = round(float(filtered[col].mean()), 2)
        except Exception:
            pass

    return json.dumps(report, ensure_ascii=False, indent=2)
