"""
TargetMind AI — 7 Ajan Öz-Farkındalıklı Pipeline
Her ajan kendi kararlarını sorgular, birbirlerini değerlendirir,
bias'larını fark ederek kendini düzeltir.
"""
import json, os
import pandas as pd
import numpy as np
from crewai.tools import tool
from pathlib import Path

CSV_PATH     = "data/customers.csv"
CLEANED_PATH = "data/customers_cleaned.csv"
SCORED_PATH  = "data/customers_scored.csv"
LOG_PATH     = "data/pipeline_log.json"


class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

def _dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, cls=_NpEncoder)


# ── Yardımcı fonksiyonlar ────────────────────────────────────────────────────

def _load(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def _load_mapping() -> dict:
    p = "data/column_mapping.json"
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {
        "id_col": "musteri_id",
        "demographic_cols": ["cinsiyet", "gelir_seviyesi", "yas"],
        "metric_cols": ["haftalik_oyun_saati", "aylik_ortalama_harcama",
                        "aylik_oturum_sayisi", "kampanya_tiklanma_orani",
                        "arkadaslardan_referans"],
        "segment_col": "tercih_edilen_tur",
    }

def _log_agent(agent_name: str, result: dict):
    """Her ajanın çıktısını pipeline log'una kaydeder."""
    log = {}
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH) as f:
            log = json.load(f)
    log[agent_name] = result
    Path(LOG_PATH).write_text(_dumps(log))

def _read_log() -> dict:
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH) as f:
            return json.load(f)
    return {}

def _demographic_shift(before: pd.DataFrame, after: pd.DataFrame, col: str) -> dict:
    """Bir kolondaki demografik dağılımın ne kadar kaydığını ölçer."""
    if col not in before.columns or col not in after.columns:
        return {}
    b = before[col].value_counts(normalize=True).round(3) * 100
    a = after[col].value_counts(normalize=True).round(3) * 100
    shift = (a - b.reindex(a.index, fill_value=0)).round(1)
    return {
        "oncesi": b.to_dict(),
        "sonrasi": a.to_dict(),
        "kayma_puan": shift.to_dict(),
        "max_kayma": round(float(shift.abs().max()), 1) if not shift.empty else 0,
    }

def _bias_contribution_score(max_kayma: float, max_skor_farki: float = 0) -> float:
    """0-1 arası bias katkı skoru. 0=temiz, 1=maksimum bias."""
    kayma_s = min(max_kayma / 10.0, 1.0)
    fark_s  = min(max_skor_farki / 15.0, 1.0)
    return round((kayma_s * 0.5 + fark_s * 0.5), 3)


# ══════════════════════════════════════════════════════════════
#  AJAN 1 — VERİ TEMİZLEME
# ══════════════════════════════════════════════════════════════

@tool("Veri Temizleme Ajanı")
def clean_data(rules_json: str = "") -> str:
    """
    Veriyi temizler VE kendi kararlarının yarattığı bias'ı ölçer.
    Öz-değerlendirme: 'Hangi temizleme kararım hangi demografiyi etkiledi?'
    """
    raw     = _load(CSV_PATH)
    mapping = _load_mapping()
    df      = raw.copy()
    report  = {"baslangic_satir": len(df), "kararlar": []}

    id_col   = mapping.get("id_col", "")
    demo_cols    = mapping.get("demographic_cols", [])
    metric_cols  = mapping.get("metric_cols", [])

    # 1. Yinelenenler
    before = len(df)
    df = df.drop_duplicates()
    report["kararlar"].append({"karar": "Yinelenen satır silme", "etki": before - len(df)})

    # 2. Metrik kolonları sayısala çevir, negatif → NaN
    for col in metric_cols:
        if col not in df.columns or col == id_col: continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() < len(df) * 0.5: continue
            df[col] = converted
        neg = (df[col] < 0).sum()
        if neg > 0:
            df.loc[df[col] < 0, col] = np.nan
            report["kararlar"].append({"karar": f"{col}: {neg} negatif → NaN", "etki": neg})

    # 3. IQR aykırı değer tespiti
    for col in metric_cols:
        if col not in df.columns or col == id_col: continue
        if not pd.api.types.is_numeric_dtype(df[col]): continue
        if df[col].dropna().nunique() < 4: continue
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 3*iqr, q3 + 3*iqr
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        if outliers > 0:
            df.loc[(df[col] < lower) | (df[col] > upper), col] = np.nan
            report["kararlar"].append({"karar": f"{col}: {outliers} aykırı değer → NaN (3×IQR)", "etki": outliers})

    # 4. ID boş satır silme
    if id_col and id_col in df.columns:
        before = len(df)
        df = df.dropna(subset=[id_col])
        report["kararlar"].append({"karar": "ID boş satır silme", "etki": before - len(df)})

    # 5. Numerik → medyan doldur
    for col in df.select_dtypes(include="number").columns:
        m = df[col].isna().sum()
        if m > 0:
            df[col] = df[col].fillna(df[col].median())
            report["kararlar"].append({"karar": f"{col}: {m} eksik → medyan", "etki": m})

    # 6. Kategorik → mod doldur
    for col in df.select_dtypes(include=["object", "category"]).columns:
        m = df[col].isna().sum()
        if m > 0 and not df[col].mode().empty:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            report["kararlar"].append({"karar": f"{col}: {m} eksik → mod ({mode_val})", "etki": m})

    df = df.reset_index(drop=True)
    df.to_csv(CLEANED_PATH, index=False)

    # ── ÖZ-DEĞERLENDİRME ──────────────────────────────────────
    oz_degerlendirme = {"demografik_kaymalar": {}, "en_riskli_karar": "", "bias_katki_skoru": 0.0}
    max_kayma_genel = 0.0
    for col in demo_cols:
        shift = _demographic_shift(raw, df, col)
        if shift:
            oz_degerlendirme["demografik_kaymalar"][col] = shift
            if shift["max_kayma"] > max_kayma_genel:
                max_kayma_genel = shift["max_kayma"]
                oz_degerlendirme["en_riskli_karar"] = f"'{col}' dağılımı {shift['max_kayma']:.1f} puan kaydı"

    oz_degerlendirme["bias_katki_skoru"] = _bias_contribution_score(max_kayma_genel)
    oz_degerlendirme["sonuc"] = (
        "⚠ Temizleme kararlarım demografik dağılımı anlamlı şekilde bozdu. "
        "Mod dolgu belirli grupları yapay olarak güçlendiriyor olabilir."
        if max_kayma_genel > 3 else
        "✓ Temizleme kararlarım demografik dağılımı kabul edilebilir düzeyde etkiledi."
    )

    report.update({
        "bitis_satir": len(df),
        "elenen_satir": report["baslangic_satir"] - len(df),
        "oz_degerlendirme": oz_degerlendirme,
    })

    _log_agent("Veri Temizleme", {
        "ozet": f"{report['baslangic_satir']} → {len(df)} satır",
        "oz_degerlendirme": oz_degerlendirme,
    })
    return _dumps(report)


# ══════════════════════════════════════════════════════════════
#  AJAN 2 — SEGMENTASYON & ANALİZ
# ══════════════════════════════════════════════════════════════

@tool("Segmentasyon Ajanı")
def segmentation_analysis(_: str = "") -> str:
    """
    Segmentasyon yapar VE kendi sınıflandırma kararlarının
    demografik temsili nasıl etkilediğini ölçer.
    """
    df      = _load(CLEANED_PATH)
    mapping = _load_mapping()
    demo_cols   = mapping.get("demographic_cols", [])
    metric_cols = mapping.get("metric_cols", [])
    seg_col     = mapping.get("segment_col", "")

    analysis = {"toplam_kayit": len(df)}

    # Segment dağılımı
    for col in demo_cols + ([seg_col] if seg_col else []):
        if col and col in df.columns:
            analysis[f"{col}_dagilimi"] = df[col].value_counts().head(15).to_dict()

    # Metrik istatistikler
    metric_stats = {}
    for col in metric_cols:
        if col in df.columns:
            metric_stats[col] = {
                "ortalama": round(float(df[col].mean()), 3),
                "medyan":   round(float(df[col].median()), 3),
                "std":      round(float(df[col].std()), 3),
            }
    analysis["metrik_istatistikler"] = metric_stats

    # Geriye dönük uyumluluk
    analysis["tur_dagilimi"]      = analysis.get(f"{seg_col}_dagilimi", {})
    analysis["platform_dagilimi"] = {}

    # ── ÖZ-DEĞERLENDİRME ──────────────────────────────────────
    oz = {"temsil_analizi": {}, "bias_katki_skoru": 0.0}
    max_temsil_farki = 0.0
    for col in demo_cols:
        if col not in df.columns: continue
        dist = df[col].value_counts(normalize=True) * 100
        max_v = float(dist.max())
        min_v = float(dist.min())
        fark  = round(max_v - min_v, 1)
        oz["temsil_analizi"][col] = {
            "dagilim": dist.round(1).to_dict(),
            "en_fazla_temsil_edilen": dist.idxmax(),
            "temsil_farki_puan": fark,
        }
        if fark > max_temsil_farki:
            max_temsil_farki = fark

    oz["bias_katki_skoru"] = _bias_contribution_score(max_temsil_farki / 2)
    oz["sonuc"] = (
        f"⚠ Segmentasyonda demografik dengesizlik var. "
        f"Maksimum temsil farkı: {max_temsil_farki:.1f} puan."
        if max_temsil_farki > 20 else
        "✓ Segmentasyon demografik dağılım açısından kabul edilebilir."
    )
    analysis["oz_degerlendirme"] = oz

    _log_agent("Segmentasyon", {"ozet": f"{len(df)} kayıt segmentlendi", "oz_degerlendirme": oz})
    return _dumps(analysis)


# ══════════════════════════════════════════════════════════════
#  AJAN 3 — İLK SKORLAMA
# ══════════════════════════════════════════════════════════════

@tool("İlk Skorlama Ajanı")
def score_customers(weights_json: str = "") -> str:
    """
    İlk skoru üretir VE kendi ağırlık kararlarının yarattığı
    demografik skor farklarını ölçer.
    """
    df      = _load(CLEANED_PATH)
    mapping = _load_mapping()
    metric_cols = [c for c in mapping.get("metric_cols", [])
                   if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    demo_cols   = [c for c in mapping.get("demographic_cols", []) if c in df.columns]

    if not metric_cols:
        return _dumps({"hata": "Metrik kolon bulunamadı."})

    try:
        w = json.loads(weights_json) if weights_json.strip() else {}
    except Exception:
        w = {}

    if not w:
        equal = round(1.0 / len(metric_cols), 4)
        w = {col: equal for col in metric_cols}

    def norm(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-9)

    skor = pd.Series(0.0, index=df.index)
    kullanilan = {}
    for col in metric_cols:
        agirlik = w.get(col, 1.0 / len(metric_cols))
        skor += norm(df[col]) * agirlik
        kullanilan[col] = round(agirlik, 4)

    df["potansiyel_skor"] = (skor * 100).round(2)
    df["skor_segmenti"] = pd.cut(
        df["potansiyel_skor"],
        bins=[0, 30, 50, 70, 100],
        labels=["Düşük", "Orta", "Yüksek", "Prime"]
    )
    df = df.drop(columns=[c for c in df.columns if c.startswith("_s_")])
    df.to_csv(SCORED_PATH, index=False)

    # ── ÖZ-DEĞERLENDİRME ──────────────────────────────────────
    oz = {"demografik_skor_farklari": {}, "en_riskli_agirlik": "", "bias_katki_skoru": 0.0}
    max_fark = 0.0
    for col in demo_cols:
        try:
            g = df.groupby(col)["potansiyel_skor"].mean().round(2)
            fark = round(float(g.max() - g.min()), 2)
            oz["demografik_skor_farklari"][col] = {
                "grup_ortalamalar": g.to_dict(),
                "max_fark_puan": fark,
                "risk": "YÜKSEK" if fark > 10 else "ORTA" if fark > 5 else "DÜŞÜK",
            }
            if fark > max_fark:
                max_fark = fark
        except Exception:
            pass

    # Hangi metrik en çok skoru domine ediyor?
    korlar = {}
    for col in metric_cols:
        try:
            korlar[col] = round(float(df[col].corr(df["potansiyel_skor"])), 3)
        except Exception:
            pass
    if korlar:
        en_dominant = max(korlar, key=lambda x: abs(korlar[x]))
        oz["en_riskli_agirlik"] = f"'{en_dominant}' (r={korlar[en_dominant]:+.3f}) skoru domine ediyor"
    oz["metrik_skor_korelasyonlari"] = korlar
    oz["bias_katki_skoru"] = _bias_contribution_score(0, max_fark)
    oz["sonuc"] = (
        f"⚠ Ağırlık kararlarım {max_fark:.1f} puanlık demografik skor farkı yarattı."
        if max_fark > 5 else
        "✓ Ağırlık kararlarım demografik gruplar arasında kabul edilebilir fark yarattı."
    )

    id_col = mapping.get("id_col", "")
    top_cols = ([id_col] if id_col and id_col in df.columns else []) + demo_cols + metric_cols + ["potansiyel_skor"]
    top_cols = list(dict.fromkeys([c for c in top_cols if c in df.columns]))

    result = {
        "kullanilan_agirliklar": kullanilan,
        "skor_dagilimi": df["skor_segmenti"].astype(str).value_counts().to_dict(),
        "prime_musteri_sayisi": int((df["skor_segmenti"] == "Prime").sum()),
        "oz_degerlendirme": oz,
        "top_10": df.nlargest(10, "potansiyel_skor")[top_cols].fillna("—").to_dict(orient="records"),
    }

    _log_agent("Skorlama", {"kullanilan_agirliklar": kullanilan, "oz_degerlendirme": oz})
    return _dumps(result)


# ══════════════════════════════════════════════════════════════
#  AJAN 4 — PROXY & BIAS TESPİTİ
# ══════════════════════════════════════════════════════════════

@tool("Proxy ve Bias Tespit Ajanı")
def detect_proxy_and_bias(_: str = "") -> str:
    """
    Proxy değişkenleri VE demografik bias'ı birlikte tespit eder.
    Öz-değerlendirme: 'Hangi değişkenleri görmezden geldim?'
    """
    df      = _load(SCORED_PATH)
    mapping = _load_mapping()
    demo_cols   = [c for c in mapping.get("demographic_cols", []) if c in df.columns]
    metric_cols = [c for c in mapping.get("metric_cols", []) if c in df.columns]
    id_col      = mapping.get("id_col", "")
    skip        = set(demo_cols + [id_col, "skor_segmenti", "potansiyel_skor"])

    # Proxy analizi
    proxy_analizi = []
    potential_proxies = [c for c in df.select_dtypes(include=["object","category"]).columns if c not in skip]
    for proxy in potential_proxies:
        for sens in demo_cols:
            try:
                ct   = pd.crosstab(df[proxy], df[sens])
                n, k = len(df), min(ct.shape)
                chi2 = sum(((ct[c] - ct[c].sum()*ct.sum(axis=1)/n)**2 / (ct[c].sum()*ct.sum(axis=1)/n + 1e-9)).sum() for c in ct.columns)
                cv   = round(float(np.sqrt(chi2 / (n*(k-1)+1e-9))), 3) if k > 1 else 0
                if cv > 0.15:
                    proxy_analizi.append({
                        "degisken": proxy, "hassas_degisken": sens,
                        "cramer_v": cv,
                        "risk": "YÜKSEK" if cv > 0.35 else "ORTA",
                    })
            except Exception:
                continue

    proxy_analizi.sort(key=lambda x: x["cramer_v"], reverse=True)

    # Bias tespiti
    high = df[df["skor_segmenti"].isin(["Yüksek", "Prime"])]
    bias_tespiti = {}
    for col in demo_cols:
        total = df[col].value_counts(normalize=True).round(3) * 100
        h_dist = high[col].value_counts(normalize=True).round(3) * 100
        gap   = (h_dist - total).round(1)
        bias_tespiti[col] = {
            "genel_dagilim": total.to_dict(),
            "yuksek_segment": h_dist.to_dict(),
            "fark": gap.dropna().to_dict(),
            "max_fark": round(float(gap.abs().max()), 1) if not gap.empty else 0,
        }

    # Metrik-skor korelasyonları
    korlar = {}
    for col in metric_cols:
        try: korlar[col] = round(float(df[col].corr(df["potansiyel_skor"])), 3)
        except Exception: pass

    # ── ÖZ-DEĞERLENDİRME ──────────────────────────────────────
    gozden_kacirilanlar = []
    for col in demo_cols:
        for m in metric_cols:
            if m not in df.columns: continue
            try:
                corr = df.groupby(col)[m].mean()
                fark = float(corr.max() - corr.min())
                if fark > df[m].std() * 0.5:
                    gozden_kacirilanlar.append(f"'{m}' metriği '{col}' ile {fark:.1f} fark taşıyor — skorlamada dolaylı bias kaynağı olabilir")
            except Exception:
                pass

    oz = {
        "gozden_kacirilan_iliskiler": gozden_kacirilanlar,
        "bias_katki_skoru": _bias_contribution_score(len(proxy_analizi) * 2),
        "sonuc": (
            f"⚠ {len([p for p in proxy_analizi if p['risk']=='YÜKSEK'])} yüksek riskli proxy tespit ettim. "
            f"Bu değişkenleri gözden kaçırmak bias değerlendirmemi eksik bıraktı."
            if any(p["risk"] == "YÜKSEK" for p in proxy_analizi) else
            "✓ Proxy değişken analizi tamamlandı."
        ),
    }

    result = {
        "proxy_analizi": proxy_analizi,
        "yuksek_riskli_proxy": sum(1 for p in proxy_analizi if p["risk"] == "YÜKSEK"),
        "bias_tespiti": bias_tespiti,
        "metrik_skor_korelasyonlari": korlar,
        "oz_degerlendirme": oz,
    }

    _log_agent("Proxy & Bias Tespiti", {"yuksek_proxy": result["yuksek_riskli_proxy"], "oz_degerlendirme": oz})
    return _dumps(result)


# ══════════════════════════════════════════════════════════════
#  AJAN 5 — ÇAPRAZ AJAN DEĞERLENDİRMESİ (ELEŞTİRMEN)
# ══════════════════════════════════════════════════════════════

@tool("Çapraz Ajan Değerlendirme Ajanı")
def inter_agent_critique(_: str = "") -> str:
    """
    Tüm ajanların öz-değerlendirmelerini okur, doğrular, çelişkileri bulur.
    Her ajanın toplam bias katkısını hesaplar ve düzeltme önerileri sunar.
    """
    log     = _read_log()
    mapping = _load_mapping()
    demo_cols   = mapping.get("demographic_cols", [])
    metric_cols = mapping.get("metric_cols", [])

    degerlendirmeler = []
    duzeltme_onerileri = {}

    # ── TEMIZLEME AJANI DEĞERLENDİRMESİ ──────────────────────
    if "Veri Temizleme" in log:
        vt = log["Veri Temizleme"]["oz_degerlendirme"]
        max_k = max((v.get("max_kayma", 0) for v in vt.get("demografik_kaymalar", {}).values()), default=0)
        dogrulama = "ONAYLANDI" if max_k > 2 else "HAFİFE ALINDI"
        degerlendirmeler.append({
            "ajan": "Veri Temizleme",
            "kendi_bildirdigi_bias_skoru": vt.get("bias_katki_skoru", 0),
            "dogrulama": dogrulama,
            "elestiri": (
                f"Mod dolgu kararları {max_k:.1f} puanlık demografik kayma yarattı. "
                "Eksik değerleri doldurmak yerine 'Bilinmiyor' kategorisi oluşturulmalıydı."
                if max_k > 3 else
                "Temizleme kararları bias açısından yeterince değerlendirilmiş."
            ),
            "duzeltme": "Eksik kategorik değerler için 'Bilinmiyor' etiketi kullan" if max_k > 3 else None,
        })
        if max_k > 3:
            duzeltme_onerileri["Veri Temizleme"] = "Mod dolgu yerine 'Bilinmiyor' kategorisi"

    # ── SKORLAMA AJANI DEĞERLENDİRMESİ ───────────────────────
    if "Skorlama" in log:
        sk = log["Skorlama"]["oz_degerlendirme"]
        max_fark = max((v.get("max_fark_puan", 0) for v in sk.get("demografik_skor_farklari", {}).values()), default=0)
        dominant_metrik = sk.get("en_riskli_agirlik", "")
        dogrulama = "ONAYLANDI" if max_fark > 5 else "HAFİFE ALINDI"
        degerlendirmeler.append({
            "ajan": "Skorlama",
            "kendi_bildirdigi_bias_skoru": sk.get("bias_katki_skoru", 0),
            "dogrulama": dogrulama,
            "elestiri": (
                f"{max_fark:.1f} puanlık demografik skor farkı oluştu. "
                f"{dominant_metrik}. Eşit ağırlık dağılımı daha adil olurdu."
                if max_fark > 5 else
                "Skorlama ağırlıkları demografik açıdan kabul edilebilir sonuç vermiş."
            ),
            "duzeltme": f"Tüm metrikleri eşit ağırlıkla skorla" if max_fark > 5 else None,
        })
        if max_fark > 5:
            n = len(metric_cols)
            equal_w = {c: round(1.0/n, 4) for c in metric_cols} if n else {}
            duzeltme_onerileri["Skorlama"] = {"yeni_agirliklar": equal_w, "beklenen_fark_azalmasi": f"%{min(int(max_fark*3), 60)}"}

    # ── PROXY AJANI DEĞERLENDİRMESİ ──────────────────────────
    if "Proxy & Bias Tespiti" in log:
        pb = log["Proxy & Bias Tespiti"]["oz_degerlendirme"]
        degerlendirmeler.append({
            "ajan": "Proxy & Bias Tespiti",
            "kendi_bildirdigi_bias_skoru": pb.get("bias_katki_skoru", 0),
            "dogrulama": "ONAYLANDI",
            "elestiri": pb.get("sonuc", ""),
            "duzeltme": None,
        })

    # ── TOPLAM BİAS KATKI SIRALAMASI ─────────────────────────
    siralama = sorted(
        [{"ajan": d["ajan"], "skor": d["kendi_bildirdigi_bias_skoru"]} for d in degerlendirmeler],
        key=lambda x: x["skor"], reverse=True
    )

    en_yuksek = siralama[0]["ajan"] if siralama else "—"

    result = {
        "ajan_degerlendirmeleri": degerlendirmeler,
        "bias_katki_siramasi": siralama,
        "en_yuksek_katkili_ajan": en_yuksek,
        "duzeltme_onerileri": duzeltme_onerileri,
        "ozet": (
            f"En yüksek bias katkısı '{en_yuksek}' ajanından geliyor. "
            f"{len(duzeltme_onerileri)} ajan için düzeltme önerildi."
        ),
    }

    _log_agent("Çapraz Değerlendirme", {"ozet": result["ozet"], "duzeltme_onerileri": duzeltme_onerileri})
    return _dumps(result)


# ══════════════════════════════════════════════════════════════
#  AJAN 6 — DÜZELTİLMİŞ YENİDEN SKORLAMA
# ══════════════════════════════════════════════════════════════

@tool("Düzeltilmiş Skorlama Ajanı")
def corrected_scoring(_: str = "") -> str:
    """
    Eleştirmen ajanın önerilerine göre yeni ağırlıklarla skoru yeniden hesaplar.
    Öncesi/sonrası karşılaştırması yapar.
    """
    df      = _load(CLEANED_PATH)
    mapping = _load_mapping()
    log     = _read_log()
    metric_cols = [c for c in mapping.get("metric_cols", [])
                   if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    demo_cols   = [c for c in mapping.get("demographic_cols", []) if c in df.columns]

    # Düzeltme önerilerini oku
    duzeltmeler = log.get("Çapraz Değerlendirme", {}).get("duzeltme_onerileri", {})
    yeni_agirliklar = duzeltmeler.get("Skorlama", {}).get("yeni_agirliklar", {})

    if not yeni_agirliklar:
        n = len(metric_cols)
        yeni_agirliklar = {c: round(1.0/n, 4) for c in metric_cols} if n else {}

    def norm(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-9)

    # Önceki skorları oku
    onceki_df = _load(SCORED_PATH) if os.path.exists(SCORED_PATH) else df.copy()
    onceki_farklar = {}
    for col in demo_cols:
        try:
            g = onceki_df.groupby(col)["potansiyel_skor"].mean()
            onceki_farklar[col] = round(float(g.max() - g.min()), 2)
        except Exception:
            onceki_farklar[col] = 0

    # Yeni skoru hesapla
    skor = sum(norm(df[c]) * yeni_agirliklar.get(c, 1.0/len(metric_cols)) for c in metric_cols)
    df["potansiyel_skor"] = (skor * 100).round(2)
    df["skor_segmenti"] = pd.cut(
        df["potansiyel_skor"], bins=[0,30,50,70,100],
        labels=["Düşük","Orta","Yüksek","Prime"]
    )
    df.to_csv(SCORED_PATH, index=False)

    # Yeni farklar
    sonraki_farklar = {}
    for col in demo_cols:
        try:
            g = df.groupby(col)["potansiyel_skor"].mean()
            sonraki_farklar[col] = round(float(g.max() - g.min()), 2)
        except Exception:
            sonraki_farklar[col] = 0

    # Karşılaştırma
    karsilastirma = {}
    for col in demo_cols:
        once  = onceki_farklar.get(col, 0)
        sonra = sonraki_farklar.get(col, 0)
        karsilastirma[col] = {
            "onceki_fark": once,
            "sonraki_fark": sonra,
            "iyilesme_puan": round(once - sonra, 2),
            "iyilesme_yuzde": round((once - sonra) / (once + 1e-9) * 100, 1),
        }

    toplam_iyilesme = sum(v["iyilesme_puan"] for v in karsilastirma.values())

    result = {
        "uygulanan_agirliklar": yeni_agirliklar,
        "skor_dagilimi": df["skor_segmenti"].astype(str).value_counts().to_dict(),
        "oncesi_sonrasi": karsilastirma,
        "toplam_bias_azalmasi_puan": round(toplam_iyilesme, 2),
        "ozet": (
            f"Düzeltme sonrası toplam {toplam_iyilesme:.1f} puanlık bias azaldı."
            if toplam_iyilesme > 0 else
            "Düzeltme mevcut bias düzeyini değiştirmedi."
        ),
    }

    _log_agent("Düzeltilmiş Skorlama", {"ozet": result["ozet"], "oncesi_sonrasi": karsilastirma})
    return _dumps(result)


# ══════════════════════════════════════════════════════════════
#  AJAN 7 — FİNAL OPTİMİZASYON & RAPORLAR
# ══════════════════════════════════════════════════════════════

@tool("Final Optimizasyon ve Rapor Ajanı")
def build_final_and_report(criteria_json: str = "") -> str:
    """
    Düzeltilmiş skorlarla optimal müşteri havuzunu oluşturur.
    İki ayrı rapor üretir: Süreç Raporu + Optimal Kitle Raporu.
    """
    df      = _load(SCORED_PATH)
    mapping = _load_mapping()
    log     = _read_log()

    try:
        c = json.loads(criteria_json) if criteria_json.strip() else {}
    except Exception:
        c = {}

    min_skor = c.get("min_skor", 50)
    final    = df[df["potansiyel_skor"] >= min_skor].copy()
    final.to_csv("data/final_targets.csv", index=False)

    demo_cols   = [col for col in mapping.get("demographic_cols", []) if col in final.columns]
    metric_cols = [col for col in mapping.get("metric_cols", []) if col in final.columns]
    id_col      = mapping.get("id_col", "")

    # Gelecek potansiyel
    if metric_cols:
        def norm(s): mn,mx=s.min(),s.max(); return (s-mn)/(mx-mn+1e-9)
        final["gelecek_potansiyel"] = (sum(norm(df[c]) for c in metric_cols if c in df.columns) / len(metric_cols) * 100).round(2)
    else:
        final["gelecek_potansiyel"] = final["potansiyel_skor"]

    df["gelecek_potansiyel"] = df.get("gelecek_potansiyel", df["potansiyel_skor"])
    df.to_csv(SCORED_PATH, index=False)

    optimal = final.nlargest(20, "gelecek_potansiyel")
    optimal.to_csv("data/optimal_targets.csv", index=False)

    # ── SÜREÇ RAPORU ──────────────────────────────────────────
    surec = {
        "ajan_ozeti": [],
        "toplam_bias_azalmasi": log.get("Düzeltilmiş Skorlama", {}).get("toplam_bias_azalmasi_puan", 0),
        "en_etkili_duzeltme": "",
    }
    for ajan_adi, ajan_log in log.items():
        oz = ajan_log.get("oz_degerlendirme", {})
        surec["ajan_ozeti"].append({
            "ajan": ajan_adi,
            "ozet": ajan_log.get("ozet", ""),
            "bias_katki_skoru": oz.get("bias_katki_skoru", 0),
            "fark_ettigi_bias": oz.get("sonuc", ""),
        })

    duzeltme_log = log.get("Düzeltilmiş Skorlama", {}).get("oncesi_sonrasi", {})
    if duzeltme_log:
        en_iyi = max(duzeltme_log.items(), key=lambda x: x[1].get("iyilesme_puan", 0))
        surec["en_etkili_duzeltme"] = f"'{en_iyi[0]}' bias'ı {en_iyi[1].get('iyilesme_puan',0):.1f} puan azaldı"

    # ── OPTİMAL KİTLE RAPORU ──────────────────────────────────
    display_cols = ([id_col] if id_col and id_col in final.columns else []) + demo_cols + metric_cols + ["potansiyel_skor", "gelecek_potansiyel"]
    display_cols = list(dict.fromkeys([c for c in display_cols if c in final.columns]))

    havuz = {
        "toplam_havuz": len(final),
        "elenme_orani": round((1 - len(final)/len(df))*100, 1),
        "minimum_skor": min_skor,
        "skor_dagilimi": final["skor_segmenti"].astype(str).value_counts().to_dict(),
        "optimal_20": optimal[display_cols].fillna("—").round({c:1 for c in metric_cols if c in optimal.columns}).to_dict(orient="records"),
    }
    for col in demo_cols:
        havuz[f"{col}_dagilimi"] = final[col].value_counts().to_dict()
    if len(final) > 0:
        havuz["ort_skor"] = round(float(final["potansiyel_skor"].mean()), 1)

    Path("data/optimal_targets.csv").write_text(
        optimal[display_cols].fillna("—").to_csv(index=False)
    )

    result = {
        "surec_raporu": surec,
        "optimal_kitle_raporu": havuz,
        "final_hedef_kitle": len(final),
        "elenme_orani_yuzde": round((1-len(final)/len(df))*100, 1),
        "skor_ortalama": round(float(final["potansiyel_skor"].mean()), 2) if len(final) > 0 else 0,
    }

    _log_agent("Final Optimizasyon", {"final_kitle": len(final), "surec_ozeti": surec["ajan_ozeti"]})
    return _dumps(result)


# ══════════════════════════════════════════════════════════════
#  YARDIMCI AJAN — COUNTERFACTUAL TEST
#  (Eleştirmen ajanın elindeki ek araç. "Eğer ağırlıkları değiştirseydik
#   bias nasıl olurdu?" sorusunu sayısal olarak yanıtlar.)
# ══════════════════════════════════════════════════════════════

@tool("Counterfactual Test Aracı")
def counterfactual_test(scenarios_json: str = "") -> str:
    """
    Skorlama ağırlıklarının alternatif senaryolarda bias üzerindeki
    etkisini test eder. JSON parametre verilmezse 4 default senaryo:
      - mevcut (Skorlama ajanından)
      - eşit_agirlik
      - harcama_20 (harcama %20, kalan eşit)
      - harcama_10 (harcama %10, kalan eşit)
    Her senaryo için max demografik fark + Yüksek+Prime kitle sayısı + grup
    ortalamaları döner. "En iyi" = en düşük max fark + non-zero Yüksek+Prime.
    """
    df          = _load(CLEANED_PATH)
    mapping     = _load_mapping()
    metric_cols = [c for c in mapping.get("metric_cols", [])
                   if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    demo_cols   = [c for c in mapping.get("demographic_cols", []) if c in df.columns]

    if not metric_cols:
        return _dumps({"hata": "Metrik kolon bulunamadı."})

    try:
        custom = json.loads(scenarios_json) if scenarios_json.strip() else None
    except Exception:
        custom = None

    n = len(metric_cols)
    spending_col = "aylik_ortalama_harcama" if "aylik_ortalama_harcama" in metric_cols else None

    log_scoring = _read_log().get("Skorlama", {}).get("kullanilan_agirliklar", {})

    senaryolar = dict(custom) if custom else {}
    if not senaryolar:
        if log_scoring:
            senaryolar["mevcut"] = {c: float(w) for c, w in log_scoring.items() if c in metric_cols}
        senaryolar["esit_agirlik"] = {c: round(1.0 / n, 4) for c in metric_cols}
        if spending_col and n > 1:
            kalan20 = round((1.0 - 0.2) / (n - 1), 4)
            senaryolar["harcama_20"] = {
                c: (0.2 if c == spending_col else kalan20) for c in metric_cols
            }
            kalan10 = round((1.0 - 0.1) / (n - 1), 4)
            senaryolar["harcama_10"] = {
                c: (0.1 if c == spending_col else kalan10) for c in metric_cols
            }

    def norm(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-9)

    sonuclar = {}
    for ad, w in senaryolar.items():
        skor = pd.Series(0.0, index=df.index)
        for c in metric_cols:
            skor = skor + norm(df[c]) * float(w.get(c, 1.0 / n))
        skor = (skor * 100).round(2)
        high_mask = skor >= 70

        grup_ortalamalari = {}
        max_fark = 0.0
        for col in demo_cols:
            try:
                # skor'u demografik gruba göre ortalama al
                tmp = pd.DataFrame({col: df[col].values, "_s": skor.values})
                g = tmp.groupby(col)["_s"].mean()
                fark = float(g.max() - g.min())
                grup_ortalamalari[col] = {str(k): round(float(v), 2) for k, v in g.to_dict().items()}
                if fark > max_fark:
                    max_fark = fark
            except Exception:
                pass

        yuksek_dagilim = {}
        for col in demo_cols:
            try:
                if int(high_mask.sum()) > 0:
                    dist = df.loc[high_mask, col].value_counts(normalize=True).round(3) * 100
                    yuksek_dagilim[col] = {str(k): round(float(v), 1) for k, v in dist.to_dict().items()}
            except Exception:
                pass

        sonuclar[ad] = {
            "agirliklar": {c: round(float(v), 4) for c, v in w.items()},
            "yuksek_prime_sayisi": int(high_mask.sum()),
            "max_demografik_fark": round(max_fark, 2),
            "grup_ortalamalari": grup_ortalamalari,
            "yuksek_kitle_dagilimi": yuksek_dagilim,
        }

    # En iyi senaryo: en düşük max_demografik_fark (Yüksek+Prime > 0)
    valid = {k: v for k, v in sonuclar.items() if v["yuksek_prime_sayisi"] > 0}
    if valid:
        en_iyi = min(valid, key=lambda k: valid[k]["max_demografik_fark"])
    else:
        en_iyi = next(iter(sonuclar), "—")

    karsilastirma_satirlari = []
    for ad, v in sonuclar.items():
        karsilastirma_satirlari.append({
            "senaryo": ad,
            "yuksek_prime_sayisi": v["yuksek_prime_sayisi"],
            "max_demografik_fark": v["max_demografik_fark"],
        })

    result = {
        "senaryolar": sonuclar,
        "karsilastirma_tablosu": karsilastirma_satirlari,
        "en_iyi_senaryo": en_iyi,
        "en_iyi_max_fark": sonuclar.get(en_iyi, {}).get("max_demografik_fark", 0),
        "ozet": (
            f"{len(sonuclar)} senaryo karşılaştırıldı. "
            f"En düşük demografik fark '{en_iyi}' senaryosunda "
            f"({sonuclar.get(en_iyi, {}).get('max_demografik_fark', 0):.1f} puan, "
            f"{sonuclar.get(en_iyi, {}).get('yuksek_prime_sayisi', 0)} Yüksek+Prime)."
        ),
    }

    _log_agent("Counterfactual Test", {
        "en_iyi_senaryo": en_iyi,
        "karsilastirma": karsilastirma_satirlari,
    })
    return _dumps(result)
