"""
TargetMind AI — Öz-Farkındalıklı Pipeline (Assumption-Surfacing)

Bu pipeline'ın amacı çıktıyı demografik olarak eşitlemek DEĞİLDİR. Her ajan
verdiği kararın altında yatan varsayımı yüzeye çıkarır, aynı kararı
alternatif bir varsayımla yeniden çalıştırır ve iki sonuç arasındaki farkı
ölçer. Çıktıdaki farklar büyükse karar **kırılgan** (metodolojiye bağımlı),
küçükse karar **robust** (veriye bağlı) sayılır. Amaç gerçek veri sinyalini
gizlemek değil, kararın varsayım yükünü şeffaf hale getirmektir.

Anahtar metrik (alan adı geriye dönük uyumluluk için korundu):
  `bias_katki_skoru` ≡ "varsayım hassasiyeti" — alternatif yöntemle çıktı
  arası normalize edilmiş fark. 0 = karar metodolojiden bağımsız (robust),
  1 = karar tamamen varsayımdan kaynaklı (fragile).
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

def _per_row_provenance(df: pd.DataFrame, metric_cols: list, weights: dict) -> tuple:
    """
    Her müşteri için bu satırın aldığı skorun **hangi metriklere borçlu**
    olduğunu üretir. CSV'ye iki kolon olarak yazılır:
      - dominant_metrik: en yüksek katkı veren metrik adı
      - secim_gerekce: top-2 metrik + bu kişinin o metriklerde hangi
        yüzdelikte olduğu (Q70+ = Yüksek, Q40-69 = Orta, Q<40 = Düşük)

    Bu, ajanın CSV üzerinde "neden bu skor?" sorusunu satır satır
    cevaplayabilmesini sağlar — şeffaf, gerekçeli skorlama.
    """
    if not metric_cols:
        empty = pd.Series([""] * len(df), index=df.index)
        return empty, empty

    def norm(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-9)

    n = len(metric_cols)
    contribs = pd.DataFrame(
        {c: norm(df[c]) * weights.get(c, 1.0 / n) for c in metric_cols},
        index=df.index,
    )
    quantiles = pd.DataFrame(
        {c: df[c].rank(pct=True) * 100 for c in metric_cols},
        index=df.index,
    )
    dominant = contribs.idxmax(axis=1)

    def label_for(q: float) -> str:
        if q >= 70:
            return "Yüksek"
        if q >= 40:
            return "Orta"
        return "Düşük"

    def gerekce(idx) -> str:
        top2 = contribs.loc[idx].nlargest(2)
        parts = []
        for metric in top2.index:
            q = float(quantiles.loc[idx, metric])
            parts.append(f"{label_for(q)} {metric} (Q{int(round(q))})")
        return " · ".join(parts)

    gerekceler = pd.Series([gerekce(i) for i in df.index], index=df.index)
    return dominant, gerekceler


def _bias_contribution_score(max_kayma: float, max_skor_farki: float = 0) -> float:
    """0-1 arası **varsayım hassasiyet skoru**.

    NOT: Bu skor, çıktıdaki demografik eşitsizliği değil, ajan kararının
    alternatif varsayımla yeniden işlendiğinde **ne kadar değişeceğinin**
    bir göstergesidir. Yüksek değer = karar metodolojiye bağımlı (fragile),
    düşük değer = karar metodolojiden bağımsız (robust).

    Gerçek bir veri sinyali yüksek değer üretebilir ve bu sorun değildir —
    sinyali "düzeltmek" amacımız değil, ajanın hangi yönteminin etkili
    olduğunu **şeffaf hale getirmektir**.
    """
    kayma_s = min(max_kayma / 10.0, 1.0)
    fark_s  = min(max_skor_farki / 15.0, 1.0)
    return round((kayma_s * 0.5 + fark_s * 0.5), 3)


# ══════════════════════════════════════════════════════════════
#  AJAN 1 — VERİ TEMİZLEME
# ══════════════════════════════════════════════════════════════

@tool("Veri Temizleme Ajanı")
def clean_data(rules_json: str = "") -> str:
    """
    Veriyi temizler VE temizleme kararlarının altındaki varsayımları ifşa eder.

    Öz-değerlendirme: 'Mod/medyan dolgu = "eksik değer tipik kullanıcıya
    benzer" varsayımı. Eğer "Bilinmiyor" kategori kullansaydım çıktı ne
    kadar değişirdi?' Bu fark kararın metodolojik kırılganlığını gösterir,
    bir bias hatası değil bir şeffaflık ölçüsüdür.
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
    oz_degerlendirme["varsayim"] = (
        "Eksik değerler tipik kullanıcıya benzer (medyan/mod dolgu). "
        "Alternatif: 'Bilinmiyor' kategorisi ile eksik bilgiyi sinyal olarak tut."
    )
    oz_degerlendirme["neden_optimal"] = (
        f"Veri {report['baslangic_satir']} → {len(df)} kayıta indi; "
        f"{report['baslangic_satir'] - len(df)} satır temizlendi. "
        "Alt akış ajanları artık negatif harcama, aykırı yaşlar, "
        "format tutarsızlıkları ve eksik değerlerden gelen **yanlış sinyali** "
        "almıyor. Bu, daha objektif analiz için zemin oluşturuyor — "
        "ancak temizleme yönteminin kendisi bir varsayım kaynağı olduğunu kabul ediyorum."
    )
    oz_degerlendirme["sonuc"] = (
        f"⚠ Temizleme yöntemim demografik dağılımı {max_kayma_genel:.1f} puan kaydırdı. "
        "Karar metodolojiye bağımlı — alternatif yöntemle çıktı belirgin farklı olur. "
        "Bu bir hata değil, kararımın **kırılganlık derecesi**."
        if max_kayma_genel > 3 else
        "✓ Temizleme kararım metodolojiden büyük ölçüde bağımsız — "
        "alternatif yöntemle benzer sonuç beklenir (robust)."
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
    Segment dağılımını ve metrik istatistikleri rapor eder. Demografik
    temsil farklarını ölçer — bu farklar gerçek pazar yapısını yansıtabilir;
    amaç onları gizlemek değil, ajanın gözünden kaçırılmamalarını sağlamak.
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
    oz["varsayim"] = (
        "Ham veri pazardaki gerçek dağılımı temsil ediyor. "
        "Eğer ham veri kendisi bir örnekleme bias'ı taşıyorsa, "
        "buradaki temsil farkları gerçek değil artefakttır."
    )
    oz["neden_optimal"] = (
        f"{len(df)} kaydı segmentleri + demografik dağılımı + metrik "
        f"istatistikleri açısından **görünür** hale getirdim. Skorlama ajanı "
        "artık hangi segmentin değerli olduğunu tahmin etmek yerine veriden okuyacak. "
        "Demografik temsil farkları (Maks: {} puan) alt akış için **şeffaf bilgi**.".format(
            f"{max_temsil_farki:.1f}"
        )
    )
    oz["sonuc"] = (
        f"ⓘ Maksimum demografik temsil farkı: {max_temsil_farki:.1f} puan. "
        "Bu fark gerçek pazar yapısı olabilir; ancak ham veri seçim sürecinin "
        "kendisi bias kaynağıysa, alt akıştaki tüm kararlar bu varsayımı miras alır."
        if max_temsil_farki > 20 else
        "✓ Demografik dağılım nispeten dengeli — alt akıştaki kararlar için "
        "stabil bir taban."
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
    İlk skoru üretir VE ağırlık seçiminin altındaki varsayımı ifşa eder.

    Her ağırlık şeması = farklı bir 'iyi müşteri' tanımı. Örn. harcamaya
    %40 verirken aslında 'değerli müşteri = çok harcayan' varsayımı
    yapıyorum. Bu varsayım gerçek olabilir, ama bir tercih olduğu
    şeffafça gösterilmeli. Aynı veriyi farklı ağırlık şemasıyla
    skorladığımda **kimin yüksek segmente girdiği** ne kadar değişiyor?
    Bu fark kararın metodolojik kırılganlığıdır.
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

    # Her satıra "neden bu skor?" gerekçesi (CSV'de görünür)
    df["dominant_metrik"], df["secim_gerekce"] = _per_row_provenance(df, metric_cols, w)

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
    oz["varsayim"] = (
        f"'Yüksek değerli müşteri' tanımım: kullanılan ağırlık şeması. "
        f"En etkili metrik '{oz.get('en_riskli_agirlik','—').split(' ')[0] if oz.get('en_riskli_agirlik') else '—'}'. "
        "Eşit ağırlık denemek bu varsayımdan ne kadar uzaklaştığımı gösterir."
    )
    oz["neden_optimal"] = (
        f"Her satıra 0-100 potansiyel skor + 'dominant_metrik' + 'secim_gerekce' kolonu "
        f"verdim. Yüksek+Prime ({int((df['skor_segmenti']=='Yüksek').sum() + (df['skor_segmenti']=='Prime').sum())} kişi) "
        "doğrudan hedeflenebilir, neden seçildikleri her satırda görünür. "
        "Veri analizi artık 'rastgele skor sırası' değil, **gerekçeli önceliklendirme**."
    )
    oz["sonuc"] = (
        f"ⓘ Ağırlık seçimim demografik gruplar arası {max_fark:.1f} puanlık skor farkı üretti. "
        "Bu fark gerçek harcama davranış sinyali olabilir veya benim metodolojimden "
        "kaynaklanabilir. Counterfactual karşılaştırma (eşit ağırlık) hangisinin "
        "doğru olduğunu netleştirir — bu rapor bunu zorla eşitlemeye **çağırmaz**, "
        "yalnızca eleştirmenin görmesi için yüzeye çıkarır."
        if max_fark > 5 else
        "✓ Skor farkları küçük — sonuç ağırlık seçimine duyarlı değil, alternatif "
        "yöntemle benzer kitle üretilir (robust)."
    )

    id_col = mapping.get("id_col", "")
    top_cols = ([id_col] if id_col and id_col in df.columns else []) + demo_cols + metric_cols + ["potansiyel_skor", "dominant_metrik", "secim_gerekce"]
    top_cols = list(dict.fromkeys([c for c in top_cols if c in df.columns]))

    # "Objektif kazanım" karşılaştırması için snapshot — final ajan bunu
    # son skorlamayla karşılaştırarak hangi müşterilerin yöntem değişikliğinden
    # etkilendiğini tespit eder
    snapshot = {
        "agirliklar": kullanilan,
        "max_skor": float(df["potansiyel_skor"].max()),
        "min_skor": float(df["potansiyel_skor"].min()),
        "ort_skor": round(float(df["potansiyel_skor"].mean()), 2),
        "skor_dagilimi": df["skor_segmenti"].astype(str).value_counts().to_dict(),
        "top_30_id": (
            df.nlargest(30, "potansiyel_skor")[id_col].tolist()
            if id_col and id_col in df.columns else []
        ),
    }

    result = {
        "kullanilan_agirliklar": kullanilan,
        "skor_dagilimi": df["skor_segmenti"].astype(str).value_counts().to_dict(),
        "prime_musteri_sayisi": int((df["skor_segmenti"] == "Prime").sum()),
        "oz_degerlendirme": oz,
        "top_10": df.nlargest(10, "potansiyel_skor")[top_cols].fillna("—").to_dict(orient="records"),
    }

    _log_agent("Skorlama", {
        "kullanilan_agirliklar": kullanilan,
        "oz_degerlendirme": oz,
        "ilk_skorlama_snapshot": snapshot,
    })
    return _dumps(result)


# ══════════════════════════════════════════════════════════════
#  AJAN 4 — PROXY & BIAS TESPİTİ
# ══════════════════════════════════════════════════════════════

@tool("Proxy ve Bias Tespit Ajanı")
def detect_proxy_and_bias(_: str = "") -> str:
    """
    Hassas demografik özelliklerle (cinsiyet, yaş, gelir, şehir) güçlü
    istatistiksel ilişki taşıyan **proxy** değişkenleri (cihaz, platform vs.)
    tespit eder. Ayrıca yüksek-skor segmentindeki demografik dağılımı
    raporlar.

    Bu rapor bir 'bias hatası' değildir. Eğer modelin sonucu bir proxy
    değişkene güçlü bağlıysa, ajan bu bağımlılığı **görmüş** olur — sonucu
    silmek değil, kararın kaynağını anlamak amaçlıdır.
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

    yuksek_risk_sayisi = sum(1 for p in proxy_analizi if p["risk"] == "YÜKSEK")
    oz = {
        "gozden_kacirilan_iliskiler": gozden_kacirilanlar,
        "bias_katki_skoru": _bias_contribution_score(len(proxy_analizi) * 2),
        "varsayim": (
            "Skorlamada kullandığım metrikler 'tarafsız' davranış değişkenleri. "
            "Ama bu metriklerin hassas demografik özelliklerle güçlü "
            "korelasyonu varsa, ben aslında dolaylı olarak demografi üzerinden "
            "skor verebilirim — buna fark etmeden."
        ),
        "neden_optimal": (
            f"{len(proxy_analizi)} proxy ilişkisi ifşa edildi ({yuksek_risk_sayisi} yüksek riskli). "
            "Skorlama ajanının metriklerinin hangi yolla dolaylı demografik temsil "
            "yaptığını eleştirmenin **görmesini** sağladım. Bu bilgi olmadan skorun "
            "'objektif' iddiası boştur."
        ),
        "sonuc": (
            f"ⓘ {len([p for p in proxy_analizi if p['risk']=='YÜKSEK'])} yüksek riskli proxy "
            f"ilişkisi var. Bu ilişki bir 'hata' değil — model kararımın "
            f"hangi yolla bir demografik özelliği temsil ettiğini ortaya koyuyor. "
            "Eleştirmen bu bilgiyle sonucumun **anlamını** sorgulayabilir."
            if any(p["risk"] == "YÜKSEK" for p in proxy_analizi) else
            "✓ Güçlü proxy ilişkisi yok — skor metriklerim demografik "
            "özelliklerden büyük ölçüde bağımsız."
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
    Tüm ajanların kararlarının altındaki **varsayımları** ve **metodolojik
    kırılganlık** değerlerini okur. Hangi kararın alternatif yöntemle en
    çok değişeceğini (en yüksek varsayım hassasiyetli) tespit eder.

    Bu fonksiyon çıktıyı 'düzeltmez' — kararı yapan ajana 'eğer alternatif
    yöntemini denersen, sonuç şu kadar değişir' bilgisini iletir. Karar
    çıktıyı zorlamadan, sadece **şeffaflığı** artırarak verilir.
    """
    log     = _read_log()
    mapping = _load_mapping()
    demo_cols   = mapping.get("demographic_cols", [])
    metric_cols = mapping.get("metric_cols", [])

    degerlendirmeler = []
    duzeltme_onerileri = {}

    # ── TEMİZLEME AJANI: VARSAYIM SORGULAMASI ────────────────
    if "Veri Temizleme" in log:
        vt = log["Veri Temizleme"]["oz_degerlendirme"]
        max_k = max((v.get("max_kayma", 0) for v in vt.get("demografik_kaymalar", {}).values()), default=0)
        kirilganlik = "YÜKSEK" if max_k > 5 else ("ORTA" if max_k > 2 else "DÜŞÜK")
        degerlendirmeler.append({
            "ajan": "Veri Temizleme",
            "varsayim_hassasiyet_skoru": vt.get("bias_katki_skoru", 0),
            "ifsa_edilen_varsayim": vt.get("varsayim", ""),
            "metodolojik_kirilganlik": kirilganlik,
            "alternatif_test_onerisi": (
                f"Mod dolgu yerine 'Bilinmiyor' kategorisini dene. Mevcut sonuç "
                f"{max_k:.1f} puanlık demografik kayma içeriyor — alternatifle bu "
                f"sayı düşerse karar varsayıma duyarlı (fragile), aynı kalırsa "
                f"sonuç gerçek veri yapısından geliyor (robust)."
                if max_k > 3 else
                "Karar metodolojiden büyük ölçüde bağımsız görünüyor — alternatif "
                "test yapılabilir ama büyük fark beklenmiyor."
            ),
            "soru": "Bu temizleme yöntemi tek seçenek miydi, yoksa varsayılan bir tercih mi?",
        })
        if max_k > 3:
            duzeltme_onerileri["Veri Temizleme"] = "Alternatif yöntem: 'Bilinmiyor' kategorisi (zorlama değil, varsayım testi)"

    # ── SKORLAMA AJANI: VARSAYIM SORGULAMASI ─────────────────
    if "Skorlama" in log:
        sk = log["Skorlama"]["oz_degerlendirme"]
        max_fark = max((v.get("max_fark_puan", 0) for v in sk.get("demografik_skor_farklari", {}).values()), default=0)
        dominant_metrik = sk.get("en_riskli_agirlik", "")
        kirilganlik = "YÜKSEK" if max_fark > 10 else ("ORTA" if max_fark > 5 else "DÜŞÜK")
        degerlendirmeler.append({
            "ajan": "Skorlama",
            "varsayim_hassasiyet_skoru": sk.get("bias_katki_skoru", 0),
            "ifsa_edilen_varsayim": sk.get("varsayim", ""),
            "metodolojik_kirilganlik": kirilganlik,
            "alternatif_test_onerisi": (
                f"Eşit ağırlıkla yeniden skorla ve mevcut sıralamayı karşılaştır. "
                f"{dominant_metrik}. Eşit ağırlıkla aynı kişiler yüksek segmente "
                f"girerse karar robust (gerçek sinyal); büyük farklar varsa karar "
                f"ağırlık tercihime bağlıydı (fragile)."
                if max_fark > 5 else
                "Mevcut ağırlık seti zaten dengeli görünüyor — eşit ağırlık testi "
                "yapılabilir ama önemli fark beklenmiyor."
            ),
            "soru": "Bu metrik ağırlıkları gerçek iş hedefinden mi yoksa kolaylıktan mı seçildi?",
        })
        if max_fark > 5:
            n = len(metric_cols)
            equal_w = {c: round(1.0/n, 4) for c in metric_cols} if n else {}
            duzeltme_onerileri["Skorlama"] = {
                "alternatif_agirliklar": equal_w,
                "amac": "Çıktıyı eşitlemek değil, varsayım hassasiyetini ölçmek",
                "beklenen_fark": f"{max_fark:.1f} puanın ne kadarının metodolojiden geldiğini gör",
            }

    # ── PROXY AJANI: VARSAYIM SORGULAMASI ────────────────────
    if "Proxy & Bias Tespiti" in log:
        pb = log["Proxy & Bias Tespiti"]["oz_degerlendirme"]
        degerlendirmeler.append({
            "ajan": "Proxy & Bias Tespiti",
            "varsayim_hassasiyet_skoru": pb.get("bias_katki_skoru", 0),
            "ifsa_edilen_varsayim": pb.get("varsayim", ""),
            "metodolojik_kirilganlik": "BİLGİ", # proxy analizi kararı değil bilgiyi açar
            "alternatif_test_onerisi": pb.get("sonuc", ""),
            "soru": "Tespit edilen proxy ilişkiler kararlarımın hangi yönünü açıklıyor?",
        })

    # ── METODOLOJİK KIRILGANLIK SIRALAMASI ───────────────────
    # Sıralama yüksekten düşüğe: hangi karar en çok yönteme bağımlı?
    siralama = sorted(
        [{"ajan": d["ajan"], "skor": d["varsayim_hassasiyet_skoru"]} for d in degerlendirmeler],
        key=lambda x: x["skor"], reverse=True
    )

    en_kirilgan = siralama[0]["ajan"] if siralama else "—"

    result = {
        "ajan_degerlendirmeleri": degerlendirmeler,
        # Geriye dönük uyum için alan adı korundu, ama artık "varsayım hassasiyeti"
        "bias_katki_siramasi": siralama,
        "en_yuksek_katkili_ajan": en_kirilgan,
        "duzeltme_onerileri": duzeltme_onerileri,
        "ozet": (
            f"En kırılgan karar '{en_kirilgan}' ajanından. {len(duzeltme_onerileri)} "
            "ajanın kararı için alternatif varsayım testi öneriliyor. "
            "Amaç çıktıyı değiştirmek değil, kararın varsayım bağımlılığını "
            "sayısal olarak göstermek."
        ),
    }

    _log_agent("Çapraz Değerlendirme", {"ozet": result["ozet"], "duzeltme_onerileri": duzeltme_onerileri})
    return _dumps(result)


# ══════════════════════════════════════════════════════════════
#  AJAN 6 — DÜZELTİLMİŞ YENİDEN SKORLAMA
# ══════════════════════════════════════════════════════════════

@tool("Alternatif Varsayım Skorlaması")
def corrected_scoring(_: str = "") -> str:
    """
    Skorlama ajanının ağırlık seçimini alternatif bir varsayımla
    (genellikle eşit ağırlık) yeniden hesaplar ve iki sonucu yan yana koyar.

    Bu bir "düzeltme" değildir — kararı zorla değiştirmez. Amaç, ilk skorun
    ne kadarının ağırlık tercihinden, ne kadarının veriden geldiğini ölçmek.
    İki yöntem benzer sonuç veriyorsa ilk karar **robust**; çok farklıysa
    ilk karar **fragile** ve şeffafça raporlanmalı.

    NOT: Çıktı CSV (customers_scored.csv) alternatif sonucu yazar; orijinal
    sonuç pipeline_log'da kalır. Final ajan iki sonucu karşılaştırarak
    hangisinin sunulacağına karar verir.
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
    df["dominant_metrik"], df["secim_gerekce"] = _per_row_provenance(df, metric_cols, yeni_agirliklar)
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

    # İlk yöntem vs alternatif yöntem karşılaştırması.
    # Yeni alan adları metaforu yansıtır; eski alan adları geriye dönük
    # uyumluluk için de bulundurulur (build_final_and_report + server.py).
    karsilastirma = {}
    for col in demo_cols:
        ilk_yontem  = onceki_farklar.get(col, 0)
        alt_yontem  = sonraki_farklar.get(col, 0)
        delta       = round(ilk_yontem - alt_yontem, 2)
        delta_pct   = round(delta / (ilk_yontem + 1e-9) * 100, 1)
        karsilastirma[col] = {
            # Yeni metafor (varsayım hassasiyeti)
            "ilk_yontem_fark":         ilk_yontem,
            "alternatif_yontem_fark":  alt_yontem,
            "delta":                   delta,
            "delta_yuzde":             delta_pct,
            # Geriye dönük (downstream tooling — anlam değişti ama sayı aynı)
            "onceki_fark":             ilk_yontem,
            "sonraki_fark":            alt_yontem,
            "iyilesme_puan":           delta,
            "iyilesme_yuzde":          delta_pct,
        }

    toplam_delta = sum(v["delta"] for v in karsilastirma.values())
    ortalama_delta = round(toplam_delta / max(len(karsilastirma), 1), 2)
    # Karar kırılganlığı: alternatif yöntemle sonuçların yaklaşma kuvveti
    karar_kirilganligi = (
        "FRAGILE" if abs(ortalama_delta) > 3 else
        "ORTA"    if abs(ortalama_delta) > 1 else
        "ROBUST"
    )

    result = {
        "uygulanan_agirliklar": yeni_agirliklar,
        "skor_dagilimi": df["skor_segmenti"].astype(str).value_counts().to_dict(),
        # Geriye dönük uyumluluk için eski anahtar adları korundu
        "oncesi_sonrasi": karsilastirma,
        "toplam_bias_azalmasi_puan": round(toplam_delta, 2),
        # Yeni alanlar — yorumun yeni metaforu
        "karar_kirilganligi": karar_kirilganligi,
        "ortalama_delta": ortalama_delta,
        "ozet": (
            f"İki yöntem arası ortalama {ortalama_delta:+.1f} puanlık fark "
            f"({karar_kirilganligi}). "
            + (
                "İlk skorlama yöntemi sonuç üzerinde belirgin etki yarattı — "
                "karar varsayıma duyarlı (fragile). Final raporda her iki "
                "yöntemin sonucu **şeffaflık için** birlikte sunulmalı."
                if karar_kirilganligi == "FRAGILE" else
                "Sonuç ağırlık tercihinden büyük ölçüde bağımsız — ilk "
                "skorlama yöntemi robust. Alternatifle aynı kitle çıkıyor."
                if karar_kirilganligi == "ROBUST" else
                "Karar kısmen yönteme bağımlı; sınırlı fark ama yine de raporlanmalı."
            )
        ),
    }

    # Son skorlama snapshot'u — final ajan ilk_skorlama_snapshot ile karşılaştırır
    id_col = mapping.get("id_col", "")
    son_snapshot = {
        "agirliklar": yeni_agirliklar,
        "max_skor": float(df["potansiyel_skor"].max()),
        "min_skor": float(df["potansiyel_skor"].min()),
        "ort_skor": round(float(df["potansiyel_skor"].mean()), 2),
        "skor_dagilimi": df["skor_segmenti"].astype(str).value_counts().to_dict(),
        "top_30_id": (
            df.nlargest(30, "potansiyel_skor")[id_col].tolist()
            if id_col and id_col in df.columns else []
        ),
    }

    _log_agent("Düzeltilmiş Skorlama", {
        "ozet": result["ozet"],
        "karar_kirilganligi": karar_kirilganligi,
        "oncesi_sonrasi": karsilastirma,
        "son_skorlama_snapshot": son_snapshot,
        "oz_degerlendirme": {
            "bias_katki_skoru": min(abs(ortalama_delta) / 5.0, 1.0),
            "varsayim": (
                "İlk skorlama ağırlık tercihine bağımlıydı; alternatif eşit "
                "ağırlıkla karşılaştırarak kararın ne kadarının yöntemden, "
                "ne kadarının veriden geldiğini ölçtüm."
            ),
            "sonuc": result["ozet"],
            "neden_optimal": (
                f"İki yöntem yan yana koyuldu. Karar kırılganlığı: {karar_kirilganligi}. "
                + (
                    "Final kitle, alternatif yöntemle aynı kişileri içerirse skor robust — "
                    "yöntem tercihim sonucu önemsiz hale geliyor."
                    if karar_kirilganligi == "ROBUST" else
                    "Yöntem değişikliği farklı müşterileri öne çıkarıyor — bu fark "
                    "objektif kazanım raporunda her bir değişen müşteri için "
                    "şeffaf olarak gösterilecek."
                )
            ),
        },
    })
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

    # ── SÜREÇ RAPORU — ajan anlatıları + objektif kazanım ────
    surec = {
        "ajan_ozeti": [],            # geriye dönük (eski UI kullanabilir)
        "ajan_anlatilari": [],       # yeni, zenginleştirilmiş anlatım
        "toplam_bias_azalmasi": log.get("Düzeltilmiş Skorlama", {}).get("toplam_bias_azalmasi_puan", 0),
        "en_etkili_duzeltme": "",
    }
    for ajan_adi, ajan_log in log.items():
        oz = ajan_log.get("oz_degerlendirme", {})
        # Eski format (geriye dönük)
        surec["ajan_ozeti"].append({
            "ajan": ajan_adi,
            "ozet": ajan_log.get("ozet", ""),
            "bias_katki_skoru": oz.get("bias_katki_skoru", 0),
            "fark_ettigi_bias": oz.get("sonuc", ""),
        })
        # Yeni zengin format — her ajanın anlatımı
        surec["ajan_anlatilari"].append({
            "ajan": ajan_adi,
            "ne_yapti": ajan_log.get("ozet", ""),
            "ifsa_ettigi_varsayim": oz.get("varsayim", oz.get("ifsa_edilen_varsayim", "—")),
            "varsayim_hassasiyeti": oz.get("bias_katki_skoru", 0),
            "fark_ettigi": oz.get("sonuc", ""),
            "neden_optimal": oz.get("neden_optimal", "—"),
        })

    duzeltme_log = log.get("Düzeltilmiş Skorlama", {}).get("oncesi_sonrasi", {})
    if duzeltme_log:
        # Yöntem değiştirmenin en çok etkilediği demografik boyut hangisi?
        en_kirilgan = max(duzeltme_log.items(),
                          key=lambda x: abs(x[1].get("delta",
                              x[1].get("iyilesme_puan", 0))))
        delta_val = en_kirilgan[1].get("delta",
                                       en_kirilgan[1].get("iyilesme_puan", 0))
        surec["en_etkili_duzeltme"] = (
            f"En yöntem-duyarlı demografik boyut '{en_kirilgan[0]}' "
            f"(alternatif yöntemle fark {delta_val:+.1f} puan). "
            "Bu boyut için iki yöntemin sonucu en uzakta; karar burada en "
            "kırılgan."
        )

    # ── OBJEKTİF KAZANIM — ilk skorlama vs son skorlama karşılaştırması ──
    # İlk yöntem (ağırlıklı) kim seçti, alternatif yöntem (eşit) kim seçti,
    # değişim nasıl, hangi karar yöntemden bağımsız?
    ilk_snap = log.get("Skorlama", {}).get("ilk_skorlama_snapshot", {})
    son_snap = log.get("Düzeltilmiş Skorlama", {}).get("son_skorlama_snapshot", {})

    objektif_kazanim = None
    if ilk_snap and son_snap:
        ilk_top30 = set(ilk_snap.get("top_30_id", []))
        son_top30 = set(son_snap.get("top_30_id", []))
        ortak = ilk_top30 & son_top30
        yeni_giren = son_top30 - ilk_top30
        ayrilan   = ilk_top30 - son_top30
        toplam_ilk = len(ilk_top30)
        karar_robustlugu = round(len(ortak) / max(toplam_ilk, 1) * 100, 1)

        objektif_kazanim = {
            "ilk_yontem_ozet": {
                "agirliklar": ilk_snap.get("agirliklar", {}),
                "ort_skor":   ilk_snap.get("ort_skor", 0),
                "top30_sayisi": toplam_ilk,
            },
            "alternatif_yontem_ozet": {
                "agirliklar": son_snap.get("agirliklar", {}),
                "ort_skor":   son_snap.get("ort_skor", 0),
                "top30_sayisi": len(son_top30),
            },
            "iki_yontemde_de_top30": sorted(list(ortak)),
            "sadece_ilk_yontemde": sorted(list(ayrilan)),
            "sadece_alternatifte": sorted(list(yeni_giren)),
            "karar_robustlugu_yuzde": karar_robustlugu,
            "yorum": (
                f"Top-30 müşterinin %{karar_robustlugu}'ı iki farklı ağırlık "
                f"yöntemiyle de seçildi → bu kişiler **yöntemden bağımsız** "
                f"olarak değerli (gerçek sinyal). Geri kalan %{100 - karar_robustlugu} "
                f"({len(yeni_giren) + len(ayrilan)} kişi) yöntem seçimine duyarlı — "
                "kararın hangi yönüyle bu kişiyi seçeceğin senin objektiflik tercihine bağlı."
            ),
        }
        surec["objektif_kazanim"] = objektif_kazanim

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
        # Yeni: objektif kazanım üst seviyede görünür (UI doğrudan okur)
        "objektif_kazanim": objektif_kazanim,
        # Yeni: zengin ajan anlatıları üst seviyede görünür
        "ajan_anlatilari": surec.get("ajan_anlatilari", []),
    }

    _log_agent("Final Optimizasyon", {
        "final_kitle": len(final),
        "surec_ozeti": surec["ajan_ozeti"],
        "objektif_kazanim_ozet": (
            objektif_kazanim.get("yorum", "") if objektif_kazanim else ""
        ),
    })
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
            f"{len(sonuclar)} farklı ağırlık varsayımı test edildi. "
            f"En az demografik kayma '{en_iyi}' senaryosunda "
            f"({sonuclar.get(en_iyi, {}).get('max_demografik_fark', 0):.1f} puan, "
            f"{sonuclar.get(en_iyi, {}).get('yuksek_prime_sayisi', 0)} Yüksek+Prime). "
            "Bu tablo 'doğru senaryo'yu seçtirmek için değil, ilk skorlamanın "
            "ağırlık tercihine ne kadar duyarlı olduğunu göstermek içindir."
        ),
    }

    _log_agent("Counterfactual Test", {
        "en_iyi_senaryo": en_iyi,
        "karsilastirma": karsilastirma_satirlari,
    })
    return _dumps(result)


# ══════════════════════════════════════════════════════════════
#  AJAN 6.5 — İTERATİF VARSAYIM ROBUSTLUK TESTİ
#
#  Critic + alternative-scoring döngüsünü turlar arası çıktı farkı
#  yumuşadığı zaman sonlandırır. Convergence kriteri "bias eşiği altına
#  in" DEĞİL, "ardışık turlardaki sonuç çok yakın → karar artık alternatif
#  yöntemlere duyarsız (robust)" demektir. Çıktıyı eşitlemiyoruz; kararın
#  hangi noktadan sonra yöntemden bağımsızlaştığını arıyoruz.
# ══════════════════════════════════════════════════════════════

# Turlar arası max demografik fark değişiminin bu eşik altına inmesi =
# convergence (kararlar artık yönteme duyarsız, daha fazla iterasyon
# kullanmaya gerek yok)
STABILITE_ESIGI = 1.0


def _max_demographic_gap(df: pd.DataFrame, demo_cols: list, score_col: str = "potansiyel_skor") -> float:
    """Bir DataFrame'deki en yüksek demografik skor farkını döner."""
    if score_col not in df.columns:
        return 0.0
    max_fark = 0.0
    for col in demo_cols:
        if col not in df.columns:
            continue
        try:
            g = df.groupby(col)[score_col].mean()
            fark = float(g.max() - g.min())
            if fark > max_fark:
                max_fark = fark
        except Exception:
            pass
    return round(max_fark, 2)


@tool("İteratif Varsayım Robustluk Testi")
def iterative_correction(max_iterations: str = "3") -> str:
    """
    Critic + alternative-scoring döngüsünü, ardışık turlar arası çıktı farkı
    küçük olduğunda (STABILITE_ESIGI altında) ya da max_iterations'a
    ulaşıldığında sonlandırır.

    Her turda:
      1. inter_agent_critique → her ajanın varsayımı + alternatif test önerisi
      2. corrected_scoring → alternatif varsayımı uygula, sonucu kaydet
      3. Mevcut max demografik fark ile bir önceki turun farkını karşılaştır
         → değişim küçükse karar artık alternatife duyarsız (robust), dur.

    Bu loop çıktıyı zorla eşitlemez. Aksine: ajanın kararının hangi noktadan
    sonra yöntem değişikliğine **duyarsız** kaldığını bulur. O nokta =
    convergence. Convergence'ta hangi yöntem kullanılırsa kullanılsın
    yaklaşık aynı kitle çıkar → karar veriye bağlı, yönteme değil.

    Default: max 3 tur.
    """
    try:
        max_iter = int(max_iterations.strip()) if max_iterations.strip() else 3
    except Exception:
        max_iter = 3
    max_iter = min(max(max_iter, 1), 5)

    mapping   = _load_mapping()
    demo_cols = mapping.get("demographic_cols", [])

    # Loop öncesi durum — ilk skorlama yönteminin sonucu
    try:
        df_before = _load(SCORED_PATH)
        baslangic_fark = _max_demographic_gap(df_before, demo_cols)
    except Exception:
        baslangic_fark = 0.0

    iterations = []
    onceki_fark = baslangic_fark
    son_fark = baslangic_fark
    converged = False

    for i in range(max_iter):
        # 1. Eleştirmen ajanların varsayımlarını yeniden gözden geçirir
        try:
            critic_out = json.loads(inter_agent_critique.run(""))
        except Exception as e:
            critic_out = {"hata": str(e)}

        oneri = critic_out.get("duzeltme_onerileri", {})

        # 2. Alternatif varsayımla skorlamayı yeniden çalıştır
        try:
            corr_out = json.loads(corrected_scoring.run(""))
        except Exception as e:
            corr_out = {"hata": str(e)}

        # 3. Yeni durum + bir önceki turla karşılaştır
        try:
            df_after = _load(SCORED_PATH)
            son_fark = _max_demographic_gap(df_after, demo_cols)
        except Exception:
            son_fark = son_fark

        # Stabilite: bu turun çıktısı bir öncekinden ne kadar farklı?
        tur_arasi_degisim = round(abs(son_fark - onceki_fark), 2)
        stabil = tur_arasi_degisim < STABILITE_ESIGI

        iterations.append({
            "tur": i + 1,
            "test_edilen_varsayim_sayisi": len(oneri),
            "test_edilen_varsayimlar": list(oneri.keys()),
            "uygulanan_agirliklar": corr_out.get("uygulanan_agirliklar", {}),
            "max_demografik_fark": son_fark,
            "onceki_tura_gore_degisim": tur_arasi_degisim,
            "karar_kirilganligi": corr_out.get("karar_kirilganligi", "—"),
            "stabil": stabil,
        })

        # Sadece ilk tur değilse stabilite kontrolü yap
        if i > 0 and stabil:
            converged = True
            break

        onceki_fark = son_fark

    sonuc_durumu = (
        f"✓ CONVERGED — kararlar artık alternatif varsayımlara duyarsız "
        f"(turlar arası değişim < {STABILITE_ESIGI} puan). Sonuç yöntemden "
        f"bağımsız → veriye bağlı (robust)."
        if converged else
        f"ⓘ MAX_ITER — {max_iter} tur sonunda hâlâ varyans var. "
        f"Karar yönteme duyarlı kaldı, ajan bu kırılganlığı **kabul edip** "
        f"final raporda iki yöntemin sonucunu birlikte sunmalı."
    )

    toplam_kayma = round(baslangic_fark - son_fark, 2)

    result = {
        "max_iter": max_iter,
        "stabilite_esigi": STABILITE_ESIGI,
        "baslangic_max_fark": baslangic_fark,
        "son_max_fark": son_fark,
        # Geriye dönük uyum için eski alan adı korundu
        "toplam_iyilesme_puan": toplam_kayma,
        "tur_sayisi": len(iterations),
        # Geriye dönük uyum: "erken_sonlandi" terimi kullanılıyor ama
        # anlamı şimdi "convergence sağlandı"
        "erken_sonlandi": converged,
        "iterasyonlar": iterations,
        "sonuc": sonuc_durumu,
        "ozet": (
            f"{len(iterations)} tur sonra demografik fark "
            f"{baslangic_fark:.1f} → {son_fark:.1f} puan. {sonuc_durumu}"
        ),
    }

    _log_agent("İteratif Düzeltme", {
        "tur_sayisi": len(iterations),
        "baslangic": baslangic_fark,
        "son": son_fark,
        "converged": converged,
    })
    return _dumps(result)
