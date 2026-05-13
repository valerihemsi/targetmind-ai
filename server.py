"""
Bias-Aware Data Optimization System — Web Sunucusu
Çalıştır: python server.py
"""
import os, json, queue, threading, warnings
import pandas as pd
import numpy as np
from pathlib import Path
from flask import Flask, Response, jsonify, send_from_directory, request

warnings.filterwarnings("ignore")

# .env yükle
_env = Path(__file__).parent / ".env"
if _env.exists():
    with open(_env) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

app = Flask(__name__, static_folder=".")

# Ensure data directory exists (important for Railway / cloud deployments)
Path("data").mkdir(exist_ok=True)

AGENT_NAMES = [
    "Veri Temizleme",
    "Segmentasyon",
    "İlk Skorlama",
    "Proxy & Bias Tespiti",
    "Çapraz Ajan Değerlendirmesi",
    "Alternatif Varsayım Testi",
    "Final Optimizasyon & Raporlar",
]

AGENT_DESCS = [
    "Veri temizleniyor; temizleme kararının altındaki varsayım ve metodolojik kırılganlığı ölçülüyor",
    "Segmentasyon yapılıyor; demografik temsil farkları (gerçek vs artefakt) şeffaflık için raporlanıyor",
    "İlk skor hesaplanıyor; ağırlık seçiminin altındaki varsayım ve kararın yöntem-duyarlılığı yüzeye çıkarılıyor",
    "Skorlama metriklerinin demografik özelliklerle dolaylı ilişkileri (proxy) ifşa ediliyor",
    "Ajanların varsayımları sayısal kanıtla sorgulanıyor; en kırılgan karar tespit ediliyor",
    "İlk skorlama alternatif varsayımla yeniden çalıştırılıyor; iki yöntem arası fark ölçülüyor (robust mu fragile mi?)",
    "Optimal havuz + şeffaflık raporu üretiliyor — kararın hangi yönü güvenli, hangi yönü varsayıma duyarlı?",
]


# ── Özet çıkarıcı ────────────────────────────────────────────────────────────

def _read_mapping() -> dict:
    mapping_path = Path("data/column_mapping.json")
    if mapping_path.exists():
        return json.loads(mapping_path.read_text())
    return {
        "id_col": "musteri_id",
        "demographic_cols": ["cinsiyet", "gelir_seviyesi", "yas"],
        "metric_cols": ["haftalik_oyun_saati", "aylik_ortalama_harcama",
                        "aylik_oturum_sayisi", "kampanya_tiklanma_orani", "arkadaslardan_referans"],
        "segment_col": "tercih_edilen_tur",
    }


def get_summary(idx: int) -> dict:
    """Ajan tamamlandıktan sonra ilgili CSV'yi okuyup kısa özet üretir."""
    try:
        mapping = _read_mapping()
        if idx == 0:
            raw     = pd.read_csv("data/customers.csv")
            cleaned = pd.read_csv("data/customers_cleaned.csv")
            elened  = len(raw) - len(cleaned)
            detail  = {"ham": len(raw), "temiz": len(cleaned)}
            for col in mapping.get("demographic_cols", [])[:1]:
                if col in cleaned.columns:
                    detail[col] = cleaned[col].value_counts().to_dict()
            return {
                "text": f"{len(raw)} → {len(cleaned)} kayıt · {elened} satır elendi",
                "level": "ok",
                "detail": detail,
            }
        elif idx == 1:
            cleaned = pd.read_csv("data/customers_cleaned.csv")
            seg_col = mapping.get("segment_col", "")
            if seg_col and seg_col in cleaned.columns:
                dist = cleaned[seg_col].value_counts().to_dict()
                text = f"Segment dağılımı: {', '.join(f'{k} ({v})' for k,v in list(dist.items())[:3])}"
            else:
                text = f"Temizlenmiş kayıt sayısı: {len(cleaned)}"
            return {"text": text, "level": "ok"}
        elif idx == 2:
            scored  = pd.read_csv("data/customers_scored.csv")
            dist    = scored["skor_segmenti"].value_counts().to_dict() if "skor_segmenti" in scored.columns else {}
            mapping = _read_mapping()
            demo_cols = [c for c in mapping.get("demographic_cols", []) if c in scored.columns]
            max_fark = 0.0
            for col in demo_cols:
                try:
                    g = scored.groupby(col)["potansiyel_skor"].mean()
                    max_fark = max(max_fark, float(g.max() - g.min()))
                except Exception:
                    pass
            level = "error" if max_fark > 5 else "ok"
            msg   = (f"⚠ Bias eşiği aşıldı — max grup farkı {max_fark:.1f} puan"
                     if max_fark > 5 else f"Bias eşiği OK ({max_fark:.1f} puan)")
            return {"text": msg, "level": level, "gelir_fark": round(max_fark, 1)}
        elif idx == 3:
            return {
                "text": "Temizleme kararları + skor dağılımı denetlendi",
                "level": "ok",
            }
        elif idx == 4:
            return {
                "text": "Proxy değişken ilişkileri ölçüldü",
                "level": "ok",
            }
        elif idx == 5:
            return {
                "text": "Counterfactual test tamamlandı, optimal ağırlık önerildi",
                "level": "ok",
            }
        elif idx == 6:
            if Path("data/final_targets.csv").exists():
                final   = pd.read_csv("data/final_targets.csv")
                cleaned = pd.read_csv("data/customers_cleaned.csv")
                elenme  = round((1 - len(final) / len(cleaned)) * 100, 1)
                return {
                    "text": f"Final hedef kitle: {len(final)} müşteri · %{elenme} elenme",
                    "level": "ok",
                    "final_count": len(final),
                }
    except Exception:
        pass
    return {"text": "Tamamlandı", "level": "ok"}


def get_dashboard_data() -> dict:
    """Tüm CSV'lerden dashboard için özet üretir. Column mapping'e göre generic çalışır."""
    try:
        mapping = _read_mapping()
        demo_cols   = mapping.get("demographic_cols", [])
        metric_cols = mapping.get("metric_cols", [])
        id_col      = mapping.get("id_col", "")

        raw     = pd.read_csv("data/customers.csv")
        cleaned = pd.read_csv("data/customers_cleaned.csv")
        scored  = pd.read_csv("data/customers_scored.csv")

        final_path = Path("data/final_targets.csv")
        final = pd.read_csv(final_path) if final_path.exists() else scored[scored.get("potansiyel_skor", pd.Series()) >= 50].copy()

        def bias_level(v):
            return "red" if v > 5 else "yellow" if v > 3 else "green"

        # Bias metrikleri — mapping'teki demografik kolonlar
        bias = {}
        for col in demo_cols:
            if col in scored.columns:
                try:
                    g = scored.groupby(col)["potansiyel_skor"].mean()
                    fark = round(float(g.max() - g.min()), 1) if len(g) > 1 else 0
                    bias[col] = {"fark": fark, "level": bias_level(fark)}
                except Exception:
                    pass

        # Geriye dönük uyumluluk — eski dashboard widget'ları için
        bias_compat = {
            "gelir":    bias.get(demo_cols[0], {"fark": 0, "level": "green"}) if demo_cols else {"fark": 0, "level": "green"},
            "cinsiyet": bias.get(demo_cols[1], {"fark": 0, "level": "green"}) if len(demo_cols) > 1 else {"fark": 0, "level": "green"},
            "yas":      bias.get(demo_cols[2], {"fark": 0, "level": "green"}) if len(demo_cols) > 2 else {"fark": 0, "level": "green"},
        }
        bias_compat.update(bias)

        # Skor dağılımı
        skor_dist = scored["skor_segmenti"].astype(str).value_counts().to_dict() if "skor_segmenti" in scored.columns else {}

        # Final müşteri listesi — sadece var olan kolonlar
        display_cols = ([id_col] if id_col and id_col in final.columns else [])
        display_cols += [c for c in demo_cols if c in final.columns]
        display_cols += [c for c in metric_cols if c in final.columns]
        if "potansiyel_skor" in final.columns:
            display_cols.append("potansiyel_skor")
        display_cols = list(dict.fromkeys(display_cols))

        num_cols_present = [c for c in display_cols if c in final.columns and pd.api.types.is_numeric_dtype(final[c])]
        round_dict = {c: 1 for c in num_cols_present}
        final_list = (final[display_cols].fillna("—").round(round_dict).to_dict(orient="records")
                      if display_cols else [])

        # Optimal liste
        optimal_list = []
        optimal_path = Path("data/optimal_targets.csv")
        if optimal_path.exists():
            opt = pd.read_csv(optimal_path)
            opt_cols = [c for c in display_cols + ["gelecek_potansiyel"] if c in opt.columns]
            opt_cols = list(dict.fromkeys(opt_cols))
            opt_num  = [c for c in opt_cols if c in opt.columns and pd.api.types.is_numeric_dtype(opt[c])]
            optimal_list = (opt[opt_cols].fillna("—").round({c: 1 for c in opt_num})
                            .to_dict(orient="records"))

        # Özet — hem generic hem eski dashboard alanlarını doldur
        ozet = {}
        if "potansiyel_skor" in final.columns and len(final) > 0:
            ozet["ort_skor"] = round(float(final["potansiyel_skor"].mean()), 1)
        for col in metric_cols[:3]:
            if col in final.columns and len(final) > 0:
                try:
                    ozet[f"ort_{col}"] = round(float(final[col].mean()), 1)
                except Exception:
                    pass
        # Geriye dönük uyumluluk — eski dashboard widget'ları için sabit alanlar
        for alias, src in [("ort_harcama", "aylik_ortalama_harcama"),
                           ("ort_oyun",    "haftalik_oyun_saati"),
                           ("ort_yas",     "yas")]:
            if src in final.columns and len(final) > 0:
                try:
                    ozet[alias] = round(float(final[src].mean()), 1)
                except Exception:
                    pass

        return {
            "funnel":       [len(raw), len(cleaned), len(final)],
            "bias":         bias_compat,
            "skor_dist":    skor_dist,
            "final_list":   final_list,
            "optimal_list": optimal_list,
            "ozet":         ozet,
        }
    except Exception as e:
        return {"error": str(e)}


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


def run_pipeline(msg_queue: queue.Queue, mode: str = "demo"):
    """
    7 Ajan öz-farkındalıklı pipeline.
    Her ajan kendi bias katkısını ölçer, eleştirmen çapraz değerlendirir.
    """
    import sys, traceback
    sys.path.insert(0, ".")

    def send_start(idx):
        msg_queue.put({"type": "agent_start", "agent_index": idx,
                       "agent_name": AGENT_NAMES[idx]})

    def send_done(idx, text, level="ok"):
        msg_queue.put({"type": "agent_done", "agent_index": idx,
                       "agent_name": AGENT_NAMES[idx],
                       "summary": {"text": text, "level": level}})

    try:
        from tools.data_tools import (
            clean_data, segmentation_analysis, score_customers,
            detect_proxy_and_bias, inter_agent_critique,
            corrected_scoring, build_final_and_report,
        )

        # ── Pipeline log'u sıfırla ────────────────────────────────
        Path("data/pipeline_log.json").write_text("{}")

        # ── Veri hazırla ──────────────────────────────────────────
        if mode == "demo":
            from data.generate_data import build_row, inject_issues
            rows = [build_row(i + 1) for i in range(300)]
            df = pd.DataFrame(rows)
            df = inject_issues(df)
            df = df.sample(frac=1, random_state=99).reset_index(drop=True)
            df.to_csv("data/customers.csv", index=False)
            Path("data/column_mapping.json").write_text(json.dumps({
                "id_col": "musteri_id",
                "demographic_cols": ["cinsiyet", "gelir_seviyesi", "yas"],
                "metric_cols": ["haftalik_oyun_saati", "aylik_ortalama_harcama",
                                "aylik_oturum_sayisi", "kampanya_tiklanma_orani",
                                "arkadaslardan_referans"],
                "segment_col": "tercih_edilen_tur",
            }, ensure_ascii=False, indent=2))

        count = len(pd.read_csv("data/customers.csv"))
        msg_queue.put({"type": "data_ready", "count": count})

        # ── AJAN 1: Veri Temizleme ────────────────────────────────
        send_start(0)
        r1 = json.loads(clean_data.run(json.dumps({"yas_min": 13, "yas_max": 75})))
        oz1 = r1.get("oz_degerlendirme", {})
        hassasiyet1 = oz1.get("bias_katki_skoru", 0)
        level1 = "warn" if hassasiyet1 > 0.3 else "ok"
        send_done(0,
            f"{r1['baslangic_satir']} → {r1['bitis_satir']} kayıt  ·  "
            f"{r1['elenen_satir']} elendi  ·  Varsayım hassasiyeti: {hassasiyet1:.2f}",
            level1)

        # ── AJAN 2: Segmentasyon ──────────────────────────────────
        send_start(1)
        r2 = json.loads(segmentation_analysis.run(""))
        oz2 = r2.get("oz_degerlendirme", {})
        hassasiyet2 = oz2.get("bias_katki_skoru", 0)
        col_count = len(r2.get("metrik_istatistikler", {}))
        level2 = "warn" if hassasiyet2 > 0.3 else "ok"
        send_done(1,
            f"{r2['toplam_kayit']} kayıt  ·  {col_count} metrik  ·  "
            f"Varsayım hassasiyeti: {hassasiyet2:.2f}",
            level2)

        # ── AJAN 3: İlk Skorlama ──────────────────────────────────
        send_start(2)
        r3 = json.loads(score_customers.run(""))
        oz3 = r3.get("oz_degerlendirme", {})
        hassasiyet3 = oz3.get("bias_katki_skoru", 0)
        dist3 = r3.get("skor_dagilimi", {})
        level3 = "warn" if hassasiyet3 > 0.3 else "ok"
        send_done(2,
            f"Prime: {dist3.get('Prime',0)}  Yüksek: {dist3.get('Yüksek',0)}  "
            f"Orta: {dist3.get('Orta',0)}  ·  Varsayım hassasiyeti: {hassasiyet3:.2f}",
            level3)

        # ── AJAN 4: Proxy & Bias Tespiti ──────────────────────────
        send_start(3)
        r4 = json.loads(detect_proxy_and_bias.run(""))
        oz4 = r4.get("oz_degerlendirme", {})
        hassasiyet4 = oz4.get("bias_katki_skoru", 0)
        yuksek_proxy = r4.get("yuksek_riskli_proxy", 0)
        toplam_proxy = len(r4.get("proxy_analizi", []))
        level4 = "warn" if yuksek_proxy > 0 else "ok"
        send_done(3,
            f"{toplam_proxy} proxy ilişkisi  ·  {yuksek_proxy} yüksek riskli  ·  "
            f"Varsayım hassasiyeti: {hassasiyet4:.2f}",
            level4)

        # ── AJAN 5: Çapraz Ajan Değerlendirmesi ───────────────────
        send_start(4)
        r5 = json.loads(inter_agent_critique.run(""))
        en_kirilgan = r5.get("en_yuksek_katkili_ajan", "—")
        test_sayisi = len(r5.get("duzeltme_onerileri", {}))
        level5 = "warn" if test_sayisi > 0 else "ok"
        send_done(4,
            f"En kırılgan karar: {en_kirilgan}  ·  {test_sayisi} alternatif test önerildi",
            level5)

        # ── AJAN 6: Alternatif Varsayım Testi ─────────────────────
        send_start(5)
        r6 = json.loads(corrected_scoring.run(""))
        kirilganlik = r6.get("karar_kirilganligi", "—")
        ortalama_delta = r6.get("ortalama_delta", 0)
        level6 = "warn" if kirilganlik == "FRAGILE" else "ok"
        send_done(5,
            f"Karar kırılganlığı: {kirilganlik}  ·  Yöntemler arası "
            f"ortalama fark: {ortalama_delta:+.1f} puan",
            level6)

        # ── AJAN 7: Final Optimizasyon & Raporlar ─────────────────
        send_start(6)
        r7 = json.loads(build_final_and_report.run(json.dumps({"min_skor": 50})))
        final_count = r7.get("final_hedef_kitle", 0)
        elenme = r7.get("elenme_orani_yuzde", 0)
        # Raporları ayrı dosyalara kaydet
        Path("data/surec_raporu.json").write_text(
            json.dumps(r7.get("surec_raporu", {}), ensure_ascii=False, indent=2)
        )
        Path("data/optimal_kitle_raporu.json").write_text(
            json.dumps(r7.get("optimal_kitle_raporu", {}), ensure_ascii=False, indent=2)
        )
        send_done(6,
            f"Final havuz: {final_count} müşteri  ·  %{elenme} elenme  ·  2 rapor üretildi",
            "ok")

        # ── Dashboard ─────────────────────────────────────────────
        msg_queue.put({"type": "complete", "dashboard": get_dashboard_data()})

    except Exception as e:
        msg_queue.put({"type": "error",
                       "message": str(e),
                       "detail": traceback.format_exc()})


@app.route("/api/demo", methods=["POST"])
def run_demo():
    msg_queue: queue.Queue = queue.Queue()
    mode = (request.json or {}).get("mode", "demo")

    threading.Thread(
        target=run_pipeline, args=(msg_queue, mode), daemon=True
    ).start()

    def stream():
        while True:
            msg = msg_queue.get()
            yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
            if msg["type"] in ("complete", "error"):
                break

    return Response(
        stream(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/upload", methods=["POST"])
def upload_csv():
    """Kullanıcının yüklediği CSV'yi kaydeder, kolon tiplerini otomatik tespit eder."""
    file = request.files.get("file")
    if not file or not file.filename.endswith(".csv"):
        return jsonify({"error": "Geçerli bir CSV dosyası yükleyin."}), 400
    try:
        df = pd.read_csv(file)
        if len(df) < 10:
            return jsonify({"error": "CSV en az 10 satır içermelidir."}), 400
        df.to_csv("data/customers.csv", index=False)

        # Kolon tip tespiti
        numeric_cols   = df.select_dtypes(include="number").columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # Olası kolon rolleri için ipuçları
        def guess_role(col):
            c = col.lower()
            if any(k in c for k in ["id", "key", "ref", "code", "no"]):
                return "id"
            if any(k in c for k in ["age", "yas", "birth", "dob"]):
                return "age"
            if any(k in c for k in ["gender", "cinsiyet", "sex"]):
                return "demographic"
            if any(k in c for k in ["income", "gelir", "salary", "revenue", "spend", "harcama", "price", "value", "amount", "sales"]):
                return "metric"
            if any(k in c for k in ["date", "period", "time", "tarih"]):
                return "date"
            if any(k in c for k in ["region", "city", "country", "sehir", "location", "area"]):
                return "demographic"
            if any(k in c for k in ["category", "group", "segment", "type", "tur", "sektor"]):
                return "segment"
            if col in numeric_cols:
                return "metric"
            return "other"

        columns_info = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            unique = int(df[col].nunique())
            sample = df[col].dropna().head(3).tolist()
            columns_info.append({
                "name": col,
                "dtype": dtype,
                "unique": unique,
                "sample": sample,
                "suggested_role": guess_role(col),
            })

        return jsonify({
            "ok": True,
            "rows": len(df),
            "columns_info": columns_info,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
        })
    except Exception as e:
        return jsonify({"error": f"Dosya okunamadı: {e}"}), 400


@app.route("/api/set-mapping", methods=["POST"])
def set_mapping():
    """Kullanıcının kolon eşlemesini kaydeder."""
    mapping = request.json
    if not mapping:
        return jsonify({"error": "Eşleme verisi gerekli."}), 400
    required = ["id_col", "metric_cols"]
    for r in required:
        if r not in mapping or not mapping[r]:
            return jsonify({"error": f"'{r}' alanı zorunludur."}), 400
    import json as _json
    Path("data/column_mapping.json").write_text(
        _json.dumps(mapping, ensure_ascii=False, indent=2)
    )
    return jsonify({"ok": True, "mapping": mapping})



@app.route("/api/report")
def generate_report():
    """Analiz raporunu HTML olarak üretir."""
    try:
        mapping  = _read_mapping()
        scored   = pd.read_csv("data/customers_scored.csv")
        cleaned  = pd.read_csv("data/customers_cleaned.csv")
        raw      = pd.read_csv("data/customers.csv")
        final    = pd.read_csv("data/final_targets.csv") if Path("data/final_targets.csv").exists() else pd.DataFrame()

        demo_cols   = [c for c in mapping.get("demographic_cols", []) if c in scored.columns]
        metric_cols = [c for c in mapping.get("metric_cols", [])      if c in scored.columns]

        # Bias analizi
        bias_rows = []
        for col in demo_cols:
            try:
                g = scored.groupby(col)["potansiyel_skor"].mean().round(2)
                fark = round(float(g.max() - g.min()), 2)
                seviye = "🔴 Kritik" if fark > 10 else "🟡 Yüksek" if fark > 5 else "🟢 Kabul edilebilir"
                bias_rows.append({"kolon": col, "fark": fark, "seviye": seviye,
                                  "gruplar": g.to_dict()})
            except Exception:
                pass

        # Proxy analizi
        proxy_rows = []
        id_col = mapping.get("id_col", "")
        skip   = set(demo_cols + [id_col, "skor_segmenti", "potansiyel_skor"])
        proxies = [c for c in scored.select_dtypes(include=["object","category"]).columns if c not in skip]
        for proxy in proxies[:8]:
            for sens in demo_cols[:3]:
                try:
                    ct  = pd.crosstab(scored[proxy], scored[sens])
                    n, k = len(scored), min(ct.shape)
                    chi2 = sum(((ct[c] - ct[c].sum()*ct.sum(axis=1)/n)**2 / (ct[c].sum()*ct.sum(axis=1)/n + 1e-9)).sum() for c in ct.columns)
                    cv = round(float(np.sqrt(chi2 / (n*(k-1)+1e-9))), 3) if k > 1 else 0
                    if cv > 0.15:
                        proxy_rows.append({"proxy": proxy, "hassas": sens, "cramer_v": cv,
                                           "risk": "Yüksek" if cv > 0.35 else "Orta"})
                except Exception:
                    pass

        # Skor dağılımı
        skor_dist = scored["skor_segmenti"].astype(str).value_counts().to_dict() if "skor_segmenti" in scored.columns else {}

        # Metrik korelasyonlar
        corr_rows = []
        for col in metric_cols:
            try:
                corr = round(float(scored[col].corr(scored["potansiyel_skor"])), 3)
                corr_rows.append({"metrik": col, "korelasyon": corr})
            except Exception:
                pass
        corr_rows.sort(key=lambda x: abs(x["korelasyon"]), reverse=True)

        from datetime import datetime
        now = datetime.now().strftime("%d %B %Y, %H:%M")

        def bias_color(fark):
            if fark > 10: return "#ef4444"
            if fark > 5:  return "#f97316"
            return "#22c55e"

        def render_group_table(gruplar):
            rows = "".join(f"<tr><td>{k}</td><td style='text-align:right;font-weight:600'>{v}</td></tr>"
                           for k, v in gruplar.items())
            return f"<table class='inner'><tbody>{rows}</tbody></table>"

        bias_html = ""
        for b in bias_rows:
            clr = bias_color(b["fark"])
            bias_html += f"""
            <div class='bias-block'>
              <div class='bias-header'>
                <span class='bias-col'>{b['kolon']}</span>
                <span class='bias-fark' style='color:{clr}'>{b['fark']} puan fark</span>
                <span class='bias-sev'>{b['seviye']}</span>
              </div>
              <p class='bias-desc'>
                {'Bu demografik grupta skor eşitsizliği eşiği (5 puan) aşıyor.' if b['fark'] > 5
                 else 'Bu demografik grupta skor dağılımı kabul edilebilir düzeyde.'}
                Gruplar arası maksimum fark: <strong>{b['fark']} puan</strong>.
              </p>
              {render_group_table(b['gruplar'])}
            </div>"""

        proxy_html = ""
        for p in proxy_rows[:8]:
            proxy_html += f"""<tr>
              <td>{p['proxy']}</td><td>{p['hassas']}</td>
              <td style='text-align:center'>{p['cramer_v']}</td>
              <td style='text-align:center;color:{"#ef4444" if p["risk"]=="Yüksek" else "#f97316"};font-weight:600'>{p['risk']}</td>
            </tr>"""

        corr_html = ""
        for c in corr_rows:
            w = abs(c["korelasyon"]) * 100
            clr = "#ef4444" if abs(c["korelasyon"]) > 0.5 else "#f97316" if abs(c["korelasyon"]) > 0.3 else "#22c55e"
            corr_html += f"""
            <div class='corr-row'>
              <span class='corr-label'>{c['metrik']}</span>
              <div class='corr-bar-bg'><div class='corr-bar' style='width:{w:.0f}%;background:{clr}'></div></div>
              <span class='corr-val' style='color:{clr}'>{c['korelasyon']:+.3f}</span>
            </div>"""

        dist_html = "".join(
            f"<div class='dist-item'><span class='dist-label'>{k}</span><span class='dist-val'>{v}</span></div>"
            for k, v in sorted(skor_dist.items(), key=lambda x: ["Düşük","Orta","Yüksek","Prime","nan"].index(x[0]) if x[0] in ["Düşük","Orta","Yüksek","Prime","nan"] else 99))

        html = f"""<!DOCTYPE html>
<html lang='tr'><head><meta charset='UTF-8'>
<title>TargetMind AI — Bias Raporu</title>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family: system-ui, sans-serif; background:#f7f6f3; color:#1c1917; line-height:1.6; }}
  .page {{ max-width:860px; margin:0 auto; padding:48px 32px; }}
  h1 {{ font-size:28px; font-weight:800; letter-spacing:-0.03em; margin-bottom:4px; }}
  h2 {{ font-size:13px; letter-spacing:.15em; color:#78716c; text-transform:uppercase; margin:40px 0 16px; }}
  .meta {{ font-size:13px; color:#78716c; margin-bottom:40px; }}
  .card {{ background:#fff; border:1px solid #e2dfd8; border-radius:12px; padding:24px; margin-bottom:16px; }}
  .stats {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:8px; }}
  .stat {{ background:#fff; border:1px solid #e2dfd8; border-radius:10px; padding:16px; text-align:center; }}
  .stat-val {{ font-size:24px; font-weight:800; }}
  .stat-lbl {{ font-size:11px; color:#78716c; margin-top:4px; }}
  .bias-block {{ border:1px solid #e2dfd8; border-radius:10px; padding:18px; margin-bottom:12px; }}
  .bias-header {{ display:flex; align-items:center; gap:16px; margin-bottom:10px; flex-wrap:wrap; }}
  .bias-col {{ font-weight:700; font-size:15px; }}
  .bias-fark {{ font-size:20px; font-weight:800; }}
  .bias-sev {{ font-size:13px; color:#78716c; }}
  .bias-desc {{ font-size:13px; color:#57534e; margin-bottom:12px; }}
  table.inner {{ width:100%; border-collapse:collapse; font-size:13px; }}
  table.inner td {{ padding:5px 8px; border-bottom:1px solid #f0eeea; }}
  table.inner tr:last-child td {{ border-bottom:none; }}
  table.proxy {{ width:100%; border-collapse:collapse; font-size:13px; }}
  table.proxy th {{ background:#f7f6f3; padding:8px 12px; text-align:left; font-size:11px; letter-spacing:.08em; color:#78716c; }}
  table.proxy td {{ padding:8px 12px; border-bottom:1px solid #f0eeea; }}
  .corr-row {{ display:flex; align-items:center; gap:12px; margin-bottom:10px; }}
  .corr-label {{ font-size:13px; width:200px; flex-shrink:0; }}
  .corr-bar-bg {{ flex:1; height:8px; background:#f0eeea; border-radius:4px; overflow:hidden; }}
  .corr-bar {{ height:100%; border-radius:4px; transition:width .3s; }}
  .corr-val {{ font-size:13px; font-weight:700; width:56px; text-align:right; }}
  .dist-item {{ display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid #f0eeea; font-size:14px; }}
  .dist-val {{ font-weight:700; }}
  .conclusion {{ background:#fff; border:2px solid #4338ca; border-radius:12px; padding:24px; }}
  .conclusion p {{ font-size:14px; color:#57534e; line-height:1.7; margin-bottom:10px; }}
  .conclusion p:last-child {{ margin-bottom:0; }}
  @media print {{ body {{ background:#fff; }} .page {{ padding:24px; }} }}
</style></head>
<body><div class='page'>

  <h1>TargetMind AI — Bias Analiz Raporu</h1>
  <p class='meta'>Üretildi: {now} &nbsp;·&nbsp; Veri: {len(raw)} ham kayıt &nbsp;·&nbsp; Final hedef: {len(final)} kayıt</p>

  <h2>Genel Özet</h2>
  <div class='stats'>
    <div class='stat'><div class='stat-val'>{len(raw)}</div><div class='stat-lbl'>Ham kayıt</div></div>
    <div class='stat'><div class='stat-val'>{len(cleaned)}</div><div class='stat-lbl'>Temizlenmiş</div></div>
    <div class='stat'><div class='stat-val'>{len(final)}</div><div class='stat-lbl'>Final hedef kitle</div></div>
    <div class='stat'><div class='stat-val'>{len(bias_rows)}</div><div class='stat-lbl'>Analiz edilen demografik grup</div></div>
  </div>

  <h2>Skor Dağılımı</h2>
  <div class='card'>{dist_html}</div>

  <h2>Demografik Bias Analizi</h2>
  {bias_html if bias_html else "<p style='color:#78716c;font-size:14px'>Demografik bias tespit edilmedi.</p>"}

  <h2>Metrik — Skor Korelasyonları</h2>
  <div class='card'>
    <p style='font-size:13px;color:#78716c;margin-bottom:16px'>
      Korelasyonu yüksek (> 0.5) metrikler skoru domine ediyor — bu da o metrikle ilişkili demografik gruplarda bias riski yaratır.
    </p>
    {corr_html}
  </div>

  {'<h2>Proxy Değişken Riski</h2><div class="card"><table class="proxy"><thead><tr><th>Değişken</th><th>Hassas Değişken</th><th style="text-align:center">Cramér\'s V</th><th style="text-align:center">Risk</th></tr></thead><tbody>' + proxy_html + '</tbody></table></div>' if proxy_html else ''}

  <h2>Sonuç ve Yeniden Değerlendirme</h2>
  <div class='conclusion'>
    <p>
      Sistem, demografik gruplar arasındaki skor farklılıklarını tespit ederek bias kaynaklarını belirledi.
      {'Gelir, cinsiyet ve yaş gruplarında anlamlı skor eşitsizlikleri saptandı.' if any(b['fark'] > 5 for b in bias_rows) else 'Analiz edilen demografik gruplarda skor eşitsizliği kabul edilebilir düzeyde bulundu.'}
    </p>
    <p>
      Counterfactual test farklı metrik ağırlık senaryolarını karşılaştırdı. Bias'ı en aza indiren
      ağırlık konfigürasyonu tespit edilerek skorlama bu ağırlıklarla yeniden hesaplandı.
      {'Proxy değişken analizi ' + str(len(proxy_rows)) + ' dolaylı bias riski tespit etti.' if proxy_rows else ''}
    </p>
    <p>
      Final hedef kitle <strong>{len(final)} kayıt</strong> ile oluşturuldu
      ({round((1-len(final)/max(len(cleaned),1))*100,1)}% elenme oranı).
      Bu liste bias düzeltilmiş skorlar üzerinden filtrelenmiştir.
    </p>
  </div>

</div></body></html>"""

        return Response(html, mimetype="text/html",
                        headers={"Content-Disposition": "inline; filename=bias-report.html"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/template")
def download_template():
    """Boş CSV şablonunu indirir."""
    import io
    cols = [
        "musteri_id", "yas", "cinsiyet", "sehir", "gelir_seviyesi",
        "cihaz_turu", "platform", "haftalik_oyun_saati", "tercih_edilen_tur",
        "uygulama_ici_satin_alma", "aylik_ortalama_harcama", "son_aktivite_gun",
        "abonelik_turu", "kampanya_tiklanma_orani", "arkadaslardan_referans",
        "aylik_oturum_sayisi", "tamamlanma_orani",
    ]
    buf = io.StringIO()
    buf.write(",".join(cols) + "\n")
    buf.write("C0001,28,Kadın,İstanbul,Orta,Mobil,Android,12.5,Strateji,Evet,150.0,7,Premium,0.65,2,20,0.75\n")
    buf.seek(0)
    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=targetmind_template.csv"},
    )


@app.route("/api/process-report")
def process_report():
    """Ajan süreç raporunu HTML olarak üretir — öz-değerlendirme + çapraz eleştiri."""
    try:
        log_path = Path("data/pipeline_log.json")
        log = json.loads(log_path.read_text()) if log_path.exists() else {}

        from datetime import datetime
        now = datetime.now().strftime("%d %B %Y, %H:%M")

        def bias_color(s):
            return "#ef4444" if s > 0.5 else "#f97316" if s > 0.3 else "#22c55e"

        # Ajan kartları
        ajan_html = ""
        for ajan_adi, ajan_log in log.items():
            oz = ajan_log.get("oz_degerlendirme", {})
            skor = oz.get("bias_katki_skoru", 0)
            ozet = ajan_log.get("ozet", "")
            sonuc = oz.get("sonuc", "")
            clr = bias_color(skor)
            icon = "⚠" if skor > 0.3 else "✓"

            detail_items = ""
            for key in ["demografik_kaymalar", "demografik_skor_farklari", "temsil_analizi"]:
                d = oz.get(key, {})
                for col, info in d.items():
                    if not isinstance(info, dict): continue
                    fark = info.get("max_kayma", info.get("max_fark_puan", info.get("temsil_farki_puan", 0)))
                    detail_items += f"<li><strong>{col}</strong>: {fark:.1f} puan</li>"

            ajan_html += f"""
            <div class='agent-card'>
              <div class='agent-header'>
                <span class='agent-icon' style='color:{clr}'>{icon}</span>
                <span class='agent-name'>{ajan_adi}</span>
                <span class='bias-badge' style='border-color:{clr};color:{clr}'>
                  Bias Katkı: {skor:.3f}
                </span>
              </div>
              {"<p class='agent-ozet'>" + ozet + "</p>" if ozet else ""}
              {"<ul class='detail-list'>" + detail_items + "</ul>" if detail_items else ""}
              {"<p class='agent-sonuc' style='color:" + clr + "'>" + sonuc + "</p>" if sonuc else ""}
            </div>"""

        # Çapraz değerlendirme
        critique_html = "<p style='color:var(--muted);font-size:13px'>Henüz çalıştırılmadı.</p>"
        if "Çapraz Değerlendirme" in log:
            cv = log["Çapraz Değerlendirme"]
            rows = ""
            for d in cv.get("ajan_degerlendirmeleri", []):
                clr = "#22c55e" if d.get("dogrulama") == "ONAYLANDI" else "#f97316"
                rows += f"""<tr>
                  <td>{d['ajan']}</td>
                  <td style='text-align:center;font-weight:700;color:{clr}'>{d.get('dogrulama','')}</td>
                  <td>{d.get('elestiri','')}</td>
                  <td style='color:#f97316'>{d.get('duzeltme') or '—'}</td>
                </tr>"""
            critique_html = f"""
            <p style='font-size:13px;color:#57534e;margin-bottom:16px'>{cv.get('ozet','')}</p>
            <table class='proxy'>
              <thead><tr>
                <th>Ajan</th><th>Doğrulama</th><th>Eleştiri</th><th>Düzeltme</th>
              </tr></thead>
              <tbody>{rows}</tbody>
            </table>"""

        # Düzeltme sonuçları
        correction_html = "<p style='color:var(--muted);font-size:13px'>Henüz çalıştırılmadı.</p>"
        if "Düzeltilmiş Skorlama" in log:
            ds = log["Düzeltilmiş Skorlama"]
            rows = ""
            for col, info in ds.get("oncesi_sonrasi", {}).items():
                if not isinstance(info, dict): continue
                iyilesme = info.get("iyilesme_puan", 0)
                clr = "#22c55e" if iyilesme > 0 else "#ef4444"
                rows += f"""<tr>
                  <td><strong>{col}</strong></td>
                  <td style='text-align:center'>{info.get('onceki_fark',0):.1f} puan</td>
                  <td style='text-align:center'>→</td>
                  <td style='text-align:center'>{info.get('sonraki_fark',0):.1f} puan</td>
                  <td style='text-align:center;font-weight:700;color:{clr}'>{iyilesme:+.1f}</td>
                </tr>"""
            correction_html = f"""
            <p style='font-size:13px;color:#57534e;margin-bottom:16px'>{ds.get('ozet','')}</p>
            <table class='proxy'>
              <thead><tr>
                <th>Demografik Değişken</th>
                <th style='text-align:center'>Önceki Fark</th>
                <th style='text-align:center'></th>
                <th style='text-align:center'>Sonraki Fark</th>
                <th style='text-align:center'>İyileşme</th>
              </tr></thead>
              <tbody>{rows}</tbody>
            </table>"""

        # Bias sıralaması
        siralama_html = ""
        if "Çapraz Değerlendirme" in log:
            for i, item in enumerate(log["Çapraz Değerlendirme"].get("bias_katki_siramasi", []), 1):
                clr = bias_color(item["skor"])
                w = min(item["skor"] * 100, 100)
                siralama_html += f"""
                <div class='corr-row'>
                  <span class='corr-label'>{i}. {item['ajan']}</span>
                  <div class='corr-bar-bg'><div class='corr-bar' style='width:{w:.0f}%;background:{clr}'></div></div>
                  <span class='corr-val' style='color:{clr}'>{item['skor']:.3f}</span>
                </div>"""

        html = f"""<!DOCTYPE html>
<html lang='tr'><head><meta charset='UTF-8'>
<title>TargetMind AI — Süreç Raporu</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:system-ui,sans-serif;background:#f7f6f3;color:#1c1917;line-height:1.6}}
  .page{{max-width:900px;margin:0 auto;padding:48px 32px}}
  h1{{font-size:26px;font-weight:800;letter-spacing:-.03em;margin-bottom:4px}}
  h2{{font-size:12px;letter-spacing:.15em;color:#78716c;text-transform:uppercase;margin:40px 0 16px}}
  .meta{{font-size:13px;color:#78716c;margin-bottom:40px}}
  .agent-card{{background:#fff;border:1px solid #e2dfd8;border-radius:12px;padding:20px;margin-bottom:12px}}
  .agent-header{{display:flex;align-items:center;gap:12px;margin-bottom:10px;flex-wrap:wrap}}
  .agent-icon{{font-size:18px;font-weight:800}}
  .agent-name{{font-size:15px;font-weight:700;flex:1}}
  .bias-badge{{font-size:12px;padding:3px 10px;border:1px solid;border-radius:20px;font-weight:600}}
  .agent-ozet{{font-size:13px;color:#57534e;margin-bottom:8px}}
  .agent-sonuc{{font-size:13px;font-weight:600;margin-top:8px}}
  .detail-list{{font-size:13px;color:#57534e;padding-left:20px;margin:8px 0}}
  .detail-list li{{margin-bottom:4px}}
  .card{{background:#fff;border:1px solid #e2dfd8;border-radius:12px;padding:24px;margin-bottom:16px}}
  table.proxy{{width:100%;border-collapse:collapse;font-size:13px}}
  table.proxy th{{background:#f7f6f3;padding:8px 12px;text-align:left;font-size:11px;letter-spacing:.08em;color:#78716c}}
  table.proxy td{{padding:8px 12px;border-bottom:1px solid #f0eeea}}
  .corr-row{{display:flex;align-items:center;gap:12px;margin-bottom:10px}}
  .corr-label{{font-size:13px;width:220px;flex-shrink:0}}
  .corr-bar-bg{{flex:1;height:8px;background:#f0eeea;border-radius:4px;overflow:hidden}}
  .corr-bar{{height:100%;border-radius:4px}}
  .corr-val{{font-size:13px;font-weight:700;width:50px;text-align:right}}
  @media print{{body{{background:#fff}}.page{{padding:24px}}}}
</style></head>
<body><div class='page'>

  <h1>TargetMind AI — Süreç Raporu</h1>
  <p class='meta'>Üretildi: {now} &nbsp;·&nbsp; 7 Ajan · Öz-Farkındalıklı Pipeline</p>

  <h2>Ajan Öz-Değerlendirmeleri</h2>
  {ajan_html or "<p style='color:#78716c;font-size:14px'>Henüz çalıştırılmadı.</p>"}

  <h2>Bias Katkı Sıralaması</h2>
  <div class='card'>
    <p style='font-size:13px;color:#57534e;margin-bottom:16px'>
      Hangi ajan pipeline'a en fazla bias kattı?
    </p>
    {siralama_html or "<p style='color:#78716c;font-size:13px'>Henüz hesaplanmadı.</p>"}
  </div>

  <h2>Çapraz Ajan Eleştirisi</h2>
  <div class='card'>{critique_html}</div>

  <h2>Düzeltme Sonuçları</h2>
  <div class='card'>{correction_html}</div>

</div></body></html>"""

        return Response(html, mimetype="text/html",
                        headers={"Content-Disposition": "inline; filename=surec-raporu.html"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/pool-report")
def pool_report():
    """Optimal müşteri havuzu raporunu HTML olarak üretir."""
    try:
        havuz_path = Path("data/optimal_kitle_raporu.json")
        havuz = json.loads(havuz_path.read_text()) if havuz_path.exists() else {}

        from datetime import datetime
        now = datetime.now().strftime("%d %B %Y, %H:%M")

        # Demografik dağılım
        demo_html = ""
        for key, val in havuz.items():
            if key.endswith("_dagilimi") and isinstance(val, dict):
                col = key.replace("_dagilimi", "")
                rows = "".join(
                    f"<tr><td>{k}</td><td style='text-align:right;font-weight:600'>{v}</td></tr>"
                    for k, v in sorted(val.items(), key=lambda x: -x[1])
                )
                demo_html += f"""
                <div class='demo-block'>
                  <p class='demo-col'>{col}</p>
                  <table class='inner'><tbody>{rows}</tbody></table>
                </div>"""

        # Skor dağılımı
        skor_dist = havuz.get("skor_dagilimi", {})
        dist_html = "".join(
            f"<div class='dist-item'><span>{k}</span><span style='font-weight:700'>{v}</span></div>"
            for k, v in sorted(skor_dist.items(),
                key=lambda x: ["Düşük","Orta","Yüksek","Prime","nan"].index(x[0])
                if x[0] in ["Düşük","Orta","Yüksek","Prime","nan"] else 99)
        )

        # Optimal 20 tablosu
        optimal_20 = havuz.get("optimal_20", [])
        tablo_html = ""
        if optimal_20:
            keys = list(optimal_20[0].keys())
            headers = "".join(f"<th>{k}</th>" for k in keys)
            rows = ""
            for i, row in enumerate(optimal_20):
                bg = "background:rgba(67,56,202,.04)" if i < 3 else ""
                cells = "".join(f"<td>{v}</td>" for v in row.values())
                rows += f"<tr style='{bg}'>{cells}</tr>"
            tablo_html = f"""
            <div style='overflow-x:auto;border-radius:10px;border:1px solid #e2dfd8'>
              <table style='width:100%;border-collapse:collapse;font-size:12px'>
                <thead><tr style='background:#f7f6f3'>{headers}</tr></thead>
                <tbody>{rows}</tbody>
              </table>
            </div>"""

        html = f"""<!DOCTYPE html>
<html lang='tr'><head><meta charset='UTF-8'>
<title>TargetMind AI — Optimal Kitle Raporu</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:system-ui,sans-serif;background:#f7f6f3;color:#1c1917;line-height:1.6}}
  .page{{max-width:900px;margin:0 auto;padding:48px 32px}}
  h1{{font-size:26px;font-weight:800;letter-spacing:-.03em;margin-bottom:4px}}
  h2{{font-size:12px;letter-spacing:.15em;color:#78716c;text-transform:uppercase;margin:40px 0 16px}}
  .meta{{font-size:13px;color:#78716c;margin-bottom:40px}}
  .stats{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:8px}}
  .stat{{background:#fff;border:1px solid #e2dfd8;border-radius:10px;padding:16px;text-align:center}}
  .stat-val{{font-size:28px;font-weight:800}}
  .stat-lbl{{font-size:11px;color:#78716c;margin-top:4px}}
  .card{{background:#fff;border:1px solid #e2dfd8;border-radius:12px;padding:24px;margin-bottom:16px}}
  .demo-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px}}
  .demo-block{{background:#fff;border:1px solid #e2dfd8;border-radius:10px;padding:16px}}
  .demo-col{{font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:#78716c;margin-bottom:10px;font-weight:600}}
  table.inner{{width:100%;border-collapse:collapse;font-size:13px}}
  table.inner td{{padding:5px 6px;border-bottom:1px solid #f0eeea}}
  table.inner tr:last-child td{{border-bottom:none}}
  thead tr{{background:#f7f6f3;border-bottom:1px solid #e2dfd8}}
  th{{padding:10px 12px;text-align:left;font-size:11px;letter-spacing:.08em;color:#78716c}}
  td{{padding:9px 12px;border-bottom:1px solid #f0eeea}}
  .dist-item{{display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #f0eeea;font-size:14px}}
  @media print{{body{{background:#fff}}.page{{padding:24px}}}}
</style></head>
<body><div class='page'>

  <h1>TargetMind AI — Optimal Kitle Raporu</h1>
  <p class='meta'>Üretildi: {now} &nbsp;·&nbsp; Bias düzeltilmiş skorlarla oluşturuldu</p>

  <h2>Genel Özet</h2>
  <div class='stats'>
    <div class='stat'>
      <div class='stat-val' style='color:#4338ca'>{havuz.get('toplam_havuz','—')}</div>
      <div class='stat-lbl'>Toplam Havuz</div>
    </div>
    <div class='stat'>
      <div class='stat-val'>{havuz.get('ort_skor','—')}</div>
      <div class='stat-lbl'>Ortalama Skor</div>
    </div>
    <div class='stat'>
      <div class='stat-val' style='color:#dc2626'>%{havuz.get('elenme_orani','—')}</div>
      <div class='stat-lbl'>Elenme Oranı</div>
    </div>
  </div>

  <h2>Skor Dağılımı</h2>
  <div class='card'>{dist_html}</div>

  {"<h2>Demografik Dağılım</h2><div class='demo-grid'>" + demo_html + "</div>" if demo_html else ""}

  <h2>En İyi 20 Müşteri</h2>
  {tablo_html or "<p style='color:#78716c;font-size:14px'>Henüz hesaplanmadı.</p>"}

</div></body></html>"""

        return Response(html, mimetype="text/html",
                        headers={"Content-Disposition": "inline; filename=optimal-kitle.html"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    TargetMind AI chatbot — analiz sonuçlarını bağlam olarak kullanır,
    kullanıcının sorularını Claude Haiku ile yanıtlar.
    """
    body    = request.json or {}
    message = body.get("message", "").strip()
    history = body.get("history", [])  # [{role, content}, ...]

    if not message:
        return jsonify({"error": "Mesaj boş olamaz."}), 400

    # ── Mevcut analiz durumunu özetle ──────────────────────────────
    context_parts = []
    mapping = _read_mapping()
    context_parts.append(f"Column mapping: {json.dumps(mapping, ensure_ascii=False)}")

    try:
        raw = pd.read_csv("data/customers.csv")
        context_parts.append(f"Ham veri: {len(raw)} kayıt, kolonlar: {list(raw.columns)}")
    except Exception:
        pass

    try:
        cleaned = pd.read_csv("data/customers_cleaned.csv")
        context_parts.append(f"Temizlenmiş veri: {len(cleaned)} kayıt")
    except Exception:
        pass

    try:
        scored = pd.read_csv("data/customers_scored.csv")
        if "skor_segmenti" in scored.columns:
            dist = scored["skor_segmenti"].value_counts().to_dict()
            avg  = round(float(scored["potansiyel_skor"].mean()), 1)
            context_parts.append(f"Skorlama: ort={avg}, dağılım={dist}")
        if "potansiyel_skor" in scored.columns:
            top5 = scored.nlargest(5, "potansiyel_skor")
            id_col = mapping.get("id_col", scored.columns[0])
            name_col = "name" if "name" in scored.columns else id_col
            top5_list = top5[[name_col, "potansiyel_skor"]].to_dict(orient="records")
            context_parts.append(f"Top 5 kayıt: {top5_list}")
        demo_cols = [c for c in mapping.get("demographic_cols", []) if c in scored.columns]
        for col in demo_cols:
            g = scored.groupby(col)["potansiyel_skor"].mean().round(1).to_dict()
            context_parts.append(f"{col} grup ortalamaları: {g}")
    except Exception:
        pass

    try:
        final = pd.read_csv("data/final_targets.csv")
        context_parts.append(f"Final hedef kitle: {len(final)} kayıt")
    except Exception:
        pass

    try:
        optimal = pd.read_csv("data/optimal_targets.csv")
        context_parts.append(f"Optimal kitle (gelecek potansiyel): {len(optimal)} kayıt")
    except Exception:
        pass

    analysis_context = "\n".join(context_parts)

    system_prompt = f"""Sen TargetMind AI'ın analiz asistanısın. Bu sistem 7 ajanlı **assumption-surfacing** bir pipeline yürütür: amaç çıktıyı demografik olarak eşitlemek DEĞİL, her ajanın verdiği kararın altındaki varsayımı şeffaf hale getirmek ve kararın metodolojiye mi yoksa veriye mi bağlı olduğunu ayırt etmektir.

Anahtar metafor:
- Bir karar "kırılgan" (fragile): alternatif yöntem kullanıldığında sonuç belirgin farklı çıkıyor → karar metodolojiye bağımlı, varsayıma duyarlı
- Bir karar "robust": alternatif yöntem benzer sonuç veriyor → karar veriye bağlı, varsayımdan bağımsız
- Gerçek bir veri sinyalini "düzeltmek" amacımız DEĞİL, kararın kaynağını netleştirmek

7 Ajan Pipeline:
1. Veri Temizleme — temizleme yönteminin altındaki varsayımı ifşa eder (mod dolgu = "eksik değer tipiktir"). Alternatif yöntemle sonuç ne kadar değişir? Bu fark = metodolojik kırılganlık.
2. Segmentasyon — demografik temsil farklarını şeffaf raporlar (gerçek pazar yapısı mı, artefakt mı?).
3. İlk Skorlama — ağırlık seçiminin altındaki "değerli müşteri" varsayımını ifşa eder. Eşit ağırlıkla benzer kitle çıkıyor mu?
4. Proxy & Bias Tespiti — skor metriklerinin demografik özelliklerle dolaylı ilişkilerini Cramer's V ile ifşa eder (bilgi, hata değil).
5. Çapraz Ajan Değerlendirmesi — her ajanın varsayım hassasiyetini sıralar, en kırılgan kararı tespit eder. Düzeltme değil, alternatif test önerir.
6. Alternatif Varsayım Testi — eşit ağırlıkla yeniden skorlar, iki yöntem arası farkı (karar kırılganlığı: FRAGILE/ORTA/ROBUST) ölçer. İteratif: turlar arası fark stabilize olunca convergence.
7. Final Rapor — final kitleyle birlikte her kararın hangi yöne kırılgan olduğunu şeffaflık raporu olarak sunar.

Mevcut analiz durumu:
{analysis_context}

Kullanıcının sorularını kısa, net ve Türkçe olarak yanıtla. "Bias var, düzeltelim" dilini KULLANMA — bunun yerine "karar şu varsayıma dayanıyor, alternatifle şu kadar değişir" diliyle açıkla."""

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

        messages = []
        for h in history[-10:]:  # Son 10 mesaj
            if h.get("role") in ("user", "assistant"):
                messages.append({"role": h["role"], "content": h["content"]})
        messages.append({"role": "user", "content": message})

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=600,
            system=system_prompt,
            messages=messages,
        )
        reply = response.content[0].text
        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"\n🎯  TargetMind AI başlatılıyor...")
    print(f"    http://localhost:{port}\n")
    app.run(debug=False, port=port, threaded=True)
