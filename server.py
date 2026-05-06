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

AGENT_NAMES = [
    "Veri Temizleme",
    "Segmentasyon",
    "Skorlama & Eşik Kontrolü",
    "Bias Tespiti",
    "Proxy Değişken Tespiti",
    "Eleştirmen",
    "Yeniden Optimizasyon",
    "Gelecek Harcama Potansiyeli",
]

AGENT_DESCS = [
    "Eksik değerler, aykırı değerler ve format tutarsızlıkları temizleniyor",
    "Müşteriler davranış ve potansiyele göre segmentlere ayrılıyor",
    "Potansiyel skorları hesaplanıyor, bias eşiği kontrol ediliyor",
    "Demografik gruplar arası sistematik dışlama analiz ediliyor",
    "Dolaylı bias üreten proxy değişkenler tespit ediliyor",
    "Kararlar counterfactual test ile sayısal olarak sorgulanıyor",
    "Bias bulgularına göre final hedef kitle oluşturuluyor",
    "Bağlılık, sadakat ve ikna edilebilirlik ile gelecek potansiyel hesaplanıyor",
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
        skor_dist = scored["skor_segmenti"].value_counts().to_dict() if "skor_segmenti" in scored.columns else {}

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

        # Özet
        ozet = {}
        if "potansiyel_skor" in final.columns and len(final) > 0:
            ozet["ort_skor"] = round(float(final["potansiyel_skor"].mean()), 1)
        for col in metric_cols[:3]:
            if col in final.columns and len(final) > 0:
                try:
                    ozet[f"ort_{col}"] = round(float(final[col].mean()), 1)
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
    Saf Python pipeline — LLM yok, araçlar doğrudan çağrılır.
    Deterministik, hızlı, güvenilir.
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
            bias_threshold_check, detect_bias, audit_cleaning_decisions,
            detect_proxy_variables, counterfactual_test, build_final_targets,
            calculate_future_potential,
        )

        # ── Veri hazırla ──────────────────────────────────────────
        if mode == "demo":
            from data.generate_data import build_row, inject_issues
            rows = [build_row(i + 1) for i in range(300)]
            df = pd.DataFrame(rows)
            df = inject_issues(df)
            df = df.sample(frac=1, random_state=99).reset_index(drop=True)
            df.to_csv("data/customers.csv", index=False)

        count = len(pd.read_csv("data/customers.csv"))
        msg_queue.put({"type": "data_ready", "count": count})

        # ── ADIM 1: Veri Temizleme ────────────────────────────────
        send_start(0)
        r1 = json.loads(clean_data.run(
            json.dumps({"yas_min": 13, "yas_max": 75})
        ))
        send_done(0,
            f"{r1['baslangic_satir']} → {r1['bitis_satir']} müşteri  ·  "
            f"{r1['elenen_satir']} satır elendi")

        # ── ADIM 2: Segmentasyon ──────────────────────────────────
        send_start(1)
        r2 = json.loads(segmentation_analysis.run(""))
        tur_dist  = r2.get("tur_dagilimi", {})
        top_tur   = max(tur_dist, key=tur_dist.get) if tur_dist else "—"
        metric_stats = r2.get("metrik_istatistikler", {})
        col_count = len(metric_stats)
        send_done(1,
            f"En yaygın segment: {top_tur}  ·  {col_count} metrik analiz edildi")

        # ── ADIM 3: Skorlama & Bias Eşiği ────────────────────────
        send_start(2)
        r3 = json.loads(score_customers.run(""))
        r3t = json.loads(bias_threshold_check.run(""))
        gelir_fark = r3t["gelir_fark_puan"]
        level3 = "error" if gelir_fark > 5 else "ok"
        esik_msg = (f"⚠ Bias eşiği aşıldı — gelir farkı {gelir_fark} puan"
                    if gelir_fark > 5 else f"Bias eşiği OK ({gelir_fark} puan)")
        dist = r3["skor_dagilimi"]
        send_done(2,
            f"Prime: {dist.get('Prime',0)}  Yüksek: {dist.get('Yüksek',0)}  "
            f"Orta: {dist.get('Orta',0)}  Düşük: {dist.get('Düşük',0)}  ·  {esik_msg}",
            level3)

        # ── ADIM 4: Bias Tespiti ──────────────────────────────────
        send_start(3)
        r4a = json.loads(audit_cleaning_decisions.run(""))
        r4b = json.loads(detect_bias.run(""))
        yuksek_risk = r4a["ozet"]["yuksek_riskli_karar"]
        max_bias = max(
            (v.get("max_fark", 0) for v in r4b.values() if isinstance(v, dict) and "max_fark" in v),
            default=0
        )
        level4 = "warn" if yuksek_risk > 0 else "ok"
        send_done(3,
            f"Temizleme: {yuksek_risk} yüksek riskli karar  ·  "
            f"Maks. demografik fark: {max_bias:.1f} puan",
            level4)

        # ── ADIM 5: Proxy Tespiti ─────────────────────────────────
        send_start(4)
        r5 = json.loads(detect_proxy_variables.run(""))
        yuksek_proxy = r5["yuksek_riskli_proxy_sayisi"]
        toplam_proxy = len(r5["proxy_analizi"])
        level5 = "warn" if yuksek_proxy > 0 else "ok"
        send_done(4,
            f"{toplam_proxy} proxy ilişkisi tespit edildi  ·  "
            f"{yuksek_proxy} yüksek riskli",
            level5)

        # ── ADIM 6: Counterfactual Test ───────────────────────────
        send_start(5)
        r6 = json.loads(counterfactual_test.run(""))
        best_name = r6["tavsiye_edilen_senaryo"]
        best_s    = next((s for s in r6["senaryolar"] if s["senaryo"] == best_name), {})
        best_fark = best_s.get("gelir_grubu_fark_puan", 0)
        send_done(5,
            f"Optimal: {best_name}  ·  Gelir bias farkı {best_fark} puana indi")

        # ── ADIM 7: Final Optimizasyon ────────────────────────────
        send_start(6)
        min_skor = 38 if best_fark <= 5 else 42
        r7 = json.loads(build_final_targets.run(json.dumps({
            "min_skor": min_skor,
        })))
        send_done(6,
            f"Final hedef kitle: {r7['final_hedef_kitle']} müşteri  ·  "
            f"%{r7['elenme_orani_yuzde']} elenme")

        # ── ADIM 8: Gelecek Harcama Potansiyeli ───────────────────
        send_start(7)
        r8 = json.loads(calculate_future_potential.run(""))
        send_done(7,
            f"Optimal kitle: {r8['optimal_kitle_sayisi']} müşteri  ·  "
            f"Ort. gelecek skor: {r8['optimal_ort_gelecek_skor']}")

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

    system_prompt = f"""Sen TargetMind AI'ın analiz asistanısın. Bu sistem herhangi bir CSV verisini 8 adımlı bir pipeline'dan geçirerek en yüksek potansiyelli kayıtları bulur ve demografik bias'ı tespit eder.

Pipeline adımları:
1. Veri Temizleme — duplikasyon, negatif değerler, IQR aykırı değer tespiti, medyan/mod doldurma
2. Segmentasyon — segment dağılımı ve metrik istatistikleri
3. Skorlama & Bias Eşiği — metric kolonlardan 0-100 potansiyel skor, eşik kontrolü
4. Bias Tespiti — temizleme kararları denetimi + demografik grup skor farklılıkları
5. Proxy Değişken Tespiti — Cramer's V ile dolaylı bias riski taşıyan kolonlar
6. Counterfactual Test — farklı ağırlık senaryolarında bias azaltma deneyi
7. Final Hedef Kitle — minimum skoru geçen kayıtlar
8. Gelecek Potansiyel — engagement skoru ile uzun vadeli değer tahmini

Mevcut analiz durumu:
{analysis_context}

Kullanıcının sorularını kısa, net ve Türkçe olarak yanıtla. Teknik terimleri basitçe açıkla. Sayısal sonuçları bağlama göre yorumla."""

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
