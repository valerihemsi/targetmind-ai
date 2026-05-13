"""
TargetMind AI — Assumption-Surfacing Multi-Agent System (LLM-driven)

Bu pipeline'ın amacı çıktıyı demografik olarak eşitlemek DEĞİLDİR. Her ajan:
  1. Verdiği kararın altındaki varsayımı şeffaf hale getirir
  2. Alternatif bir varsayımla aynı kararı yeniden çalıştırır
  3. İki sonuç arası farkı ölçer (= metodolojik kırılganlık)
  4. Karar kırılgansa (yönteme bağımlı) bunu kabul edip raporlar
  5. Karar robustsa (veriye bağlı) güvenle kullanılır

Anahtar prensip: gerçek bir veri sinyalini gizlemek değil, kararın
veriden mi metodolojiden mi geldiğini sayısal olarak ayırt etmek.

`server.py` ayrı bir yol kullanır (LLM olmadan tool'ları sıralı çağırır).
Bu `crew.py` kullanılırsa: gerçek ajan akıl yürütmesi + tool sonuçları.

Çalıştırma: `python main.py` (CSV üretip pipeline'ı tetikler).
"""
from crewai import Agent, Crew, LLM, Process, Task

from tools.data_tools import (
    clean_data,
    segmentation_analysis,
    score_customers,
    detect_proxy_and_bias,
    inter_agent_critique,
    corrected_scoring,
    build_final_and_report,
    counterfactual_test,
    iterative_correction,
)

llm = LLM(model="anthropic/claude-sonnet-4-6")


# ══════════════════════════════════════════════════════════════════════════════
#  AJANLAR — her biri kendi kararının varsayımını yüzeye çıkarır
# ══════════════════════════════════════════════════════════════════════════════

data_cleaning_agent = Agent(
    role="Veri Temizleme Uzmanı",
    goal=(
        "Ham müşteri verisini temizle ve her temizleme kararının altındaki "
        "varsayımı şeffaf hale getir. 'Medyan/mod ile doldurdum' kararının "
        "altında 'eksik değer tipik kullanıcıya benzer' varsayımı yatar. "
        "Alternatif varsayım ('Bilinmiyor' kategori) ile sonuç ne kadar değişir? "
        "Bu farkı sayısal olarak rapor et — düzeltmek için değil, kararının "
        "metodolojiye ne kadar bağımlı olduğunu eleştirmenin görmesi için."
    ),
    backstory=(
        "Sen veri kalitesinde uzmanlaşmış bir analistsin. Temizleme adımlarının "
        "kendisinin de varsayım taşıdığını biliyorsun. Bir mod dolgu kararı "
        "demografik dağılımı kaydırabilir, ama bu kaymanın 'bias hatası' olarak "
        "düzeltilmesi yerine **şeffaflık** ile sunulması gerektiğini bilirsin. "
        "Sonuç gerçek veri sinyali de olabilir, metodoloji artefaktı da; "
        "ikisini ayırt etmek için kararlarını alternatifle test edersin."
    ),
    tools=[clean_data],
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

segmentation_agent = Agent(
    role="Müşteri Segmentasyon Uzmanı",
    goal=(
        "Temizlenmiş veriyi segmentlere ayır ve demografik temsil dağılımını "
        "rapor et. Gözlemlediğin temsil farkları **gerçek pazar yapısı** "
        "olabilir; amaç bu farkları gizlemek değil, alt akıştaki ajanların "
        "(skorlama, eleştirmen) bu temel varsayımı görmesini sağlamak."
    ),
    backstory=(
        "Sen oyun sektörü müşteri davranışını derinden anlayan bir analistsin. "
        "Hardcore'dan casual'a farklı oyuncu profillerini tanırsın. Demografik "
        "temsil farklarının pazardaki gerçek yapıdan mı yoksa örnekleme "
        "bias'ından mı geldiğini ayırt etmenin sonradan zor olduğunu bilirsin — "
        "o yüzden temsil farklarını mutlaka şeffaf olarak rapor edersin."
    ),
    tools=[segmentation_analysis],
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

scoring_agent = Agent(
    role="Müşteri Skorlama Uzmanı",
    goal=(
        "Her müşteriye 0-100 potansiyel skor ver. Seçtiğin ağırlık şeması = "
        "'değerli müşteri' tanımındaki varsayımın. Bu varsayımı açıkça yaz: "
        "hangi metriğe neden yüksek ağırlık veriyorsun? Demografik gruplar "
        "arası ortaya çıkan skor farkı veriden mi yöntemden mi geliyor — "
        "bunu eleştirmenin counterfactual ile test edeceğini bilerek hareket et."
    ),
    backstory=(
        "Sen CLV modellemesinde deneyimli bir data scientistsin. Tek bir "
        "metriğin (örn. harcama) skoru yönlendirmesinin bir tercih olduğunu, "
        "matematiksel bir 'doğru' olmadığını bilirsin. O yüzden ağırlık "
        "kararlarını gerekçelendirir, eşit ağırlık alternatifiyle karşılaştırma "
        "olasılığını **kendi başına** dile getirirsin."
    ),
    tools=[score_customers],
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

proxy_bias_agent = Agent(
    role="Proxy ve Bias Tespit Uzmanı",
    goal=(
        "Skorlamada kullanılan metriklerin (oyun saati, cihaz türü, platform) "
        "hassas demografik özelliklerle (cinsiyet, yaş, gelir) ne kadar güçlü "
        "korelasyon taşıdığını Cramer's V ile ölç. Amaç bu metrikleri silmek "
        "değil — kararın hangi yolla dolaylı olarak demografik özelliği "
        "temsil ettiğini şeffaf hale getirmek. Eleştirmen bu bilgiyle "
        "sonucun **anlamını** sorgular."
    ),
    backstory=(
        "Sen algoritmik adalet ve istatistiksel ilişki analizinde uzmanlaşmış "
        "bir araştırmacısın. 'Cihaz türü' veya 'şehir' gibi görünüşte tarafsız "
        "değişkenlerin demografik özelliklerle ilişkili olabileceğini, bu yüzden "
        "ekonomik veya coğrafi bir tarihselliği yeniden ürettiğini bilirsin. "
        "Bunu bir 'hata' olarak değil, kararı **anlamak için bilgi** olarak sunarsın."
    ),
    tools=[detect_proxy_and_bias],
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

critic_agent = Agent(
    role="Varsayım Sorgulama Eleştirmeni",
    goal=(
        "Önceki ajanların kararlarının altındaki varsayımları yüzeye çıkar. "
        "Hangi karar alternatif yönteme en duyarlı (en kırılgan)? Bunu "
        "`İteratif Varsayım Robustluk Testi` ve `Counterfactual Test Aracı` "
        "ile sayısal olarak göster. Çıktıyı düzeltmeye **çağırmıyorsun** — "
        "ajanların hangi tercihinin sonucu domine ettiğini şeffaf hale "
        "getiriyorsun. Final ajan bu şeffaflığı son rapora taşır."
    ),
    backstory=(
        "Sen bağımsız bir denetçisin — bir kararın yalnızca sonucuna değil, "
        "üretildiği sürece bakarsın. Diğer ajanların farkında olmadan yaptığı "
        "varsayımları sayısal kanıtla ifşa edersin. Amacın sistemi 'eşitlemek' "
        "değil; gerçek veri sinyalini metodoloji artefaktından ayırt etmek. "
        "Bir karar yönteme bağımlıysa bunu açıkça söyler, gerçek bir sinyalse "
        "savunursun."
    ),
    tools=[inter_agent_critique, counterfactual_test],
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

corrected_scoring_agent = Agent(
    role="Varsayım Robustluk Test Uzmanı",
    goal=(
        "Skorlama ajanının ilk yönteminin yanına alternatif bir varsayım "
        "(eşit ağırlık) ile yeniden hesaplanan skor koy. İki sonuç arası "
        "fark = ilk yöntemin metodolojik kırılganlığı. Bu fark küçükse karar "
        "robust (veriye bağlı); büyükse karar fragile (yönteme bağlı). "
        "İteratif test ile turlar arası fark stabilize olana kadar (= "
        "convergence) yöntemleri karşılaştırmaya devam et."
    ),
    backstory=(
        "Sen 'doğru cevap' arayan değil, hangi cevabın hangi varsayıma "
        "dayandığını gösteren bir test uzmanısın. Çıktıyı zorla 'düzeltmezsin' — "
        "iki yöntemin sonuçlarını yan yana koyar, hangisinin daha az varsayıma "
        "bağlı olduğunu ölçersin. Convergence sağlandığında karar artık "
        "yönteme duyarsızdır; sağlanamazsa kırılganlığı kabul edip raporlarsın."
    ),
    tools=[corrected_scoring, iterative_correction],
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

final_report_agent = Agent(
    role="Final Rapor ve Şeffaflık Uzmanı",
    goal=(
        "Tüm pipeline çıktılarını bütünleştirerek **iki rapor** üret: "
        "(1) Süreç Raporu — hangi ajanın kararı en kırılgandı, hangi varsayım "
        "alternatifi en farklı sonuç verdi, convergence sağlandı mı; "
        "(2) Optimal Kitle Raporu — final hedef kitle + her demografik boyut "
        "için yöntem-duyarlılığı notu. Pazarlama ekibine teslim ederken "
        "kararın hangi noktaları sağlam, hangi noktaları varsayımdan etkilenmiş "
        "olarak işaretle."
    ),
    backstory=(
        "Sen şeffaflığı strateji ile birleştiren bir karar destek uzmanısın. "
        "Final raporda yalnızca 'kim seçildi' değil, 'bu seçim hangi varsayıma "
        "dayanıyor, alternatif varsayımda kim seçilirdi' bilgisini de verirsin. "
        "Pazarlama ekibi böylece kararı bilinçli olarak sahiplenir, körü "
        "körüne uygulamak zorunda kalmaz."
    ),
    tools=[build_final_and_report],
    llm=llm,
    verbose=True,
    allow_delegation=False,
)


# ══════════════════════════════════════════════════════════════════════════════
#  GÖREVLER — her görev "varsayımı yüzeye çıkar + alternatifle test et" diline
# ══════════════════════════════════════════════════════════════════════════════

task_clean = Task(
    description=(
        "Ham müşteri verisini temizle ve temizleme yönteminin altındaki "
        "varsayımı yüzeye çıkar.\n\n"
        "1. `Veri Temizleme Ajanı` aracını çalıştır.\n"
        "   - İsteğe bağlı yaş sınırı kuralı: {\"yas_min\": 13, \"yas_max\": 75}.\n"
        "2. Aracın `oz_degerlendirme.varsayim` ve `oz_degerlendirme.sonuc` "
        "alanlarını yorumla:\n"
        "   - Hangi varsayım yapıldı (mod dolgu vs.)?\n"
        "   - Demografik kayma metodolojiden mi geliyor?\n"
        "   - `bias_katki_skoru` (≡ varsayım hassasiyeti) 0.3'ün üstündeyse "
        "kararının kırılganlığını açıkça kabul et.\n"
        "3. Düzeltme önerme; alternatif yöntemin (örn. 'Bilinmiyor' kategori) "
        "**ne kadar farklı** sonuç vereceğini eleştirmenin daha sonra test "
        "edeceğini bil.\n\n"
        "Çıktı: Temizleme raporu + ifşa edilen varsayım + kırılganlık değerlendirmesi."
    ),
    expected_output=(
        "Temizleme kararları, başlangıç/bitiş satır sayısı, demografik "
        "kaymalar, ifşa edilen varsayım, varsayım hassasiyet skoru ve şeffaf yorum."
    ),
    agent=data_cleaning_agent,
)

task_segment = Task(
    description=(
        "Temizlenmiş veriyi segmentlere ayır.\n\n"
        "1. `Segmentasyon Ajanı` aracını çalıştır.\n"
        "2. Demografik temsil farklarını rapor et — bu farkın **gerçek pazar "
        "yapısından** mı yoksa örnekleme/temizleme artefaktından mı geldiğini "
        "alt akıştaki ajanlar için açık bırak.\n"
        "3. Hangi segmentlerin yüksek değerli (Strateji/RPG, mobil/PC) olduğunu "
        "veriyle gerekçelendir.\n\n"
        "Çıktı: Segment profilleri + demografik şeffaflık raporu."
    ),
    expected_output=(
        "Her segment için boyut, ortalama metrikler, demografik temsil farkı "
        "ve stratejik öncelik (yüksek/orta/düşük) gerekçeli olarak."
    ),
    agent=segmentation_agent,
    context=[task_clean],
)

task_score = Task(
    description=(
        "Her müşteriye 0-100 potansiyel skoru ver.\n\n"
        "1. `İlk Skorlama Ajanı` aracını çalıştır.\n"
        "   - Ağırlıkları JSON olarak verebilirsin (örn. harcama vurgusu) ya da "
        "eşit ağırlık için boş bırakabilirsin.\n"
        "2. Aracın `oz_degerlendirme.varsayim` alanını yorumla — hangi metriğe "
        "neden yüksek ağırlık verdin? Bu bir veri-temelli karar mı, kolay "
        "varsayılan mı?\n"
        "3. Demografik gruplar arası ortaya çıkan skor farkı (`max_fark_puan`) "
        "büyükse, bu farkın yöntem mi yoksa veri mi olduğuna eleştirmenin "
        "counterfactual ile karar vereceğini bilerek hareket et — düzeltme "
        "önerme.\n\n"
        "Çıktı: Skorlama modeli + ifşa edilen ağırlık varsayımı + şeffaflık raporu."
    ),
    expected_output=(
        "Kullanılan ağırlıklar, skor dağılımı, demografik grup ortalamaları, "
        "ifşa edilen varsayım, en dominant metrik, varsayım hassasiyet skoru."
    ),
    agent=scoring_agent,
    context=[task_clean, task_segment],
)

task_proxy_bias = Task(
    description=(
        "Skorlama metriklerinin demografik özelliklerle proxy ilişkisini ve "
        "yüksek-skor segmentindeki demografik dağılımı tespit et.\n\n"
        "1. `Proxy ve Bias Tespit Ajanı` aracını çalıştır.\n"
        "2. YÜKSEK riskli proxy ilişkiler için: bu ilişki neden var "
        "(sosyoekonomik, coğrafi, tarihsel)? Bu metriğin skorlamada "
        "kullanılması bir karar — ajanların farkında olması gereken bir "
        "varsayım yükü.\n"
        "3. Hangi bilgilerin eleştirmenin işine yarayacağını öne çıkar. "
        "Çıktıyı silme veya düzeltme önerme.\n\n"
        "Çıktı: Proxy + bias raporu — kararın anlamını netleştiren bilgi olarak."
    ),
    expected_output=(
        "Proxy listesi (Cramer's V), demografik segment farkları, ifşa edilen "
        "varsayım ve bu ilişkilerin kararı nasıl etkilediğine dair yorum."
    ),
    agent=proxy_bias_agent,
    context=[task_score],
)

task_critique = Task(
    description=(
        "Önceki ajanların kararlarının altındaki varsayımları sayısal kanıtla "
        "sorgula. Çıktıyı düzeltmeyeceksin — kararların hangi yönte ne kadar "
        "duyarlı olduğunu **yüzeye çıkaracaksın**.\n\n"
        "1. `Çapraz Ajan Değerlendirme Ajanı` aracını çalıştır — log'u okur, "
        "her ajanın varsayım hassasiyetini sıralar, alternatif test önerir.\n"
        "2. `Counterfactual Test Aracı` aracını çalıştır — 4 senaryo "
        "(mevcut / eşit_agirlik / harcama_20 / harcama_10) karşılaştır. "
        "Senaryolar arası fark büyükse ilk skorlama yöntemi yöntem-duyarlı "
        "(fragile); küçükse robust.\n"
        "3. İki çıktıyı birleştir: hangi ajanın kararı en kırılgan, hangi "
        "varsayım alternatifi en fazla fark üretiyor? Bu bilgiyi düzeltme "
        "kuralı olarak değil **şeffaflık raporu** olarak ilet.\n\n"
        "Çıktı: Varsayım hassasiyeti raporu + counterfactual karşılaştırma."
    ),
    expected_output=(
        "Ajan varsayım hassasiyet sıralaması, counterfactual karşılaştırma "
        "tablosu, en yöntem-duyarlı karar, alternatif test önerileri "
        "(zorlamadan, şeffaflık için)."
    ),
    agent=critic_agent,
    context=[task_clean, task_segment, task_score, task_proxy_bias],
)

task_corrected = Task(
    description=(
        "İlk skorlamayı alternatif bir varsayımla yeniden çalıştır ve sonuç "
        "yöntem değişikliğine ne kadar duyarlı bul.\n\n"
        "1. `İteratif Varsayım Robustluk Testi` aracını çalıştır.\n"
        "   - Default 3 tura kadar critic → alternative scoring döngüsü.\n"
        "   - Turlar arası max demografik fark değişimi 1 puan altına inince "
        "(STABILITE_ESIGI) erken sonlanır — bu noktada karar **convergence'e** "
        "ulaşmıştır: artık alternatife duyarsız (robust).\n"
        "2. `iterasyonlar` listesini yorumla:\n"
        "   - Her turda fark nasıl değişti?\n"
        "   - `karar_kirilganligi` etiketi her turda ne oldu (FRAGILE/ORTA/ROBUST)?\n"
        "   - `erken_sonlandi` true mu, kararlar yumuşadı mı?\n"
        "3. Convergence sağlanmazsa: bu bir 'başarısızlık' değil, kararın "
        "**inherently varsayıma duyarlı** olduğunun kanıtı. Final raporda "
        "her iki yöntemin sonucu birlikte sunulmalı.\n\n"
        "Çıktı: İteratif robustluk raporu + convergence analizi (zorlamasız)."
    ),
    expected_output=(
        "Tur sayısı, başlangıç/son max demografik fark, her tur için karar "
        "kırılganlığı + uygulanan ağırlıklar, convergence durumu (✓/ⓘ) ile "
        "şeffaflık yorumu."
    ),
    agent=corrected_scoring_agent,
    context=[task_critique],
)

task_final = Task(
    description=(
        "Final hedef müşteri kitlesini oluştur ve şeffaflık raporu üret.\n\n"
        "1. `Final Optimizasyon ve Rapor Ajanı` aracını çalıştır.\n"
        "   - Minimum skor eşiğini belirle (default 50).\n"
        "   - Kriterler JSON: {\"min_skor\": 55}.\n"
        "2. `surec_raporu` ve `optimal_kitle_raporu`'nu birlikte yorumla:\n"
        "   - En yöntem-duyarlı demografik boyut hangisi? (= en kırılgan karar)\n"
        "   - Ajanlar arası kim en kırılgan kararlı, kim en robust?\n"
        "3. Pazarlama ekibi için final çıktı: **kararın hangi yönü güvenle "
        "kullanılabilir, hangi yönü varsayıma duyarlı**? Bu netlik kararın "
        "körü körüne uygulanmasını engeller.\n\n"
        "Çıktı: Şeffaf final rapor + 3-5 somut pazarlama önerisi (kırılganlık notu ile)."
    ),
    expected_output=(
        "Final kitle boyutu, elenme oranı, demografik profil, ortalama skor, "
        "süreç özeti, en yöntem-duyarlı boyut ve 3-5 somut pazarlama önerisi "
        "(her birinde kararın kırılganlık seviyesi belirtilmiş)."
    ),
    agent=final_report_agent,
    context=[task_clean, task_segment, task_score, task_proxy_bias, task_critique, task_corrected],
)


# ══════════════════════════════════════════════════════════════════════════════
#  CREW
# ══════════════════════════════════════════════════════════════════════════════

def build_crew() -> Crew:
    return Crew(
        agents=[
            data_cleaning_agent,
            segmentation_agent,
            scoring_agent,
            proxy_bias_agent,
            critic_agent,
            corrected_scoring_agent,
            final_report_agent,
        ],
        tasks=[
            task_clean,
            task_segment,
            task_score,
            task_proxy_bias,
            task_critique,
            task_corrected,
            task_final,
        ],
        process=Process.sequential,
        verbose=True,
    )
