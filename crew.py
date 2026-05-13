"""
TargetMind AI — Bias-Aware Multi-Agent Customer Targeting (LLM-driven)

Bu modül, deterministik tool'ları (tools/data_tools.py) gerçek bir LLM-driven
multi-agent zinciri ile kullanır. Her ajan Claude Sonnet 4.6 üzerinden ilgili
tool'u çağırır, sayısal çıktıyı yorumlar, kararlarının gerekçesini açıklar.

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
#  AJANLAR — her biri data_tools.py'deki bir veya iki tool'u kullanır
# ══════════════════════════════════════════════════════════════════════════════

data_cleaning_agent = Agent(
    role="Veri Temizleme Uzmanı",
    goal=(
        "Ham müşteri verisindeki hataları, eksiklikleri ve format tutarsızlıklarını "
        "düzelt. Aldığın her temizleme kararının demografik dağılımı nasıl "
        "etkilediğini fark et ve riskli olanları açıkça bildir."
    ),
    backstory=(
        "Sen veri kalitesinde uzmanlaşmış bir analistsin. Kirli verinin yanıltıcı "
        "sonuçlar doğurduğunu biliyorsun ama temizleme kararlarının kendisinin de "
        "bias kaynağı olabileceğinin farkındasın. Her dolgu ve eleme adımının "
        "demografik etkisini ölçer, hangi grupları dolaylı olarak güçlendirdiğini "
        "veya zayıflattığını ifşa edersin."
    ),
    tools=[clean_data],
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

segmentation_agent = Agent(
    role="Müşteri Segmentasyon Uzmanı",
    goal=(
        "Temizlenmiş veriyi anlamlı segmentlere ayır ve her segmentin demografik "
        "temsil farkını saptayarak yorumla. Hedef oyun şirketinin stratejik "
        "öncelikleri açısından hangi segmentler değerli, hangileri risk taşıyor?"
    ),
    backstory=(
        "Sen oyun sektörü müşteri davranışını derinlemesine anlayan bir analistsin. "
        "Hardcore gamer'dan casual oyuncuya, mobile spender'dan PC entuziyastına "
        "kadar farklı profilleri tanırsın. Her segmentin pazarlama değerini "
        "değerlendirirken demografik dengeyi de gözetirsin."
    ),
    tools=[segmentation_analysis],
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

scoring_agent = Agent(
    role="Müşteri Skorlama Uzmanı",
    goal=(
        "Her müşteriye 0-100 arası bir potansiyel skor ver. Skorlama ağırlıklarını "
        "seçerken — örneğin harcama ağırlığını ne kadar yüksek tutacağını — "
        "ortaya çıkacak demografik skor farklarını öngör. Risk taşıyan ağırlık "
        "kararlarını açıkça not et."
    ),
    backstory=(
        "Sen CLV ve dönüşüm olasılığı modellemesinde deneyimli bir data scientistsin. "
        "Tek bir metriğin (örn. harcama) tek başına skoru yönlendirmesinin yanıltıcı "
        "olduğunu bilirsin. Dengeli ağırlık dağılımıyla hem iş hedefini hem demografik "
        "adaleti bir arada düşünürsün."
    ),
    tools=[score_customers],
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

proxy_bias_agent = Agent(
    role="Proxy ve Bias Tespit Uzmanı",
    goal=(
        "Hassas demografik özellikleri (cinsiyet, yaş, gelir, şehir) dolaylı olarak "
        "temsil eden proxy değişkenleri Cramer's V ile tespit et. Aynı zamanda "
        "skorlama sonucundaki yüksek segmentlerde demografik dağılım farklarını "
        "ölç. İki tabloyu birleştirerek hangi bias zincirinin en kritik olduğunu "
        "açıkla."
    ),
    backstory=(
        "Sen algoritmik adalet ve istatistiksel ilişki analizinde uzmanlaşmış bir "
        "araştırmacısın. 'Cihaz türü' veya 'şehir' gibi 'masum' görünen "
        "değişkenlerin tarihsel eşitsizlikleri nasıl yeniden ürettiğini bilirsin. "
        "Bir bias'ı sadece sayısal olarak değil, neden oluştuğu hangi grubu "
        "etkilediğiyle birlikte açıklarsın."
    ),
    tools=[detect_proxy_and_bias],
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

critic_agent = Agent(
    role="Çapraz Ajan Eleştirmeni",
    goal=(
        "Tüm önceki ajanların öz-değerlendirmelerini, kararlarını ve bias katkı "
        "skorlarını doğrula. Çelişkileri bul, hangi ajanın hangi bias'a en çok "
        "katkı verdiğini sırala. Eleştiriyi sayısal kanıtla destekle: "
        "Counterfactual Test aracıyla farklı skorlama senaryolarını karşılaştır "
        "ve en savunulabilir senaryoyu öner."
    ),
    backstory=(
        "Sen bağımsız bir denetçisin — bir kararın yalnızca sonucuna değil, "
        "üretildiği sürece bakarsın. Diğer ajanların farkında olmadan yaptığı "
        "varsayımları sayısal kanıtla ifşa edersin. Amacın sistemi yıkmak değil, "
        "daha şeffaf ve savunulabilir hale getirmek; her itirazını "
        "counterfactual veriyle desteklersin."
    ),
    tools=[inter_agent_critique, counterfactual_test],
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

corrected_scoring_agent = Agent(
    role="İteratif Düzeltme Uzmanı",
    goal=(
        "Eleştirmenin önerilerini uygulayarak skorlamayı düzelt, sonucun yeterli "
        "olmadığı durumda eleştirmen + düzeltme döngüsünü tekrarla. Sistem "
        "demografik bias eşiğinin altına inene ya da maksimum tur sayısına "
        "ulaşana kadar iteratif çalış. Her turun sonucunu sayısal olarak göster."
    ),
    backstory=(
        "Sen geri besleme döngülerinin gerçek değerini bilen bir optimizasyon "
        "uzmanısın. Bir sistemin kendini düzeltebilmesi için tek bir pasın çoğu "
        "zaman yetmediğini, ardışık turların gerektiğini bilirsin. Her düzeltme "
        "turunu ölçer, bias hâlâ eşik üstündeyse eleştirmenden yeni öneri "
        "isteyip yeniden uygularsın — bir convergence noktası bulana kadar."
    ),
    tools=[corrected_scoring, iterative_correction],
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

final_report_agent = Agent(
    role="Final Optimizasyon ve Rapor Uzmanı",
    goal=(
        "Tüm pipeline çıktılarını — temizleme, segmentasyon, skorlama, bias/proxy "
        "tespiti, eleştirmen değerlendirmesi, düzeltilmiş skorlama — bütünleştirerek "
        "final hedef müşteri kitlesini oluştur. İki rapor üret: (1) Süreç Raporu "
        "(ajan bias katkı sıralaması + en etkili düzeltme), (2) Optimal Kitle "
        "Raporu (demografik profil + pazarlama önerileri)."
    ),
    backstory=(
        "Sen strateji ve veri analizini birleştiren bir karar destek uzmanısın. "
        "Sadece en yüksek skoru almak değil, sürdürülebilir, etik ve "
        "savunulabilir bir hedefleme oluşturmak önceliğindir. Pazarlama ekibi "
        "için somut, gerekçeli aksiyonlar üretirsin."
    ),
    tools=[build_final_and_report],
    llm=llm,
    verbose=True,
    allow_delegation=False,
)


# ══════════════════════════════════════════════════════════════════════════════
#  GÖREVLER
# ══════════════════════════════════════════════════════════════════════════════

task_clean = Task(
    description=(
        "Ham müşteri verisini analiz et ve temizle.\n\n"
        "1. `Veri Temizleme Ajanı` aracını çalıştır.\n"
        "   - İsteğe bağlı yaş sınırı kuralı JSON olarak verilebilir.\n"
        "   - Örn: {\"yas_min\": 13, \"yas_max\": 75}.\n"
        "2. Aracın çıktısındaki `oz_degerlendirme.demografik_kaymalar` bölümünü "
        "yorumla — hangi kolon dağılımı en çok değişti? Bu hangi temizleme "
        "kararının sonucu olabilir?\n"
        "3. `oz_degerlendirme.bias_katki_skoru` 0.3'ün üstündeyse açıkça uyar.\n\n"
        "Çıktı: Temizleme raporu + bias risk değerlendirmesi."
    ),
    expected_output=(
        "Yapılan temizleme kararları listesi, başlangıç/bitiş satır sayısı, "
        "demografik kaymalar, en riskli karar, bias_katki_skoru ve sözel yorum."
    ),
    agent=data_cleaning_agent,
)

task_segment = Task(
    description=(
        "Temizlenmiş veriyi segmentlere ayır.\n\n"
        "1. `Segmentasyon Ajanı` aracını çalıştır.\n"
        "2. Demografik temsil analizinde hangi grupların aşırı/eksik temsil "
        "edildiğini ifşa et.\n"
        "3. Oyun şirketinin hedef profili (Strateji/RPG türü, mobil/PC platformu) "
        "açısından hangi segmentler değerli, hangileri risk?\n\n"
        "Çıktı: Segment profilleri + demografik denge raporu."
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
        "   - Ağırlıkları JSON olarak verebilirsin; vermezsen eşit ağırlık.\n"
        "   - Örnek (hedef: harcama vurgusu): {\"aylik_ortalama_harcama\": 0.4, "
        "     \"haftalik_oyun_saati\": 0.2, ...}.\n"
        "2. Aracın `oz_degerlendirme.demografik_skor_farklari` bölümünü yorumla — "
        "hangi grupta skor farkı yüksek? Hangi metrik skoru domine ediyor?\n"
        "3. `bias_katki_skoru` 0.3'ü aşıyorsa hangi ağırlık değişikliğinin riski "
        "azaltacağını öner.\n\n"
        "Çıktı: Skorlama modeli + bias risk değerlendirmesi."
    ),
    expected_output=(
        "Kullanılan ağırlıklar, skor dağılımı, demografik grup ortalamaları, "
        "en riskli ağırlık, bias_katki_skoru ve gerekçeli yorum."
    ),
    agent=scoring_agent,
    context=[task_clean, task_segment],
)

task_proxy_bias = Task(
    description=(
        "Proxy değişkenleri ve demografik bias'ı birlikte tespit et.\n\n"
        "1. `Proxy ve Bias Tespit Ajanı` aracını çalıştır.\n"
        "2. `proxy_analizi` listesindeki YÜKSEK riskli değişkenler için: bu ilişki "
        "neden oluşuyor (sosyoekonomik, coğrafi)? Bu değişkenin skorlamada "
        "kullanılması adil mi?\n"
        "3. `bias_tespiti` bölümündeki demografik fark sayılarını yorumla.\n\n"
        "Çıktı: Proxy + bias birleşik raporu + öneriler."
    ),
    expected_output=(
        "Proxy listesi (Cramer's V değeriyle), demografik segment farkları, "
        "hangi değişkenin tutulmalı / ağırlığı azaltılmalı / çıkarılmalı önerisi."
    ),
    agent=proxy_bias_agent,
    context=[task_score],
)

task_critique = Task(
    description=(
        "Önceki ajanların kararlarını sayısal kanıtla denetle.\n\n"
        "1. `Çapraz Ajan Değerlendirme Ajanı` aracını çalıştır — pipeline log'unu "
        "okur, her ajanın bias katkısını sıralar, düzeltme önerileri üretir.\n"
        "2. `Counterfactual Test Aracı` aracını çalıştır — 4 senaryo karşılaştır "
        "(mevcut / eşit_agirlik / harcama_20 / harcama_10) ve `en_iyi_senaryo`'yu "
        "tespit et.\n"
        "3. İki çıktıyı birleştir: hangi ajan hangi bias'a katkıda bulundu, "
        "hangi counterfactual senaryosu en düşük bias verdi, hangi düzeltme "
        "yapılmalı?\n\n"
        "Çıktı: Sayısal kanıtlı eleştiri raporu + 2-3 somut düzeltme kuralı."
    ),
    expected_output=(
        "Ajan bias katkı sıralaması, counterfactual karşılaştırma tablosu, "
        "en iyi senaryo + max demografik fark, 2-3 somut düzeltme kuralı."
    ),
    agent=critic_agent,
    context=[task_clean, task_segment, task_score, task_proxy_bias],
)

task_corrected = Task(
    description=(
        "Eleştirmenin önerdiği düzeltmeyi uygula ve gerekirse iteratif tekrar et.\n\n"
        "1. `İteratif Öz-Düzeltme Loop` aracını çalıştır.\n"
        "   - Default 3 tura kadar critic → corrected_scoring döngüsü yapar.\n"
        "   - Max demografik fark 5 puanın altına inerse erken sonlandırır.\n"
        "   - max_iterations parametresi ile sınırı değiştirebilirsin (1-5).\n"
        "2. Aracın çıktısındaki `iterasyonlar` listesini yorumla:\n"
        "   - Her turda max demografik fark nasıl değişti?\n"
        "   - Hangi turda eşiğin altına inildi (veya inilemedi)?\n"
        "   - `erken_sonlandi` true mu, max iter'a mı çarpıldı?\n"
        "3. `son_max_fark` hâlâ eşik üstündeyse: hangi yapısal sınırın "
        "convergence'i engellediğini açıkla (örn. tek alternatif öneri 'eşit "
        "ağırlık', daha radikal bir müdahale gerekebilir).\n\n"
        "Çıktı: İteratif düzeltme raporu + convergence analizi."
    ),
    expected_output=(
        "Tur sayısı, başlangıç/son max demografik fark, her tur için iyileşme "
        "puanı ve uygulanan ağırlıklar, convergence durumu (✓/⚠) ile sözel yorum."
    ),
    agent=corrected_scoring_agent,
    context=[task_critique],
)

task_final = Task(
    description=(
        "Final hedef müşteri kitlesini oluştur ve iki rapor üret.\n\n"
        "1. `Final Optimizasyon ve Rapor Ajanı` aracını çalıştır.\n"
        "   - Minimum skor eşiğini belirle (default 50).\n"
        "   - Kriterleri JSON olarak verebilirsin: {\"min_skor\": 55}.\n"
        "2. `surec_raporu`'ndaki ajan bias katkı sıralamasını ve en etkili "
        "düzeltmeyi öne çıkar.\n"
        "3. `optimal_kitle_raporu`'ndaki demografik profili pazarlama ekibi için "
        "yorumla: kime, hangi kanalla, hangi mesajla ulaşılmalı?\n\n"
        "Çıktı: Pazarlama ekibine teslim edilebilir final rapor."
    ),
    expected_output=(
        "Final kitle boyutu, elenme oranı, demografik profil, ortalama skor, "
        "süreç özeti, en etkili düzeltme ve 3-5 somut pazarlama önerisi."
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
