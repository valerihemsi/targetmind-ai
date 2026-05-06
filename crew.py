"""
Bias-Aware Multi-Agent Data Optimization System
Oyun şirketi için 7 ajanlı müşteri hedefleme sistemi.
"""
from crewai import Agent, Crew, LLM, Process, Task
from tools.data_tools import (
    raw_data_summary,
    clean_data,
    segmentation_analysis,
    score_customers,
    detect_bias,
    detect_proxy_variables,
    build_final_targets,
    # Yeni denetim araçları
    audit_cleaning_decisions,
    bias_threshold_check,
    counterfactual_test,
)

llm = LLM(model="anthropic/claude-sonnet-4-6")


# ══════════════════════════════════════════════════════════════════════════════
#  AJANLAR
# ══════════════════════════════════════════════════════════════════════════════

data_cleaning_agent = Agent(
    role="Veri Temizleme Uzmanı",
    goal=(
        "Ham müşteri verisindeki hataları, eksiklikleri, tekrarları ve "
        "format tutarsızlıklarını tespit edip temizle. "
        "Temizleme sonrası verinin analiz için hazır olduğunu doğrula."
    ),
    backstory=(
        "Sen veri kalitesi konusunda uzmanlaşmış bir analistsin. "
        "Kirli veri üzerinde yapılan analizlerin yanıltıcı sonuçlar doğurduğunu "
        "çok iyi bilirsin. Her temizleme kararını gerekçelendirir, "
        "hangi satırın neden elendiğini veya dönüştürüldüğünü kayıt altına alırsın."
    ),
    tools=[raw_data_summary, clean_data],
    llm=llm,
    verbose=True,
)

segmentation_agent = Agent(
    role="Müşteri Segmentasyon Uzmanı",
    goal=(
        "Temizlenmiş veriyi anlamlı müşteri segmentlerine ayır. "
        "Oyun şirketinin hedef kitlesi için en belirleyici segmentasyon "
        "boyutlarını belirle: platform, tür tercihi, harcama eğilimi, aktivite."
    ),
    backstory=(
        "Sen oyun sektörü müşteri davranışlarını derinlemesine anlayan "
        "bir segmentasyon analistsin. Farklı oyuncu profillerini — "
        "hardcore gamer'dan casual kullanıcıya, mobile spender'dan "
        "PC entuziyastına — tanırsın ve her segmentin pazarlama değerini "
        "farklı kriterlere göre değerlendirirsin."
    ),
    tools=[segmentation_analysis],
    llm=llm,
    verbose=True,
)

scoring_agent = Agent(
    role="Müşteri Skorlama Uzmanı",
    goal=(
        "Her müşteriye 0-100 arası bir potansiyel skor ver. "
        "Skor; oyun saati, harcama geçmişi, platform uyumu, tür tercihi "
        "ve kampanya etkileşimini dengeli biçimde yansıtmalı. "
        "Ağırlıkları oyun şirketinin amacına göre optimize et."
    ),
    backstory=(
        "Sen müşteri yaşam boyu değeri (CLV) ve dönüşüm olasılığı modellemesinde "
        "deneyimli bir data scientistsin. Tek bir metriğe odaklanmanın "
        "yanıltıcı olduğunu bilirsin — dengeli bir skorlama modeli oluşturmak "
        "için birden fazla sinyali bir araya getirirsin."
    ),
    tools=[score_customers, bias_threshold_check],
    llm=llm,
    verbose=True,
)

bias_detection_agent = Agent(
    role="Bias Tespit Uzmanı",
    goal=(
        "Skorlama modelinin cinsiyet, yaş, gelir ve şehir bazında "
        "ayrımcı sonuçlar üretip üretmediğini ölç. "
        "Gerçek performans farkını yapay bias'tan ayırt et. "
        "Demographic Parity ve Equal Opportunity metriklerini uygula."
    ),
    backstory=(
        "Sen algoritmik adalet ve veri etiği alanında uzmanlaşmış "
        "bir araştırmacısın. Sistemlerin 'objektif' göründüğü halde "
        "tarihsel eşitsizlikleri nasıl yeniden ürettiğini bilirsin. "
        "Bias'ı sadece tespit etmekle kalmaz, hangi kararı etkilediğini "
        "ve neden sorun oluşturduğunu da açıklarsın."
    ),
    tools=[detect_bias, audit_cleaning_decisions],
    llm=llm,
    verbose=True,
)

proxy_detection_agent = Agent(
    role="Proxy Değişken Tespiti Uzmanı",
    goal=(
        "Doğrudan kullanılmayan ama hassas demografik özellikleri dolaylı "
        "olarak temsil eden değişkenleri bul. "
        "Cihaz türü, platform, şehir, abonelik gibi değişkenlerin "
        "cinsiyet veya gelirle olan gizli ilişkisini ölç."
    ),
    backstory=(
        "Sen istatistiksel ilişki analizi ve özellik mühendisliğinde "
        "deneyimli bir analistsin. Bir özelliğin neden 'masum' "
        "görünse de bias üretebileceğini anlarsın — örneğin "
        "belirli bir şehrin seçilmesi, düşük gelir grubunun "
        "sistematik olarak dışlanmasına yol açabilir."
    ),
    tools=[detect_proxy_variables],
    llm=llm,
    verbose=True,
)

critic_agent = Agent(
    role="Sistem Eleştirmeni",
    goal=(
        "Sistemin verdiği her büyük eleme kararına itiraz et. "
        "'Bu müşteri neden elendin?', 'Bu skor hangi varsayıma dayanıyor?', "
        "'Aynı davranışı gösteren başka bir grup neden farklı muamele gördü?' "
        "sorularını sor. Örtülü varsayımları ifşa et."
    ),
    backstory=(
        "Sen bir kararın yalnızca sonucuna değil, karar sürecine bakan "
        "bağımsız bir denetçisin. Veri bilimcilerin ve analistlerin "
        "farkında olmadan yaptığı varsayımları görürsün. "
        "Amacın sistemi yıkmak değil — daha şeffaf ve savunulabilir "
        "hale getirmek."
    ),
    tools=[counterfactual_test, bias_threshold_check],
    llm=llm,
    verbose=True,
)

reoptimization_agent = Agent(
    role="Yeniden Optimizasyon Uzmanı",
    goal=(
        "Tüm analizleri — temizleme, skorlama, bias ve proxy raporlarını, "
        "eleştirmenlerin itirazlarını — bütünleştirerek final hedef müşteri "
        "kitlesini oluştur. Daha adil, şeffaf ve etkili bir hedefleme "
        "stratejisi öner. Final CSV'yi kaydet."
    ),
    backstory=(
        "Sen strateji ve veri analizini birleştiren bir optimizasyon uzmanısın. "
        "Sadece en yüksek skoru almak değil, sürdürülebilir, etik ve "
        "savunulabilir bir hedefleme modeli oluşturmak önceliğindir. "
        "Son kararı alırken hem iş hedeflerini hem de bias risklerini "
        "dengede tutarsın."
    ),
    tools=[build_final_targets],
    llm=llm,
    verbose=True,
)


# ══════════════════════════════════════════════════════════════════════════════
#  GÖREVLER
# ══════════════════════════════════════════════════════════════════════════════

task_clean = Task(
    description=(
        "Ham müşteri verisini analiz et ve temizle.\n\n"
        "1. `Ham Veri Özeti` aracını çalıştır — kaç satır, hangi sütunlar, "
        "kaç eksik değer, hangi format sorunları var?\n"
        "2. Bulguları değerlendirerek `Veri Temizleme İşlemi` aracını çalıştır.\n"
        "   - Yaş sınırlarını belirle (oyun oynayabilecek yaş aralığı)\n"
        "   - Cinsiyet format haritalamasını tanımla\n"
        "   - Diğer kurallara karar ver\n"
        "   - Araç çağrısında kuralları JSON olarak ver\n"
        "3. Temizleme raporunu yorumla: kaç satır elendi, neden?\n\n"
        "Çıktı: Temizleme kararlarının gerekçeli raporu."
    ),
    expected_output=(
        "Temizleme raporu: başlangıç/bitiş satır sayısı, yapılan işlemler listesi, "
        "eleme gerekçeleri ve temizlenmiş verinin demografik özeti."
    ),
    agent=data_cleaning_agent,
)

task_segment = Task(
    description=(
        "Temizlenmiş veriyi segmentlere ayır.\n\n"
        "1. `Segmentasyon Analizi` aracını çalıştır.\n"
        "2. Oyun şirketinin stratejik hedefleri için en değerli segmentleri belirle:\n"
        "   - Hedef oyun: Strateji/RPG türü, mobil/PC platformu\n"
        "   - Yüksek değerli segment kriterleri neler?\n"
        "   - Hangi segmentler kesinlikle hedef dışı?\n"
        "3. Her segment için kısa profil yaz.\n\n"
        "Çıktı: Segment tanımları ve öncelik sıralaması."
    ),
    expected_output=(
        "Her segment için: boyut, ortalama harcama, platform dağılımı, "
        "öncelik seviyesi (yüksek/orta/düşük) ve gerekçe."
    ),
    agent=segmentation_agent,
    context=[task_clean],
)

task_score = Task(
    description=(
        "Her müşteriye potansiyel skoru ver ve bias eşiğini kontrol et.\n\n"
        "1. Segmentasyon raporundaki bulguları göz önünde bulundur.\n"
        "2. `Müşteri Skorlama` aracını çalıştır.\n"
        "   - Ağırlıkları oyun şirketinin amacına göre ayarla\n"
        "   - Strateji/RPG tercihine yüksek ağırlık ver\n"
        "   - Ağırlıkları JSON olarak araç çağrısında belirt\n"
        "3. Skorlama bittikten HEMEN SONRA `Bias Eşiği Kontrolü` aracını çalıştır.\n"
        "   - Gelir grubu skor farkı 5 puanı aşıyor mu?\n"
        "   - Harcama ağırlığı skoru domine ediyor mu?\n"
        "   - Eşik aşıldıysa: uyarıyı raporla ve ağırlık düzeltme önerisi sun\n"
        "   - Eşik aşılmadıysa: skorlamayı onayla\n\n"
        "Çıktı: Skorlama modeli + bias eşiği kontrol sonucu."
    ),
    expected_output=(
        "Kullanılan ağırlıklar ve gerekçeleri, skor dağılımı, "
        "bias eşiği kontrol sonucu (🔴/🟡/🟢), "
        "eşik aşıldıysa ağırlık düzeltme önerisi."
    ),
    agent=scoring_agent,
    context=[task_clean, task_segment],
)

task_bias = Task(
    description=(
        "Hem skorlama modelindeki bias'ı hem de temizleme ajanının "
        "kararlarının bias etkisini denetle.\n\n"
        "ADIM 1 — Temizleme kararlarını denetle:\n"
        "1. `Temizleme Kararı Denetimi` aracını çalıştır.\n"
        "   - Cinsiyet mod dolgusu ('Erkek') demografiyi bozdu mu?\n"
        "   - Gelir mod dolgusu dağılımı nasıl değiştirdi?\n"
        "   - Bu kararlar aşağı akış bias'ına (downstream bias) yol açıyor mu?\n\n"
        "ADIM 2 — Skorlama bias'ını tespit et:\n"
        "2. `Bias Tespiti` aracını çalıştır.\n"
        "   - Cinsiyet, yaş, gelir ve şehir bazında seçilme oranlarını ölç\n"
        "   - Her fark için: gerçek performans farkı mı, "
        "yoksa temizleme/skorlama kaynaklı yapay bias mı?\n\n"
        "ADIM 3 — İki raporu birleştir:\n"
        "   - Temizleme biasları skorlama biaslarını nasıl besliyor?\n"
        "   - Hangi bias zinciri en kritik?\n\n"
        "Çıktı: Entegre bias raporu — kaynak zinciriyle birlikte."
    ),
    expected_output=(
        "Temizleme kararı denetim sonuçları (her karar için risk seviyesi), "
        "skorlama bias raporu, bias zinciri analizi: "
        "'Ajan 1 kararı X → Ajan 3'te Y etkisi → Z grubunu dışlıyor'."
    ),
    agent=bias_detection_agent,
    context=[task_score],
)

task_proxy = Task(
    description=(
        "Dolaylı bias üreten proxy değişkenleri tespit et.\n\n"
        "1. `Proxy Değişken Tespiti` aracını çalıştır.\n"
        "2. Yüksek ilişki gücü tespit edilen değişkenler için:\n"
        "   - Bu ilişki neden oluşuyor? (sosyoekonomik, coğrafi, vb.)\n"
        "   - Bu değişkenin modelde tutulması adil mi?\n"
        "   - Ağırlığının azaltılması veya kaldırılması önerilmeli mi?\n"
        "3. Her proxy değişken için risk ve öneri yaz.\n\n"
        "Çıktı: Proxy değişken listesi, risk açıklamaları, aksiyonlar."
    ),
    expected_output=(
        "Her proxy değişken için: hangi hassas özellikle ilişkili, "
        "Cramer V değeri, risk seviyesi, tutulmalı mı / ağırlığı azaltılmalı mı?"
    ),
    agent=proxy_detection_agent,
    context=[task_score, task_bias],
)

task_critic = Task(
    description=(
        "Sistemin kararlarını veriyle test ederek sorgula.\n\n"
        "Artık yalnızca metin eleştirisi yapma — sayısal olarak doğrula.\n\n"
        "ADIM 1 — Bias eşiğini yeniden kontrol et:\n"
        "1. `Bias Eşiği Kontrolü` aracını çalıştır.\n"
        "   - Scoring ajanının ürettiği skorlarda eşik hâlâ aşılıyor mu?\n"
        "   - Eşik aşılmışsa: bu bir kabul edilemez risk mi, "
        "yoksa iş hedefiyle meşrulaştırılabilir mi?\n\n"
        "ADIM 2 — Counterfactual test yap:\n"
        "2. `Counterfactual Test` aracını çalıştır.\n"
        "   - En az 3 senaryo: mevcut ağırlık, %20 harcama, %10 harcama\n"
        "   - Her senaryoda: gelir farkı, yüksek+prime müşteri sayısı, seçilme oranları\n"
        "   - Hangi senaryo en iyi bias/verimlilik dengesini sağlıyor?\n\n"
        "ADIM 3 — Veriyle desteklenmiş eleştiri yaz:\n"
        "3. Her itirazını counterfactual veya eşik testi sonuçlarıyla destekle\n"
        "4. Sistemin kör noktalarını tespit et:\n"
        "   - Temizleme → Skorlama bias zincirini kıran bir nokta var mı?\n"
        "   - 'Orta gelir' grubunun %0 seçilmesi kabul edilebilir mi?\n"
        "   - 'Diğer' cinsiyet grubunun sistematik dışlanması nasıl açıklanıyor?\n"
        "5. Final optimizasyon için counterfactual sonuçlarına dayanan "
        "3 somut kural öner\n\n"
        "Çıktı: Sayısal kanıtlı eleştiri raporu + 3 somut kural."
    ),
    expected_output=(
        "Bias eşiği kontrol sonucu, counterfactual karşılaştırma tablosu, "
        "her eleştiri için sayısal kanıt, "
        "sistemin kör noktaları, "
        "final optimizasyon için 3 veri destekli kural."
    ),
    agent=critic_agent,
    context=[task_clean, task_segment, task_score, task_bias, task_proxy],
)

task_reoptimize = Task(
    description=(
        "Tüm analizleri bütünleştir ve final hedef müşteri kitlesini oluştur.\n\n"
        "1. Eleştirmen raporundaki 3 kuralı ve bias bulgularını göz önünde bulundur.\n"
        "2. `Final Hedef Kitle Oluştur` aracını çalıştır:\n"
        "   - Minimum skor eşiğini belirle\n"
        "   - Maksimum son aktivite günü belirle (inaktif kullanıcıları ele)\n"
        "   - Hedef tür listesini belirle\n"
        "   - Kriterleri JSON olarak araç çağrısında ver\n"
        "3. Final raporu yaz:\n"
        "   - Kaç müşteriden kaçına indik? (elenme oranı)\n"
        "   - Final kitlenin demografik profili nedir?\n"
        "   - Bu hedefleme stratejisi neden hem etkili hem de adil?\n"
        "   - Pazarlama ekibine öneriler\n\n"
        "Çıktı: Final hedef müşteri raporu + pazarlama stratejisi önerileri."
    ),
    expected_output=(
        "Final hedef kitle boyutu, elenme oranı, demografik profil, "
        "ortalama skor ve harcama, platform/tür dağılımı, "
        "pazarlama ekibi için 3-5 somut öneri."
    ),
    agent=reoptimization_agent,
    context=[task_clean, task_segment, task_score, task_bias, task_proxy, task_critic],
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
            bias_detection_agent,
            proxy_detection_agent,
            critic_agent,
            reoptimization_agent,
        ],
        tasks=[
            task_clean,
            task_segment,
            task_score,
            task_bias,
            task_proxy,
            task_critic,
            task_reoptimize,
        ],
        process=Process.sequential,
        verbose=True,
    )
