# AI Research Assistant - Proje Uygulama Planı

Bu doküman, AI Research Assistant projesinin uzmanlar (specialist) bazında hangi sırayla geliştirileceğini, katmanlar arası bağımlılıkları ve sözleşme gereksinimlerini belirler.

## 1. Uzman Geliştirme Sırası

Proje, temel veri ve araçlardan başlayarak sırasıyla şu sırada geliştirilecektir:

1. **Tools & Data Specialist (Araç ve Veri Uzmanı)**
   - *Durum:* İlk örnek veriler (`data/samples/`) ve ground-truth veri seti oluşturuldu.
   - *Sonraki Adım:* Ajan tarafından kullanılacak yapılandırılmış araçların (hesap makinesi, arama vb.) şemaları ve fonksiyonları.
2. **Document Processing / RAG Specialist (Doküman İşleme ve RAG Uzmanı)**
   - *Bağımlılık:* Tools & Data tarafından sağlanan örnek dokümanlar (`data/samples/`).
   - *Görev:* Doküman parçalama (chunking), embedding üretimi ve vektör veritabanı indeksleme.
3. **Agent Specialist (Ajan Uzmanı)**
   - *Bağımlılık:* RAG retrieval arayüzü ve Tools/Data araç şemaları.
   - *Görev:* ReAct döngüsü, Gemini entegrasyonu ve kaynak atıflı (citation) yanıt üretimi.
4. **Backend / Database Specialist (Arka Uç ve Veritabanı Uzmanı)**
   - *Bağımlılık:* Agent ve RAG katmanlarının çalışır arayüzleri.
   - *Görev:* FastAPI endpoint'leri, oturum yönetimi ve veritabanı kalıcılığı (SQLite).
5. **Frontend / Deployment Specialist (Ön Uç ve Dağıtım Uzmanı)**
   - *Bağımlılık:* Backend API sözleşmeleri (`contracts/`).
   - *Görev:* Streamlit veya hafif web arayüzü ve Docker/dağıtım yapılandırması.

---

## 2. Katmanlar Arası Bağımlılıklar ve Ön Gereksinimler

- **Ön Koşul (Başlangıç):** `data/` altındaki ham dokümanlar ve test veri seti.
- **RAG Başlangıç Şartı:** `data/samples/` altındaki Markdown ve TXT dosyalarının hazır olması.
- **Agent Başlangıç Şartı:** RAG getirme fonksiyonunun (`retrieval`) ve araçların giriş/çıkış şemalarının netleşmesi.
- **Backend Başlangıç Şartı:** Agent ve RAG modüllerinin Python fonksiyonları olarak çağrılabilir olması.
- **Frontend Başlangıç Şartı:** Backend REST/WebSocket API'lerinin tamamlanmış olması.

---

## 3. Gerekli Sözleşmeler (`contracts/`)

Her katman bir sonraki katmana geçilmeden önce şu sözleşmeleri sağlamalıdır:
- `contracts/rag_contract.json` (veya `.py`): RAG modülünün kabul ettiği sorgu formatı ve döndürdüğü chunk/citation yapısı.
- `contracts/tools_contract.json`: Araçların Pydantic / JSON şemaları.
- `contracts/backend_api_contract.md`: Frontend'in tüketeceği API endpoint tanımları.

---

## 4. Bağımsız Geliştirilebilir Alanlar

- **Tools & Data** ve **RAG** katmanlarının ilk hazırlıkları (veri setleri ve indeksleme) diğer katmanlardan bağımsız olarak test edilebilir.
- **Frontend** tasarımı, Backend API sözleşmeleri (`contracts/backend_api_contract.md`) netleştiği andan itibaren mock verilerle paralelde geliştirilebilir.
