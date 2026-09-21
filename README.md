# AI Research Assistant

Bu proje, teknik dokümanlarla çalışabilen, üretim kalitesine hazır (production-ready) modüler bir **AI Araştırma Asistanı** sistemidir.

## Özellikler
- **Tools & Data:** Yapılandırılmış araçlar (hesap makinesi, tarih) ve örnek doküman yönetimi.
- **RAG Katmanı:** Yerel doküman parçalama ve sorgu tabanlı bilgi getirme.
- **Agent Katmanı:** Akıllı karar mekanizması ve kaynak atıflı yanıt üretimi.
- **Backend (API):** FastAPI tabanlı güvenli REST uç noktaları ve SQLite veri tabanı kalıcılığı.
- **Frontend & Deployment:** Streamlit tabanlı kullanıcı arayüzü, Docker ve Docker Compose desteği.

## Projeyi Çalıştırma Rehberi

### 1. Bağımlılıkların Yüklenmesi
Projeyi çalıştırmadan önce gerekli Python kütüphanelerini yükleyin:
```bash
pip install -r requirements.txt
```

### 2. Testlerin Çalıştırılması
Tüm birim ve entegrasyon testlerini doğrulamak için:
```bash
pytest
```

### 3. Backend Servisinin Başlatılması
FastAPI arka uç sunucusunu başlatmak için:
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Frontend Arayüzünün Başlatılması
Yeni bir terminal açarak Streamlit arayüzünü çalıştırın:
```bash
streamlit run frontend/app.py
```

### 5. Docker ile Çalıştırma
Projeyi kapsayıcı (container) olarak tek komutla ayağa kaldırmak için:
```bash
docker-compose up --build
```
