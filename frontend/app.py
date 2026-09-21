import streamlit as st
import requests
import os
import glob

st.set_page_config(page_title="AI Araştırma Asistanı", page_icon="🤖", layout="wide")

st.title("🤖 AI Araştırma Asistanı")
st.markdown("Teknik dokümanlar üzerinde arama yapın, dosya yükleyin ve yapay zeka destekli yanıtlar alın.")

backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")

# --- Sidebar: Doküman Yönetimi ve Sistem Durumu ---
st.sidebar.header("📂 Doküman Kütüphanesi")

# 1. Sağlık Kontrolü
try:
    health_res = requests.get(f"{backend_url}/health", timeout=2)
    if health_res.status_code == 200:
        st.sidebar.success("Backend Durumu: Çevrimiçi 🟢")
    else:
        st.sidebar.error("Backend Durumu: Yanıt Vermiyor 🔴")
except Exception:
    st.sidebar.error("Backend Servisine Bağlanılamadı ❌")

# 2. Mevcut Dokümanları Listeleme
samples_dir = "data/samples"
os.makedirs(samples_dir, exist_ok=True)
existing_files = []
if os.path.exists(samples_dir):
    existing_files = os.listdir(samples_dir)

if existing_files:
    st.sidebar.markdown("**Mevcut Dosyalar:**")
    for f in existing_files:
        st.sidebar.text(f"📄 {f}")
else:
    st.sidebar.info("Henüz yüklenmiş dosya yok.")

# 3. Dosya Yükleme Alanı
uploaded_file = st.sidebar.file_uploader("Yeni Dosya Yükle (PDF, MD, TXT)", type=["pdf", "md", "txt"])
if uploaded_file is not None:
    file_path = os.path.join(samples_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success(f"'{uploaded_file.name}' başarıyla eklendi!")
    st.rerun()

# --- Ana Ekran: Sohbet ve Sorgulama ---
st.markdown("---")

# Oturum Durumu (Chat History)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Geçmiş mesajları göster
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("tool_output"):
            st.info(f"**Araç Çıktısı:** {message['tool_output']}")
        if message.get("sources"):
            with st.expander("📚 Kaynaklar ve Atıflar"):
                for src in message["sources"]:
                    st.markdown(f"- **{src.get('source')}** (Skor: {src.get('score')})")

# Kullanıcı Girişi
if user_query := st.chat_input("Bir soru sorun (örn: 'architecture nedir?', '2 + 2 kaç yapar?', 'Bugün tarih ne?')"):
    # Kullanıcı mesajını ekle
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Asistan Yanıtı
    with st.chat_message("assistant"):
        with st.spinner("Yapay zeka araştırıyor ve yanıt üretiyor..."):
            try:
                response = requests.post(f"{backend_url}/chat", json={"query": user_query}, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "Yanıt alınamadı.")
                    tool_output = data.get("tool_output")
                    sources = data.get("sources", [])

                    st.markdown(answer)
                    
                    if tool_output:
                        st.info(f"**Araç Çıktısı:** {tool_output}")

                    if sources:
                        with st.expander("📚 Kaynaklar ve Atıflar"):
                            for src in sources:
                                st.markdown(f"- **{src.get('source')}** (Skor: {src.get('score')})")

                    # Oturuma kaydet
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "tool_output": tool_output,
                        "sources": sources
                    })
                else:
                    err_msg = f"Sunucu Hatası ({response.status_code}): {response.text}"
                    st.error(err_msg)
            except Exception as e:
                st.error(f"Bağlantı hatası oluştu: {str(e)}")
