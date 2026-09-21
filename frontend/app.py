import streamlit as st
import requests

st.title("AI Araştırma Asistanı")

backend_url = "http://localhost:8000"

# Sağlık kontrolü
try:
    health_res = requests.get(f"{backend_url}/health")
    if health_res.status_code == 200:
        st.sidebar.success("Backend Servisi Çalışıyor (Online)")
    else:
        st.sidebar.error("Backend Servisi Yanıt Vermiyor")
except Exception:
    st.sidebar.error("Backend Servisine Bağlanılamadı")

query = st.text_input("Araştırmak istediğiniz konuyu veya soruyu yazın:")

if st.button("Gönder"):
    if query.strip():
        with st.spinner("Yapay zeka araştırıyor..."):
            try:
                response = requests.post(f"{backend_url}/chat", json={"query": query})
                if response.status_code == 200:
                    data = response.json()
                    st.subheader("Yanıt:")
                    st.write(data.get("answer", ""))
                    
                    if data.get("tool_output"):
                        st.info(f"Araç Çıktısı: {data.get('tool_output')}")
                        
                    sources = data.get("sources", [])
                    if sources:
                        st.subheader("Kaynaklar:")
                        for src in sources:
                            st.markdown(f"- **{src.get('source')}** (Skor: {src.get('score')})")
                else:
                    st.error(f"Hata oluştu: {response.text}")
            except Exception as e:
                st.error(f"Bağlantı hatası: {str(e)}")
    else:
        st.warning("Lütfen geçerli bir soru girin.")
