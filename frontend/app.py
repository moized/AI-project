from __future__ import annotations

import os

import requests
import streamlit as st


st.set_page_config(
    page_title="AI Research Asistanı",
    page_icon="🤖",
    layout="wide",
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = (5, 180)


@st.cache_data(ttl=5)
def get_backend_status() -> bool:
    try:
        response = requests.get(
            f"{BACKEND_URL}/health",
            timeout=REQUEST_TIMEOUT[0],
        )
        return response.ok
    except requests.RequestException:
        return False


@st.cache_data(ttl=5)
def get_documents() -> list[str]:
    try:
        response = requests.get(
            f"{BACKEND_URL}/api/v1/documents",
            timeout=REQUEST_TIMEOUT[0],
        )
        response.raise_for_status()
        return response.json().get("documents", [])
    except requests.RequestException as exc:
        st.error(f"Doküman listesi alınamadı: {exc}")
        return []


def upload_document(uploaded_file) -> bool:
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/v1/documents/upload",
            files={
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type or "application/octet-stream",
                )
            },
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return True
    except requests.HTTPError as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        st.error(f"Yükleme başarısız: {detail}")
    except requests.RequestException as exc:
        st.error(f"Backend'e bağlanılamadı: {exc}")
    return False


def index_documents() -> bool:
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/v1/documents/index",
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        result = response.json()

        if result.get("status") == "partial":
            st.warning(
                f"İndeksleme kısmen tamamlandı: {result.get('result')}"
            )
        else:
            st.success("Doküman indeksi güncellendi.")

        return True
    except requests.HTTPError as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        st.error(f"İndeksleme başarısız: {detail}")
    except requests.RequestException as exc:
        st.error(f"Backend'e bağlanılamadı: {exc}")
    return False


def render_documents() -> None:
    st.subheader("📂 Doküman Kütüphanesi")

    documents = get_documents()
    if documents:
        for name in documents:
            st.write(f"📄 {name}")
    else:
        st.info("Henüz doküman yok.")

    st.divider()

    uploaded_file = st.file_uploader(
        "PDF, Markdown veya TXT yükleyin",
        type=["pdf", "md", "txt"],
    )

    if uploaded_file is not None and st.button(
        "Yükle ve İndeksle",
        type="primary",
    ):
        with st.spinner("Doküman yükleniyor..."):
            uploaded = upload_document(uploaded_file)

        if uploaded:
            with st.spinner("RAG indeksi güncelleniyor..."):
                indexed = index_documents()

            if indexed:
                get_documents.clear()


with st.sidebar:
    if get_backend_status():
        st.success("Backend: Çevrimiçi 🟢")
    else:
        st.error("Backend: Bağlı Değil 🔴")

    render_documents()


st.title("🤖 AI Research Asistanı")
st.caption("Teknik dokümanlardan kaynaklı cevaplar ve kontrollü araç kullanımı.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        sources = message.get("sources", [])
        if sources:
            with st.expander("📚 Kaynaklar"):
                for source in sources:
                    location = source["source"]
                    if source.get("page"):
                        location += f", sayfa {source['page']}"
                    st.write(
                        f"- {location} (skor: {source['score']:.3f})"
                    )

        tool_calls = message.get("tool_calls", [])
        if tool_calls:
            with st.expander("🛠 Araç çağrıları"):
                for call in tool_calls:
                    st.write(
                        f"- {call['name']}: {call['arguments']}"
                    )


query = st.chat_input("Bir soru sorun...")

if query:
    query = query.strip()

    if not query:
        st.warning("Lütfen geçerli bir soru girin.")
        st.stop()

    st.session_state.messages.append(
        {"role": "user", "content": query}
    )

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"), st.spinner("Araştırılıyor..."):
        try:
            response = requests.post(
                f"{BACKEND_URL}/api/v1/chat",
                json={"query": query},
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()

            data = response.json()
            answer = data["answer"]
            sources = data.get("sources", [])
            tool_calls = data.get("tool_calls", [])

            st.markdown(answer)

            if sources:
                with st.expander("📚 Kaynaklar"):
                    for source in sources:
                        location = source["source"]
                        if source.get("page"):
                            location += f", sayfa {source['page']}"
                        st.write(
                            f"- {location} (skor: {source['score']:.3f})"
                        )

            if tool_calls:
                with st.expander("🛠 Araç çağrıları"):
                    for call in tool_calls:
                        st.write(
                            f"- {call['name']}: {call['arguments']}"
                        )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "tool_calls": tool_calls,
                }
            )

        except requests.HTTPError as exc:
            detail = exc.response.text if exc.response is not None else str(exc)
            st.error(f"Sunucu hatası: {detail}")
        except requests.Timeout:
            st.error("Backend isteği zaman aşımına uğradı.")
        except requests.RequestException as exc:
            st.error(f"Backend bağlantı hatası: {exc}")
        except ValueError:
            st.error("Backend geçersiz JSON döndürdü.")
