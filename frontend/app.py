from __future__ import annotations

import hashlib
import os
from pathlib import Path

import requests
import streamlit as st


# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="AI Research Asistanı",
    page_icon="🤖",
    layout="wide",
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://localhost:8000",
)

SAMPLES_DIR = Path(
    os.getenv(
        "RAG_SAMPLES_DIR",
        "data/samples",
    )
)


# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_signatures" not in st.session_state:
    st.session_state.uploaded_signatures = set()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def file_signature(uploaded_file) -> str:
    """
    Create a stable signature for the uploaded file.

    This prevents the same Streamlit-uploaded file from being written
    repeatedly during later app reruns.
    """
    data = uploaded_file.getvalue()

    digest = hashlib.sha256(data).hexdigest()

    return (
        f"{uploaded_file.name}:"
        f"{len(data)}:"
        f"{digest}"
    )


def safe_filename(filename: str) -> str:
    """
    Prevent path traversal and keep only the actual filename.
    """
    return Path(filename).name


def save_uploaded_file(uploaded_file) -> Path:
    SAMPLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = safe_filename(
        uploaded_file.name
    )

    file_path = SAMPLES_DIR / filename

    with file_path.open("wb") as file:
        file.write(
            uploaded_file.getbuffer()
        )

    return file_path


@st.cache_data(ttl=5)
def check_backend(backend_url: str) -> bool:
    """
    Lightweight cached backend health check.
    """
    try:
        response = requests.get(
            f"{backend_url}/health",
            timeout=2,
        )

        return response.status_code == 200

    except requests.RequestException:
        return False


def get_existing_files() -> list[str]:
    """
    Return supported documents currently stored in data/samples.
    """
    SAMPLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    supported_extensions = {
        ".pdf",
        ".md",
        ".txt",
    }

    files = [
        path.name
        for path in SAMPLES_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower()
        in supported_extensions
    ]

    return sorted(
        files,
        key=str.lower,
    )


# ---------------------------------------------------------------------
# Sidebar / document management
# ---------------------------------------------------------------------

@st.fragment
def document_sidebar() -> None:
    st.header("📂 Doküman Kütüphanesi")

    # -----------------------------------------------------------------
    # Backend status
    # -----------------------------------------------------------------

    backend_online = check_backend(
        BACKEND_URL
    )

    if backend_online:
        st.success(
            "Backend Durumu: Çevrimiçi 🟢"
        )
    else:
        st.error(
            "Backend Durumu: Bağlı Değil 🔴"
        )

    st.divider()

    # -----------------------------------------------------------------
    # Existing documents
    # -----------------------------------------------------------------

    st.subheader("Mevcut Dosyalar")

    existing_files = get_existing_files()

    if existing_files:
        for filename in existing_files:
            st.text(
                f"📄 {filename}"
            )
    else:
        st.info(
            "Henüz yüklenmiş dosya yok."
        )

    st.divider()

    # -----------------------------------------------------------------
    # Upload
    # -----------------------------------------------------------------

    st.subheader("Yeni Dosya")

    uploaded_file = st.file_uploader(
        "PDF, Markdown veya TXT yükleyin",
        type=["pdf", "md", "txt"],
        key="document_uploader",
    )

    if uploaded_file is not None:

        signature = file_signature(
            uploaded_file
        )

        # -------------------------------------------------------------
        # Check whether this exact file was already processed
        # -------------------------------------------------------------

        if (
            signature
            not in st.session_state.uploaded_signatures
        ):

            try:

                # -----------------------------------------------------
                # 1. Save uploaded file
                # -----------------------------------------------------

                file_path = save_uploaded_file(
                    uploaded_file
                )

                st.info(
                    f"'{file_path.name}' kaydedildi."
                )

                # -----------------------------------------------------
                # 2. Index documents
                # -----------------------------------------------------

                with st.spinner(
                    "Doküman RAG indeksine ekleniyor..."
                ):

                    index_response = requests.post(
                        f"{BACKEND_URL}/documents/index",
                        timeout=180,
                    )

                # -----------------------------------------------------
                # 3. Indexing result
                # -----------------------------------------------------

                if index_response.status_code == 200:

                    st.session_state.uploaded_signatures.add(
                        signature
                    )

                    st.success(
                        f"'{file_path.name}' başarıyla yüklendi "
                        "ve RAG indeksine eklendi. ✅"
                    )

                    # Refresh cached backend/file information
                    st.cache_data.clear()

                else:

                    st.error(
                        "Dosya kaydedildi ancak RAG indeksleme "
                        "başarısız oldu."
                    )

                    st.error(
                        f"Backend "
                        f"({index_response.status_code}): "
                        f"{index_response.text}"
                    )

            except OSError as exc:

                st.error(
                    f"Dosya kaydedilemedi: {exc}"
                )

            except requests.Timeout:

                st.error(
                    "Doküman indeksleme işlemi "
                    "çok uzun sürdü."
                )

            except requests.ConnectionError:

                st.error(
                    "Backend servisine bağlanılamadı."
                )

            except requests.RequestException as exc:

                st.error(
                    f"İndeksleme isteği başarısız: {exc}"
                )

        else:

            st.caption(
                "Bu dosya zaten yüklendi ve indekslendi."
            )
with st.sidebar:
    document_sidebar()


# ---------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------

st.title(
    "🤖 AI Research Asistanı"
)

st.markdown(
    "Teknik dokümanlar üzerinde araştırma yapın, "
    "dosya yükleyin ve yapay zeka destekli yanıtlar alın."
)

st.divider()


# ---------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(
        message["role"]
    ):
        st.markdown(
            message["content"]
        )

        if message.get(
            "tool_output"
        ):
            st.info(
                f"**Araç Çıktısı:** "
                f"{message['tool_output']}"
            )

        sources = message.get(
            "sources",
            [],
        )

        if sources:
            with st.expander(
                "📚 Kaynaklar ve Atıflar"
            ):
                for source in sources:
                    source_name = source.get(
                        "source",
                        "Bilinmeyen kaynak",
                    )

                    page = source.get(
                        "page"
                    )

                    score = source.get(
                        "score"
                    )

                    if page:
                        st.markdown(
                            f"- **{source_name}**, "
                            f"sayfa {page} "
                            f"(Skor: {score})"
                        )
                    else:
                        st.markdown(
                            f"- **{source_name}** "
                            f"(Skor: {score})"
                        )


# ---------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------

user_query = st.chat_input(
    "Bir soru sorun..."
)

if user_query:
    user_query = user_query.strip()

    if not user_query:
        st.warning(
            "Lütfen geçerli bir soru girin."
        )
        st.stop()

    # Save/display user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query,
        }
    )

    with st.chat_message("user"):
        st.markdown(
            user_query
        )

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner(
            "Yapay zeka araştırıyor..."
        ):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={
                        "query": user_query
                    },
                    timeout=120,
                )

                if response.status_code == 200:
                    data = response.json()

                    answer = data.get(
                        "answer",
                        "Yanıt alınamadı.",
                    )

                    tool_output = data.get(
                        "tool_output"
                    )

                    sources = data.get(
                        "sources",
                        [],
                    )

                    st.markdown(
                        answer
                    )

                    if tool_output:
                        st.info(
                            f"**Araç Çıktısı:** "
                            f"{tool_output}"
                        )

                    if sources:
                        with st.expander(
                            "📚 Kaynaklar ve Atıflar"
                        ):
                            for source in sources:
                                source_name = source.get(
                                    "source",
                                    "Bilinmeyen kaynak",
                                )

                                page = source.get(
                                    "page"
                                )

                                score = source.get(
                                    "score"
                                )

                                if page:
                                    st.markdown(
                                        f"- **{source_name}**, "
                                        f"sayfa {page} "
                                        f"(Skor: {score})"
                                    )
                                else:
                                    st.markdown(
                                        f"- **{source_name}** "
                                        f"(Skor: {score})"
                                    )

                    # Save assistant message
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "tool_output": tool_output,
                            "sources": sources,
                        }
                    )

                else:
                    error_text = (
                        f"Sunucu Hatası "
                        f"({response.status_code}): "
                        f"{response.text}"
                    )

                    st.error(
                        error_text
                    )

            except requests.Timeout:
                st.error(
                    "Backend yanıt vermek için "
                    "çok uzun sürdü."
                )

            except requests.ConnectionError:
                st.error(
                    "Backend servisine bağlanılamadı."
                )

            except requests.RequestException as exc:
                st.error(
                    f"İstek hatası: {exc}"
                )

            except ValueError:
                st.error(
                    "Backend geçersiz bir JSON yanıtı döndürdü."
                )
