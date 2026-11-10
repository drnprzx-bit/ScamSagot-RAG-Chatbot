import streamlit as st
import requests
import time
from pathlib import Path

from rag_engine import (
    EMBEDDING_MODEL_NAME,
    build_index,
    corpus_sha256,
    create_chunks,
    extract_pages,
    has_relevant_evidence,
    ollama_chat_stream,
    retrieve,
)

st.set_page_config(page_title="ScamSagot", page_icon="⚖️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp { background: #0A0A0A; color: #F5F5F5; }
[data-testid="stHeader"] { background: rgba(10,10,10,.96); }
[data-testid="stSidebar"] { background: #111111; border-right: 1px solid #3A3A3A; }
.block-container { max-width: 1480px; padding-top: 3.5rem; padding-bottom: 2rem; }
.brand { display: flex; align-items: center; gap: 9px; color: #FFD700; font-weight: 800; font-size: 1.5rem; letter-spacing: .015em; }
.brand-mark { width: 28px; height: 28px; flex: 0 0 28px; color: #FFD700; }
.subtle { color: #C8C8C8; }
.hero { background: linear-gradient(115deg,#202020,#3A3A3A); border: 1px solid #4A4A4A; border-radius: 22px; padding: 1.6rem 1.8rem; margin: .6rem 0 1.2rem; }
.hero-kicker { color: #E5C100; text-transform: uppercase; font-size: .72rem; font-weight: 750; letter-spacing: .13em; }
.hero-title { color: #FFFFFF; font-size: clamp(1.6rem,3vw,2.7rem); line-height: 1.08; font-weight: 800; margin: .55rem 0 .45rem; }
.hero-copy { color: #E1E1E1; max-width: 680px; font-size: 1rem; }
.section-label { color: #E5C100; font-size: .72rem; font-weight: 750; letter-spacing: .12em; text-transform: uppercase; margin: 1rem 0 .5rem; }
[data-testid="stChatMessage"] { background: #222222; border: 1px solid #3A3A3A; border-radius: 15px; }
.conversation-scroll { height: 62vh; min-height: 420px; max-height: 760px; overflow-y: auto; padding: .25rem .65rem .75rem .1rem; scrollbar-color: #555 #171717; }
[data-testid="stChatInput"] { border: 1px solid #B8860B; border-radius: 14px; background: #222222; }
[data-testid="stChatInput"] textarea { color: #FFFFFF; }
[data-testid="stBottom"], [data-testid="stBottomBlockContainer"] { background: #0A0A0A !important; }
[data-testid="stBottomBlockContainer"] { max-width: 1100px !important; width: calc(100% - 2rem) !important; margin-left: auto !important; margin-right: auto !important; }
[data-testid="stChatInput"] > div { background: #222222 !important; border-color: #B8860B !important; }
[data-testid="stChatInput"] button { background: #3A3A3A !important; color: #F5F5F5 !important; }
[data-testid="stExpander"] { border: 1px solid #3A3A3A; border-radius: 12px; background: #161616; }
[data-testid="stFileUploader"] section { background: #202020; border: 1px dashed #B8860B; border-radius: 12px; }
.stButton > button { background: #242424; border: 1px solid #444444; color: #F5F5F5; border-radius: 12px; transition: .15s ease; }
.stButton > button:hover { border-color: #B8860B; color: #FFD700; }
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] { background: #FFD700; color: #0A0A0A; border: 1px solid #FFD700; font-weight: 750; }
.stButton > button[kind="primary"]:hover, .stButton > button[data-testid="stBaseButton-primary"]:hover { background: #E5C100; color: #0A0A0A; }
.source-heading { color: #FFFFFF; font-weight: 700; font-size: .95rem; }
.source-meta { color: #E5C100; font-size: .78rem; font-weight: 650; }
.disclaimer { color: #E5E5E5; background: #202020; border: 1px solid #3A3A3A; border-left: 3px solid #B8860B; border-radius: 10px; padding: 11px 13px; font-size: .82rem; line-height: 1.5; }
[data-testid="stAlert"] { background: #242424; border-color: #B8860B; color: #F5F5F5; }
@media (max-width: 900px) { .block-container { padding-left: 1rem; padding-right: 1rem; } }
</style>
""", unsafe_allow_html=True)

CORPUS_PATH = Path(__file__).resolve().parent / "data" / "ScamSagot_Consolidated_Reference_RAG.pdf"

CONFIGS = {
    "A_800_0": (800, 0),
    "B_800_100": (800, 100),
    "C_1200_0": (1200, 0),
    "D_1200_100": (1200, 100),
}
SUGGESTIONS = [
    "What rights are identified for financial consumers under R.A. 11765?",
    "What does R.A. 12010 say about financial account scams?",
    "What is the Transaction Pause Period under BSP Circular 1213?",
]

# Read endpoint/model from Streamlit secrets; never expose endpoint settings to app visitors.
try:
    OLLAMA_BASE_URL = str(st.secrets["OLLAMA_BASE_URL"]).strip().rstrip("/")
    OLLAMA_MODEL = str(st.secrets["OLLAMA_MODEL"]).strip()
except Exception:
    OLLAMA_BASE_URL = ""
    OLLAMA_MODEL = ""

@st.cache_resource(show_spinner="Loading embedding model…")
def get_embedding_model(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)

@st.cache_resource(show_spinner="Preparing corpus and FAISS index…")
def get_rag_resources(pdf_bytes: bytes, chunk_size: int, overlap: int):
    pages = extract_pages(pdf_bytes)
    chunks = create_chunks(pages, chunk_size, overlap)
    model = get_embedding_model(EMBEDDING_MODEL_NAME)
    index = build_index(chunks, model)
    return pages, chunks, index, model


def check_ollama(url: str, model: str):
    if not url.startswith(("http://", "https://")):
        return False, "Enter a reachable Ollama URL beginning with http:// or https://."
    if not model.strip():
        return False, "Enter the exact model identifier installed in Ollama."
    try:
        response = requests.get(f"{url.rstrip('/')}/api/tags", timeout=15)
        response.raise_for_status()
        names = [item.get("name", "") for item in response.json().get("models", [])]
        if model not in names:
            return False, "Ollama responded, but that exact model identifier was not listed by /api/tags."
        return True, "Ollama is reachable and the selected model is available."
    except requests.RequestException as exc:
        return False, f"Could not reach Ollama ({type(exc).__name__}). Check the URL, Ollama service, and network/tunnel access."
    except (ValueError, KeyError):
        return False, "Ollama returned an unexpected response from /api/tags."


def new_chat():
    st.session_state.messages = []
    st.session_state.active_chat_id += 1

if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = 1

with st.sidebar:
    st.markdown('<div class="brand"><svg class="brand-mark" viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round" role="img" aria-label="Scales of justice"><line x1="32" y1="10" x2="32" y2="52"/><circle cx="32" cy="8" r="2.5" fill="currentColor" stroke="none"/><path d="M20 56h24"/><path d="M24 56c0-3 3.5-4 8-4s8 1 8 4"/><path d="M10 16h44"/><path d="M10 16L3 38"/><path d="M10 16l13 22"/><path d="M54 16l-13 22"/><path d="M54 16l7 22"/><path d="M3 38h20c0 5.5-4.5 9-10 9s-10-3.5-10-9z" fill="currentColor" fill-opacity="0.15"/><path d="M41 38h20c0 5.5-4.5 9-10 9s-10-3.5-10-9z" fill="currentColor" fill-opacity="0.15"/></svg><span>ScamSagot</span></div><div class="subtle">Legal Info for a Safer You</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.button("＋  New chat", on_click=new_chat, use_container_width=True, type="primary")
    st.markdown('<div class="section-label">Current conversation</div>', unsafe_allow_html=True)
    st.caption(f"{len([m for m in st.session_state.messages if m['role']=='user'])} questions")
    st.markdown("---")
    st.markdown('<div class="section-label">Ollama connection</div>', unsafe_allow_html=True)
    if OLLAMA_BASE_URL and OLLAMA_MODEL:
        st.caption(f"Configured model: {OLLAMA_MODEL}")
        if st.button("Check connection", use_container_width=True):
            ok, message = check_ollama(OLLAMA_BASE_URL, OLLAMA_MODEL)
            (st.success if ok else st.error)(message)
    else:
        st.info("Add OLLAMA_BASE_URL and OLLAMA_MODEL in Streamlit Secrets to enable answers.")
    st.markdown("---")
    with st.expander("Corpus & retrieval settings", expanded=not st.session_state.get("rag_ready", False)):
        st.caption("Uses the corpus PDF bundled with this app. Choose a chunking configuration, then extract and index the PDF.")
        config_id = st.selectbox("Chunking configuration", list(CONFIGS.keys()), format_func=lambda key: f"{key} — {CONFIGS[key][0]} characters / {CONFIGS[key][1]} overlap")
        if st.button("Extract PDF and prepare retrieval", type="primary", use_container_width=True):
            try:
                if not CORPUS_PATH.is_file():
                    raise FileNotFoundError(f"Bundled corpus not found: {CORPUS_PATH.name}")
                pdf_bytes = CORPUS_PATH.read_bytes()
                size, overlap = CONFIGS[config_id]
                pages, chunks, faiss_index, embedding_model = get_rag_resources(pdf_bytes, size, overlap)
                st.session_state.rag_ready = True
                st.session_state.pdf_bytes = pdf_bytes
                st.session_state.config_id = config_id
                st.session_state.extracted_text = "\n\n".join(
                    f"Source: {page['source_document'] or 'Unidentified'}\n"
                    f"Combined corpus PDF page: {page['pdf_page']}\n{page['text']}"
                    for page in pages
                )
                st.success(f"Corpus ready · {len(chunks):,} chunks · {config_id}")
            except Exception as exc:
                st.session_state.rag_ready = False
                st.error(f"Could not extract or prepare the bundled corpus: {exc}")
        elif st.session_state.get("rag_ready") and st.session_state.get("config_id") == config_id:
            st.success(f"Corpus ready · {st.session_state.get('config_id')} · {CORPUS_PATH.name}")
        else:
            st.info(f"Bundled corpus: {CORPUS_PATH.name}. Select a configuration and choose Extract PDF to start.")

        if st.session_state.get("rag_ready") and st.session_state.get("extracted_text"):
            st.download_button(
                "Download extracted corpus text",
                data=st.session_state.extracted_text,
                file_name="ScamSagot_Extracted_Corpus.txt",
                mime="text/plain; charset=utf-8",
                use_container_width=True,
            )
    st.markdown("---")
    with st.expander("FAQs"):
        st.write("ScamSagot answers using passages retrieved from its bundled corpus. Ask a specific question and review its cited sources.")
    with st.expander("About"):
        st.write("ScamSagot is a RAG prototype for Philippine financial consumer protection, cybercrime, and scam-related legal information.")
    with st.expander("Disclaimer"):
        st.write("ScamSagot provides legal information based on Philippine laws. It is not a substitute for professional legal advice.")

st.markdown("""
<div class="hero">
  <div class="hero-kicker">Philippine legal information · RAG assistant</div>
  <div class="hero-title">Clear answers,<br>grounded in your sources.</div>
  <div class="hero-copy">Ask about the laws and regulations in the ScamSagot corpus. Review the passages behind each response in the Sources panel.</div>
</div>
""", unsafe_allow_html=True)

@st.dialog("Retrieved sources", width="large")
def show_sources(message_index: int):
    message = st.session_state.messages[message_index]
    st.caption(message.get("question", "Sources for this answer"))
    for source in message.get("sources", []):
        pdf_pages = ", ".join(map(str, source["pdf_pages"])) or "unavailable"
        source_pages = ", ".join(map(str, source["source_pages"])) or "not embedded / unavailable"
        with st.container(border=True):
            st.markdown(f"<div class='source-meta'>SOURCE {source['rank']}</div><div class='source-heading'>{source['source_document']}</div>", unsafe_allow_html=True)
            st.caption(f"Corpus PDF page(s): {pdf_pages} · Chunk {source['chunk_id']} · Score {source['faiss_score']:.3f}")
            with st.expander("View retrieved passage"):
                st.write(source["text"])
                st.caption(f"Original source page(s): {source_pages}")

st.markdown('<div class="section-label">Conversation</div>', unsafe_allow_html=True)
with st.container(height=560, border=False):
    if not st.session_state.messages:
        st.markdown('<div class="panel"><strong>Welcome to ScamSagot</strong><br><span class="subtle">Ask about the laws and regulations represented in the ScamSagot corpus. Each response is paired with the passages retrieved for your question.</span></div>', unsafe_allow_html=True)
    for message_index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                if message.get("elapsed_seconds") is not None:
                    st.caption(f"Completed in {message['elapsed_seconds']:.1f} seconds")
                if message.get("sources"):
                    if st.button(f"Sources · {len(message['sources'])}", key=f"sources_{st.session_state.active_chat_id}_{message_index}"):
                        show_sources(message_index)

st.markdown('<div class="section-label">Try asking</div>', unsafe_allow_html=True)
suggestion_cols = st.columns(len(SUGGESTIONS))
for col, suggestion in zip(suggestion_cols, SUGGESTIONS):
    with col:
        if st.button(suggestion, key=f"suggest_{st.session_state.active_chat_id}_{SUGGESTIONS.index(suggestion)}", use_container_width=True):
            st.session_state.pending_question = suggestion
            st.rerun()
typed_question = st.chat_input("Ask a question about the corpus…")

question = typed_question or st.session_state.pop("pending_question", None)
if question:
    if not st.session_state.get("rag_ready"):
        st.warning("Choose a chunking configuration and click Extract PDF and prepare retrieval first.")
    elif st.session_state.get("config_id") != config_id:
        st.warning("The chunking configuration changed. Click Extract PDF and prepare retrieval to rebuild the index.")
    elif not OLLAMA_BASE_URL or not OLLAMA_MODEL:
        st.warning("Ollama is not configured. Add OLLAMA_BASE_URL and OLLAMA_MODEL in Streamlit Secrets.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.container():
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                request_started = time.perf_counter()
                try:
                    with st.status("Retrieving evidence…", expanded=False) as status:
                        size, overlap = CONFIGS[st.session_state.config_id]
                        pages, chunks, faiss_index, embedding_model = get_rag_resources(st.session_state.pdf_bytes, size, overlap)
                        sources = retrieve(question, chunks, faiss_index, embedding_model)
                        if not has_relevant_evidence(sources):
                            answer = (
                                "I can answer questions when the ScamSagot corpus contains relevant evidence "
                                "about Philippine financial consumer protection, cybercrime, or scam-related laws "
                                "and regulations. I couldn't find enough relevant support for this question. "
                                "Please ask about a topic covered by the corpus."
                            )
                            status.update(label="No sufficiently relevant evidence found", state="complete")
                            st.markdown(answer)
                            answer_sources = []
                        else:
                            status.update(label="Generating answer…", state="running")
                            prior_history = st.session_state.messages[:-1][-3:]
                            answer = st.write_stream(
                                ollama_chat_stream(
                                    question,
                                    sources,
                                    OLLAMA_BASE_URL,
                                    OLLAMA_MODEL,
                                    history=prior_history,
                                )
                            )
                            status.update(label="Answer ready", state="complete")
                            answer_sources = sources
                    elapsed_seconds = time.perf_counter() - request_started
                    st.caption(f"Completed in {elapsed_seconds:.1f} seconds")
                    st.session_state.messages.append({"role": "assistant", "content": answer, "question": question, "sources": answer_sources, "elapsed_seconds": elapsed_seconds})
                except requests.RequestException as exc:
                    error = f"Ollama request failed ({type(exc).__name__}). Check that Ollama is running and reachable."
                    st.error(error)
                    st.session_state.messages.append({"role": "assistant", "content": error, "question": question, "sources": []})
                except Exception as exc:
                    error = f"I couldn't complete that request: {exc}"
                    st.error(error)
                    st.session_state.messages.append({"role": "assistant", "content": error, "question": question, "sources": []})
        st.rerun()
