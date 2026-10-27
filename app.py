"""Streamlit UI: upload one document, chat about it with cited sources."""

import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag import SUPPORTED, answer, build_index, chunk, label, load_document

load_dotenv()  # local .env; on Streamlit Cloud, root-level secrets are already env vars

st.set_page_config(page_title="Paper Assistant", page_icon="📄", layout="wide")
# Animated pastel gradient behind the chat; pale tones keep text contrast, motion off for reduced-motion users.
st.html("""<style>
.stApp {
  background: linear-gradient(-45deg, #e0f2fe, #ede9fe, #fce7f3, #dcfce7);
  background-size: 400% 400%;
  animation: bg 18s ease infinite;
}
[data-testid="stHeader"], [data-testid="stBottom"] > div { background: transparent; }
[data-testid="stChatMessage"] { background: rgba(255,255,255,.7); border-radius: .75rem; }
@keyframes bg { 0%,100% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } }
@media (prefers-reduced-motion: reduce) { .stApp { animation: none; } }
</style>""")

ss = st.session_state
ss.setdefault("messages", [])
ss.setdefault("index", None)
ss.setdefault("doc", None)
ss.setdefault("uploader_key", 0)


def show_sources(sources: list[dict]) -> None:
    with st.expander("📚 Sumber"):
        for s in sources:
            st.markdown(f"**{label(s)}** · relevansi {s['score']:.2f}")
            st.text(s["text"][:400] + ("…" if len(s["text"]) > 400 else ""))  # plain: snippet markdown must not render


def tidy(text: str) -> str:
    """Model sometimes cites with 【…】; show the [ … ] labels the prompt asked for."""
    return text.replace("【", "[").replace("】", "]")


with st.sidebar:
    st.title("📄 Paper Assistant")
    # Size limit (20 MB) is enforced by server.maxUploadSize in .streamlit/config.toml
    file = st.file_uploader("Upload dokumen", type=[e.lstrip(".") for e in SUPPORTED],
                            key=f"up{ss.uploader_key}")
    if file and file.file_id != ss.doc:
        with st.spinner("Memproses dokumen..."), tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / Path(file.name).name
            path.write_bytes(file.getbuffer())
            try:
                chunks = chunk(load_document(path))
                ss.index, ss.doc, ss.messages, ss.n_chunks = build_index(chunks), file.file_id, [], len(chunks)
                st.toast(f"{file.name} siap ditanya")
            except ValueError as e:
                ss.index, ss.doc = None, file.file_id
                st.error(str(e))
    if ss.index:
        st.success(f"Dokumen terproses: {ss.n_chunks} potongan")
    if st.button("Reset", width="stretch"):
        ss.messages, ss.index, ss.doc = [], None, None
        ss.uploader_key += 1
        st.rerun()
    st.caption("Jawaban hanya berdasarkan isi dokumen. Jangan upload dokumen rahasia di versi demo.")

for m in ss.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            show_sources(m["sources"])

if question := st.chat_input("Tanya tentang dokumen..."):
    with st.chat_message("user"):
        st.markdown(question)
    if not ss.index:
        st.warning("Upload dokumen dulu di sidebar.")
    else:
        history = [{"role": m["role"], "content": m["content"]} for m in ss.messages]
        ss.messages.append({"role": "user", "content": question})
        with st.chat_message("assistant"):
            stream, sources = answer(question, ss.index, history)
            reply = st.write_stream(tidy(p) for p in stream) or ""
            show_sources(sources)
        ss.messages.append({"role": "assistant", "content": reply, "sources": sources})
