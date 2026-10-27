"""RAG pipeline: load document, chunk, embed, retrieve."""

import warnings
from pathlib import Path

import docx2txt
import streamlit as st
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from pypdf import PdfReader

with warnings.catch_warnings():  # langchain-community emits a sunset DeprecationWarning
    warnings.simplefilter("ignore")
    from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

SUPPORTED = (".pdf", ".docx", ".md", ".txt")
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise ValueError("Encoding file tidak didukung, simpan ulang sebagai UTF-8.") from None


def load_document(path: str | Path) -> list[dict]:
    """Return [{'text', 'source', ...location}] per page/section. Raises ValueError on bad input."""
    path = Path(path)
    ext = path.suffix.lower()
    src = path.name
    try:
        if ext == ".pdf":
            docs = [{"text": p.extract_text() or "", "source": src, "page": i}
                    for i, p in enumerate(PdfReader(path).pages, 1)]
        elif ext == ".docx":
            docs = [{"text": docx2txt.process(str(path)), "source": src}]
        elif ext == ".txt":
            docs = [{"text": _read_text(path), "source": src}]
        elif ext == ".md":
            parts = MarkdownHeaderTextSplitter(_HEADERS, strip_headers=False).split_text(_read_text(path))
            docs = [{"text": d.page_content, "source": src,
                     "heading": d.metadata.get("h3") or d.metadata.get("h2") or d.metadata.get("h1") or "(Pembuka)"}
                    for d in parts]
        else:
            raise ValueError(f"Format {ext or '(tanpa ekstensi)'} tidak didukung. Gunakan: {', '.join(SUPPORTED)}.")
    except ValueError:
        raise
    except Exception as e:  # corrupt / unreadable file
        raise ValueError(f"File tidak bisa dibaca (rusak atau bukan {ext}): {e}") from e

    docs = [d for d in docs if d["text"].strip()]
    if not docs:
        raise ValueError("Tidak ada teks yang bisa diekstrak (file kosong atau PDF hasil scan).")
    return docs


def chunk(docs: list[dict], size: int = 1000, overlap: int = 200) -> list[dict]:
    """Split docs into chunks, keeping metadata and adding sequential chunk_id."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    out = []
    for d in docs:
        meta = {k: v for k, v in d.items() if k != "text"}
        for piece in splitter.split_text(d["text"]):
            out.append({**meta, "text": piece, "chunk_id": len(out)})
    return out


@st.cache_resource(show_spinner="Memuat model embedding...")
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL, encode_kwargs={"normalize_embeddings": True})


def build_index(chunks: list[dict]) -> FAISS:
    """Build an in-memory FAISS index."""
    return FAISS.from_texts(
        [c["text"] for c in chunks],
        get_embeddings(),
        metadatas=[{k: v for k, v in c.items() if k != "text"} for c in chunks],
        # Vectors are normalized, FAISS returns squared L2 = 2 - 2*cos, so cos = 1 - d/2 (clipped at 0).
        relevance_score_fn=lambda d: max(0.0, 1.0 - d / 2),
    )


def retrieve(query: str, index: FAISS, k: int = 4) -> list[dict]:
    """Top-k chunks with relevance score in [0, 1] (higher = more relevant)."""
    hits = index.similarity_search_with_relevance_scores(query, k=k)
    return [{**doc.metadata, "text": doc.page_content, "score": round(float(s), 3)}
            for doc, s in hits]
