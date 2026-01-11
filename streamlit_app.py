from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import streamlit as st

# Reuse your improved RAG components (retrieval quality upgrades)
import rag_chain_qwen_improve_quality_of_rag as rag


REPO_DIR = Path(__file__).resolve().parent
DEFAULT_PERSIST_DIR = str(REPO_DIR / "vector_db" / "chroma_qwen")

# Local dev convenience: load `.env` if present (Streamlit Cloud won't read your repo `.env`)
try:
    from dotenv import load_dotenv

    load_dotenv(REPO_DIR / ".env", override=False)
except Exception:
    pass


def _get_secret(name: str) -> Optional[str]:
    # Streamlit Cloud secrets first, then env vars
    try:
        # Accessing st.secrets may raise if no secrets.toml exists (common locally)
        if hasattr(st, "secrets"):
            val = st.secrets.get(name)
            if val is not None:
                return str(val)
    except Exception:
        pass
    return os.environ.get(name)


def _build_cfg(
    *,
    base_url: str,
    embedding_model: str,
    chat_model: str,
    persist_directory: str,
    api_key_override: Optional[str],
    k: int,
    fetch_k: int,
    use_mmr: bool,
    mmr_lambda: float,
    enable_filter: bool,
    rewrite_query: bool,
    temperature: float,
) -> rag.Config:
    api_key = (api_key_override or "").strip() or _get_secret("SILICONFLOW_API_KEY") or _get_secret("OPENAI_API_KEY")
    return rag.Config(
        base_url=base_url,
        api_key=api_key,
        embedding_model=embedding_model,
        chat_model=chat_model,
        persist_directory=persist_directory,
        k=k,
        fetch_k=fetch_k,
        use_mmr=use_mmr,
        mmr_lambda=mmr_lambda,
        enable_filter=enable_filter,
        temperature=temperature,
        rewrite_query=rewrite_query,
    )


@st.cache_resource(show_spinner=False)
def _load_rag_resources(cfg: rag.Config):
    embedding = rag.build_embedding(cfg)
    llm = rag.build_llm(cfg)
    vectordb = rag.load_vectordb(cfg, embedding)
    retrieval_chain = rag.build_retrieval_chain(cfg, llm, vectordb)
    qa_chain = rag.build_qa_chain(cfg, llm, retrieval_chain)
    return llm, vectordb, retrieval_chain, qa_chain


def main() -> None:
    st.set_page_config(page_title="RAG Demo", layout="wide")
    st.title("RAG Demo (Chroma + SiliconFlow/Qwen Embedding + Chat)")

    with st.sidebar:
        st.header("Settings")

        api_key_override = st.text_input(
            "SILICONFLOW_API_KEY (optional, for this session)",
            value="",
            type="password",
            help="If set, this overrides Secrets/env for the current session only. On Streamlit Cloud, prefer using Secrets.",
        )

        persist_directory = st.text_input("Chroma persist directory", value=DEFAULT_PERSIST_DIR)

        base_url = st.text_input("OpenAI-compatible base_url", value="https://api.siliconflow.cn/v1")
        embedding_model = st.text_input("Embedding model", value="Qwen/Qwen3-Embedding-4B")
        chat_model = st.text_input("Chat model", value="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
        temperature = st.slider("temperature", 0.0, 1.0, 0.0, 0.05)

        st.divider()
        k = st.slider("k (final docs)", 1, 10, 3, 1)
        fetch_k = st.slider("fetch_k (candidates)", 5, 100, 20, 1)
        use_mmr = st.checkbox("Use MMR", value=False)
        mmr_lambda = st.slider("MMR lambda", 0.0, 1.0, 0.5, 0.05)
        enable_filter = st.checkbox("Enable TOC/noise filter", value=True)
        rewrite_query = st.checkbox("Rewrite query (LLM)", value=False)

        st.divider()
        st.caption(
            "Secrets: set SILICONFLOW_API_KEY (recommended) or OPENAI_API_KEY in Streamlit Cloud → Settings → Secrets."
        )

    q = st.text_input("Question", value="西瓜书是什么？")
    col1, col2 = st.columns([1, 1])
    with col1:
        run = st.button("Ask", type="primary", use_container_width=True)
    with col2:
        show_context = st.checkbox("Show retrieved context", value=False)

    if run:
        cfg = _build_cfg(
            base_url=base_url,
            embedding_model=embedding_model,
            chat_model=chat_model,
            persist_directory=persist_directory,
            api_key_override=api_key_override,
            k=k,
            fetch_k=fetch_k,
            use_mmr=use_mmr,
            mmr_lambda=mmr_lambda,
            enable_filter=enable_filter,
            rewrite_query=rewrite_query,
            temperature=temperature,
        )

        if not cfg.api_key:
            st.error("Missing API key. Please set SILICONFLOW_API_KEY (recommended) or OPENAI_API_KEY in Secrets.")
            st.stop()

        if not Path(persist_directory).exists():
            st.error(
                f"Vector DB not found: `{persist_directory}`. "
                "Make sure you committed `vector_db/chroma_qwen` to the repo, or update the path in the sidebar."
            )
            st.stop()

        with st.spinner("Loading vector DB / models..."):
            _llm, _vectordb, retrieval_chain, qa_chain = _load_rag_resources(cfg)

        with st.spinner("Generating answer..."):
            answer = qa_chain.invoke(q)

        st.subheader("Answer")
        st.write(answer)

        if show_context:
            st.subheader("Retrieved context (preview)")
            ctx = retrieval_chain.invoke(q)
            st.code(ctx[:4000])


if __name__ == "__main__":
    main()

