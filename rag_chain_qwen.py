"""
rag_chain_qwen.py

按 DataWhale《动手学大模型应用开发》C4.2（构建检索问答链）实现一个最小可用 RAG：
1) 加载 Chroma 向量库（用 SiliconFlow 的 Qwen Embedding 构建/加载）
2) 构建 retriever
3) 构建 retrieval_chain（retriever -> combine_docs）
4) 构建 qa_chain（RunnableParallel -> PromptTemplate -> ChatOpenAI -> StrOutputParser）

参考页面：
https://datawhalechina.github.io/llm-universe/#/C4/C4?id=_42-%e6%9e%84%e5%bb%ba%e6%a3%80%e7%b4%a2%e9%97%ae%e7%ad%94%e9%93%be
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re

from langchain_community.document_loaders import PyMuPDFLoader  # noqa: F401 (kept as reference / optional)
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

try:
    from dotenv import find_dotenv, load_dotenv

    _ = load_dotenv(find_dotenv())
except Exception:
    # dotenv 可选：没有也能用环境变量
    pass


@dataclass
class Config:
    # SiliconFlow(OpenAI compatible) settings
    base_url: str = "https://api.siliconflow.cn/v1"
    api_key: str | None = None

    # Models
    embedding_model: str = "Qwen/Qwen3-Embedding-4B"
    # NOTE: chat model 由用户指定（SiliconFlow/OpenAI-compatible）
    # Default set to a known-working model in this workspace:
    chat_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

    # Vector DB (repo-relative by default; works on Streamlit Cloud)
    persist_directory: str = str(Path(__file__).resolve().parent / "vector_db" / "chroma_qwen")

    # Retrieval
    k: int = 3

    # LLM params
    temperature: float = 0.0


def _get_api_key(cfg: Config) -> str:
    if cfg.api_key:
        return cfg.api_key
    key = os.environ.get("SILICONFLOW_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError(
            "Missing API key. Set SILICONFLOW_API_KEY (recommended) or OPENAI_API_KEY, "
            "or pass --api-key."
        )
    return key


def load_vectordb(cfg: Config) -> Chroma:
    api_key = _get_api_key(cfg)
    embedding = OpenAIEmbeddings(
        model=cfg.embedding_model,
        openai_api_key=api_key,
        openai_api_base=cfg.base_url,
    )
    vectordb = Chroma(
        persist_directory=cfg.persist_directory,
        embedding_function=embedding,
    )
    return vectordb


def build_retrieval_chain(vectordb: Chroma, k: int):
    retriever = vectordb.as_retriever(search_kwargs={"k": k})

    def combine_docs(docs) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    combiner = RunnableLambda(combine_docs)
    retrieval_chain = retriever | combiner
    return retriever, retrieval_chain


def build_qa_chain(cfg: Config, retrieval_chain):
    api_key = _get_api_key(cfg)
    llm = ChatOpenAI(
        model_name=cfg.chat_model,
        temperature=cfg.temperature,
        openai_api_key=api_key,
        openai_api_base=cfg.base_url,
    )

    template = """使用以下上下文来回答最后的问题。
如果你不知道答案，就说你不知道，不要试图编造答案。
最多使用三句话。尽量使答案简明扼要。
请你在回答的最后说“谢谢你的提问！”。
请只输出最终答案，不要输出思考过程，不要输出任何 <think>...</think> 或类似标记。

{context}

问题: {input}
"""

    prompt = PromptTemplate(template=template, input_variables=["context", "input"])

    def strip_think(text: str) -> str:
        # Remove <think>...</think> blocks (and anything before the closing tag if model prepends it)
        if "</think>" in text:
            text = re.sub(r"(?s)^.*?</think>\s*", "", text)
        text = re.sub(r"(?s)<think>.*?</think>\s*", "", text).strip()
        return text

    qa_chain = (
        RunnableParallel({"context": retrieval_chain, "input": RunnablePassthrough()})
        | prompt
        | llm
        | StrOutputParser()
        | RunnableLambda(strip_think)
    )
    return qa_chain


def preview_retrieval(retriever, question: str):
    docs = retriever.invoke(question)
    print(f"检索到的内容数：{len(docs)}")
    for i, doc in enumerate(docs):
        page = doc.metadata.get("page", "N/A")
        print(f"\n--- doc[{i}] page={page} ---")
        print(doc.page_content[:300])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a minimal RAG QA chain using SiliconFlow-Qwen + Chroma.")
    parser.add_argument("--question", type=str, required=True, help="用户问题")
    parser.add_argument("--k", type=int, default=3, help="检索返回条数")
    parser.add_argument("--persist-directory", type=str, default=Config.persist_directory, help="Chroma 持久化目录")
    parser.add_argument("--embedding-model", type=str, default=Config.embedding_model, help="Embedding 模型名")
    parser.add_argument("--chat-model", type=str, default=Config.chat_model, help="Chat 模型名")
    parser.add_argument("--base-url", type=str, default=Config.base_url, help="OpenAI 兼容 base_url（SiliconFlow）")
    parser.add_argument("--api-key", type=str, default=None, help="API Key（可选；否则读环境变量）")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature")
    parser.add_argument("--print-retrieved", action="store_true", help="先打印检索到的原始片段（调试用）")

    args = parser.parse_args(argv)

    cfg = Config(
        base_url=args.base_url,
        api_key=args.api_key,
        embedding_model=args.embedding_model,
        chat_model=args.chat_model,
        persist_directory=args.persist_directory,
        k=args.k,
        temperature=args.temperature,
    )

    vectordb = load_vectordb(cfg)
    print(f"向量库中存储的数量：{vectordb._collection.count()}")

    retriever, retrieval_chain = build_retrieval_chain(vectordb, k=cfg.k)

    if args.print_retrieved:
        print("\n========== 检索预览 ==========")
        preview_retrieval(retriever, args.question)
        print("\n==============================\n")

    qa_chain = build_qa_chain(cfg, retrieval_chain)

    print("========== RAG Answer ==========")
    answer = qa_chain.invoke(args.question)
    print(answer)
    print("================================")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

