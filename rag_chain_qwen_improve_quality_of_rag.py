"""
rag_chain_qwen_improve_quality_of_rag.py

目标：在 rag_chain_qwen.py 的基础上，重点提升 RAG 的“检索质量”，并提供交互式测试入口。

改进点（可开关）：
1) 更大候选召回：fetch_k（先取更多候选，再过滤/重排）
2) 启发式过滤：过滤目录页/噪声块（大量点线/数字比例过高/包含“目录”等）
3) 可选 MMR：在候选中增加多样性（lambda_mult）
4) 可选 Query Rewrite：先用 LLM 把问题改写得更“像书里说法”，再检索（对“它/这本书”这类指代更有用）
5) 输出清理：去掉 <think>...</think>，只保留最终答案

默认配置（按你说明）：
- chat model: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
- embedding model: Qwen/Qwen3-Embedding-4B

参考教程（4.2 构建检索问答链）：
https://datawhalechina.github.io/llm-universe/#/C4/C4?id=_42-%e6%9e%84%e5%bb%ba%e6%a3%80%e7%b4%a2%e9%97%ae%e7%ad%94%e9%93%be
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

try:
    from dotenv import find_dotenv, load_dotenv

    _ = load_dotenv(find_dotenv())
except Exception:
    pass


@dataclass
class Config:
    # SiliconFlow (OpenAI compatible)
    base_url: str = "https://api.siliconflow.cn/v1"
    api_key: str | None = None

    # Models
    embedding_model: str = "Qwen/Qwen3-Embedding-4B"
    chat_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

    # Vector DB (repo-relative by default; works on Streamlit Cloud)
    persist_directory: str = str(Path(__file__).resolve().parent / "vector_db" / "chroma_qwen")

    # Retrieval
    k: int = 3
    fetch_k: int = 20
    use_mmr: bool = False
    mmr_lambda: float = 0.5
    enable_filter: bool = True

    # LLM params
    temperature: float = 0.0

    # Query rewrite
    rewrite_query: bool = False


def _get_api_key(cfg: Config) -> str:
    if cfg.api_key:
        return cfg.api_key
    key = os.environ.get("SILICONFLOW_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError(
            "Missing API key. Set SILICONFLOW_API_KEY (recommended) or OPENAI_API_KEY, or pass --api-key."
        )
    return key


def build_embedding(cfg: Config) -> OpenAIEmbeddings:
    api_key = _get_api_key(cfg)
    return OpenAIEmbeddings(model=cfg.embedding_model, openai_api_key=api_key, openai_api_base=cfg.base_url)


def build_llm(cfg: Config) -> ChatOpenAI:
    api_key = _get_api_key(cfg)
    return ChatOpenAI(
        model_name=cfg.chat_model,
        temperature=cfg.temperature,
        openai_api_key=api_key,
        openai_api_base=cfg.base_url,
    )


def load_vectordb(cfg: Config, embedding: OpenAIEmbeddings) -> Chroma:
    vectordb = Chroma(persist_directory=cfg.persist_directory, embedding_function=embedding)
    return vectordb


def _looks_like_toc_or_noise(text: str) -> bool:
    """启发式判断：目录/噪声块过滤（尽量保守，避免误杀）。"""
    t = text.strip()
    if not t:
        return True

    # 强信号：包含“目录”
    if "目录" in t[:80]:
        return True

    # 强信号：大量点线（目录常见）
    dot_ratio = t.count(".") / max(1, len(t))
    if dot_ratio > 0.08:
        return True

    # 数字比例过高（目录/公式页码索引常见）
    digit_ratio = sum(ch.isdigit() for ch in t) / max(1, len(t))
    if digit_ratio > 0.25 and len(t) < 1500:
        return True

    # 过短且像页码索引
    if len(t) < 60 and any(ch.isdigit() for ch in t):
        return True

    return False


def filter_docs(docs_with_scores: List[Tuple[object, float]], enable_filter: bool) -> List[Tuple[object, float]]:
    if not enable_filter:
        return docs_with_scores
    out: List[Tuple[object, float]] = []
    for doc, score in docs_with_scores:
        content = getattr(doc, "page_content", "") or ""
        if _looks_like_toc_or_noise(content):
            continue
        out.append((doc, score))
    return out


def combine_docs(docs: Iterable[object]) -> str:
    return "\n\n".join(getattr(d, "page_content", "") for d in docs)


def strip_think(text: str) -> str:
    # Remove <think>...</think> blocks (and anything before the closing tag if model prepends it)
    if "</think>" in text:
        text = re.sub(r"(?s)^.*?</think>\s*", "", text)
    text = re.sub(r"(?s)<think>.*?</think>\s*", "", text).strip()
    return text


def maybe_rewrite_query(llm: ChatOpenAI, query: str, enabled: bool) -> str:
    if not enabled:
        return query
    prompt = PromptTemplate(
        template=(
            "你是检索增强问答系统的查询改写器。\n"
            "请把用户问题改写成更完整、更具体、便于从一本机器学习教材/讲义中检索的查询。\n"
            "要求：\n"
            "- 保持原意\n"
            "- 不要编造事实\n"
            "- 只输出改写后的查询（不要解释）\n\n"
            "用户问题：{q}\n"
        ),
        input_variables=["q"],
    )
    rewritten = (prompt | llm | StrOutputParser() | RunnableLambda(strip_think)).invoke({"q": query})
    return rewritten.strip() or query


def retrieve_docs(
    *,
    vectordb: Chroma,
    query: str,
    k: int,
    fetch_k: int,
    use_mmr: bool,
    mmr_lambda: float,
    enable_filter: bool,
) -> List[object]:
    """
    先取候选（fetch_k），过滤目录噪声，再取前 k。
    - similarity: similarity_search_with_score
    - mmr: max_marginal_relevance_search（不带 score）
    """
    if use_mmr:
        # MMR 本身就做“相关性+多样性”，但仍可能命中目录噪声，所以也做过滤
        candidates = vectordb.max_marginal_relevance_search(query, k=fetch_k, lambda_mult=mmr_lambda)
        if enable_filter:
            candidates = [d for d in candidates if not _looks_like_toc_or_noise(d.page_content or "")]
        return candidates[:k]

    candidates_with_scores = vectordb.similarity_search_with_score(query, k=fetch_k)
    filtered = filter_docs(candidates_with_scores, enable_filter=enable_filter)
    # 距离越小越相似，保持原排序
    docs = [doc for doc, _score in filtered][:k]
    return docs


def build_qa_chain(cfg: Config, llm: ChatOpenAI, retrieval_chain):
    template = """使用以下上下文来回答最后的问题。
如果你不知道答案，就说你不知道，不要试图编造答案。
最多使用三句话。尽量使答案简明扼要。
请你在回答的最后说“谢谢你的提问！”。
请只输出最终答案，不要输出思考过程，不要输出任何 <think>...</think> 或类似标记。

{context}

问题: {input}
"""
    prompt = PromptTemplate(template=template, input_variables=["context", "input"])

    qa_chain = (
        RunnableParallel({"context": retrieval_chain, "input": RunnablePassthrough()})
        | prompt
        | llm
        | StrOutputParser()
        | RunnableLambda(strip_think)
    )
    return qa_chain


def build_retrieval_chain(cfg: Config, llm: ChatOpenAI, vectordb: Chroma):
    def retrieval_run(user_query: str) -> str:
        q = maybe_rewrite_query(llm, user_query, enabled=cfg.rewrite_query)
        docs = retrieve_docs(
            vectordb=vectordb,
            query=q,
            k=cfg.k,
            fetch_k=cfg.fetch_k,
            use_mmr=cfg.use_mmr,
            mmr_lambda=cfg.mmr_lambda,
            enable_filter=cfg.enable_filter,
        )
        return combine_docs(docs)

    return RunnableLambda(retrieval_run)


def run_once(cfg: Config, question: str, show_context: bool) -> None:
    embedding = build_embedding(cfg)
    llm = build_llm(cfg)
    vectordb = load_vectordb(cfg, embedding)

    print(f"向量库中存储的数量：{vectordb._collection.count()}")

    retrieval_chain = build_retrieval_chain(cfg, llm, vectordb)
    qa_chain = build_qa_chain(cfg, llm, retrieval_chain)

    if show_context:
        ctx = retrieval_chain.invoke(question)
        print("\n========== Retrieved Context (preview) ==========")
        print(ctx[:1500])
        print("===============================================\n")

    print("========== RAG Answer ==========")
    print(qa_chain.invoke(question))
    print("================================")


def interactive_loop(cfg: Config) -> None:
    print("\n进入交互模式（输入 exit/quit 退出）")
    print(
        f"当前检索参数：k={cfg.k}, fetch_k={cfg.fetch_k}, use_mmr={cfg.use_mmr}, "
        f"lambda={cfg.mmr_lambda}, filter={cfg.enable_filter}, rewrite={cfg.rewrite_query}"
    )
    while True:
        try:
            q = input("\nQuestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return

        if not q:
            continue
        if q.lower() in {"exit", "quit", "q"}:
            print("bye")
            return

        run_once(cfg, q, show_context=False)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Improved RAG chain (SiliconFlow-Qwen embedding + DeepSeek chat).")
    parser.add_argument("--question", type=str, default=None, help="单次问题（不传则默认跑：西瓜书是什么？）")
    parser.add_argument("--interactive", action="store_true", help="跑完一次后进入交互模式供你测试")
    parser.add_argument("--show-context", action="store_true", help="打印检索到的上下文预览（调试用）")

    parser.add_argument("--persist-directory", type=str, default=Config.persist_directory)
    parser.add_argument("--base-url", type=str, default=Config.base_url)
    parser.add_argument("--api-key", type=str, default=None)

    parser.add_argument("--embedding-model", type=str, default=Config.embedding_model)
    parser.add_argument("--chat-model", type=str, default=Config.chat_model)
    parser.add_argument("--temperature", type=float, default=0.0)

    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--fetch-k", type=int, default=20)
    parser.add_argument("--use-mmr", action="store_true")
    parser.add_argument("--mmr-lambda", type=float, default=0.5)
    parser.add_argument("--disable-filter", action="store_true", help="关闭目录/噪声过滤")
    parser.add_argument("--rewrite-query", action="store_true", help="开启 query 改写（更贵，但对指代问题更稳）")

    args = parser.parse_args(argv)

    cfg = Config(
        base_url=args.base_url,
        api_key=args.api_key,
        embedding_model=args.embedding_model,
        chat_model=args.chat_model,
        persist_directory=args.persist_directory,
        k=args.k,
        fetch_k=args.fetch_k,
        use_mmr=args.use_mmr,
        mmr_lambda=args.mmr_lambda,
        enable_filter=not args.disable_filter,
        temperature=args.temperature,
        rewrite_query=args.rewrite_query,
    )

    # 按用户要求：先问“西瓜书是什么？”
    question = args.question or "西瓜书是什么？"
    run_once(cfg, question, show_context=args.show_context)

    # 然后进入交互模式，让你自己测试
    if args.interactive:
        interactive_loop(cfg)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

