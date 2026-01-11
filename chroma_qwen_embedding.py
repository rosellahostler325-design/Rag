"""
chroma_qwen_embedding.py - 使用 SiliconFlow API 的 Qwen Embedding 构建向量数据库
基于 LangChain 和 Chroma 实现 PDF 文档的向量化和检索
使用 Qwen/Qwen3-Embedding-4B 模型（通过 SiliconFlow API）
"""

import re
import os
import sys
import argparse
from typing import List, Optional, Tuple
from dataclasses import dataclass

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# 使用 dotenv 加载环境变量
try:
    from dotenv import load_dotenv, find_dotenv
    _ = load_dotenv(find_dotenv())
except ImportError:
    print("警告: python-dotenv 未安装，将使用系统环境变量")

# 使用 OpenAI 兼容的 Embedding（因为 SiliconFlow 兼容 OpenAI API）
try:
    from langchain_community.embeddings import OpenAIEmbeddings
except ImportError:
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError:
        print("请安装 langchain-community 或 langchain-openai")
        raise


# ==================== 配置类 ====================

@dataclass
class Config:
    """配置类，统一管理所有参数"""
    # 文件路径
    pdf_path: str = "/home/user/文档/RL4LLM/Rag_data/pumpkin_book.pdf"
    persist_directory: str = "/home/user/文档/RL4LLM/Rag_data/vector_db/chroma_qwen"
    
    # 文档分割参数
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Embedding 模型配置（SiliconFlow API）
    embedding_model: str = "Qwen/Qwen3-Embedding-4B"
    api_key: Optional[str] = None  # 如果为 None，从环境变量读取
    api_base: str = "https://api.siliconflow.cn/v1"
    
    # 数据清洗选项
    enable_cleaning: bool = False  # 是否启用数据清洗
    remove_bullets: bool = True  # 是否删除项目符号
    remove_spaces: bool = False  # 是否删除空格（不推荐，会影响可读性）
    
    # 检索参数
    default_k: int = 5  # 默认返回结果数量
    mmr_lambda: float = 0.5  # MMR 多样性参数
    
    # 是否重建向量库
    rebuild: bool = False  # True: 强制重建，False: 如果存在则加载


# ==================== 3.3.3 数据清洗 ====================

def clean_pdf_content(pdf_page, config: Config):
    """
    清洗 PDF 文档内容
    
    参数:
        pdf_page: PDF 页面文档对象
        config: 配置对象
    """
    # 使用正则表达式匹配并删除非中文字符之间的换行符
    pattern = re.compile(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]', re.DOTALL)
    pdf_page.page_content = re.sub(
        pattern, 
        lambda match: match.group(0).replace('\n', ''), 
        pdf_page.page_content
    )
    
    # 根据配置决定是否删除项目符号和空格
    if config.remove_bullets:
        pdf_page.page_content = pdf_page.page_content.replace('•', '')
    if config.remove_spaces:
        pdf_page.page_content = pdf_page.page_content.replace(' ', '')
    
    return pdf_page


def clean_markdown_content(md_page):
    """
    清洗 Markdown 文档内容
    - 将双换行符替换为单换行符
    """
    md_page.page_content = md_page.page_content.replace('\n\n', '\n')
    return md_page


# ==================== 3.3.4 文档分割 ====================

def create_text_splitter(chunk_size: int, chunk_overlap: int):
    """创建递归字符文本分割器"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter


# ==================== 3.4.2 构建 Chroma 向量库 ====================

def load_pdf_documents(pdf_path: str):
    """加载 PDF 文档"""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")
    
    print(f"正在加载 PDF 文档: {pdf_path}")
    loader = PyMuPDFLoader(pdf_path)
    pdf_pages = loader.load()
    print(f"成功加载 {len(pdf_pages)} 页")
    return pdf_pages


def clean_documents(documents, config: Config, is_pdf: bool = True):
    """清洗文档列表"""
    if not config.enable_cleaning:
        return documents
    
    cleaned_docs = []
    for doc in documents:
        if is_pdf:
            cleaned_doc = clean_pdf_content(doc, config)
        else:
            cleaned_doc = clean_markdown_content(doc)
        cleaned_docs.append(cleaned_doc)
    return cleaned_docs


def create_vector_database(documents, embedding_model, persist_directory: str, rebuild: bool = False):
    """
    创建 Chroma 向量数据库
    
    参数:
        documents: 分割后的文档列表
        embedding_model: Embedding 模型
        persist_directory: 持久化目录路径
        rebuild: 是否强制重建
    """
    # 如果目录存在且需要重建，先删除旧数据库
    if rebuild and os.path.exists(persist_directory):
        import shutil
        print(f"删除旧的向量数据库: {persist_directory}")
        shutil.rmtree(persist_directory)
    
    # 创建向量数据库
    print("正在创建向量数据库...")
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    
    print(f"向量库中存储的数量：{vectordb._collection.count()}")
    return vectordb


def load_vector_database(persist_directory: str, embedding_model):
    """加载已存在的向量数据库"""
    if not os.path.exists(persist_directory):
        raise FileNotFoundError(f"向量数据库不存在: {persist_directory}")
    
    print(f"正在加载向量数据库: {persist_directory}")
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )
    print(f"向量库中存储的数量：{vectordb._collection.count()}")
    return vectordb


# ==================== 3.4.3 向量检索 ====================

def similarity_search(vectordb, question: str, k: int = 3):
    """
    相似度搜索（带相似度分数）
    
    返回:
        List[Tuple[Document, float]]: (文档, 相似度分数) 的列表
    """
    print(f"\n正在搜索: {question}")
    
    # 使用 similarity_search_with_score 获取相似度分数
    sim_docs_with_scores = vectordb.similarity_search_with_score(question, k=k)
    print(f"检索到的内容数：{len(sim_docs_with_scores)}")
    
    for i, (doc, score) in enumerate(sim_docs_with_scores):
        print(f"\n检索到的第{i+1}个内容 (相似度分数: {score:.4f}, 距离越小越相似):")
        print(f"页面: {doc.metadata.get('page', 'N/A')}")
        print(f"内容: {doc.page_content[:300]}...")
        print("-" * 50)
    
    return sim_docs_with_scores


def mmr_search(vectordb, question: str, k: int = 3, lambda_mult: float = 0.5):
    """
    最大边际相关性搜索 (MMR)
    
    参数:
        lambda_mult: 控制多样性的参数 (0-1)
            - 0: 完全偏向多样性（可能相关性较低）
            - 1: 完全偏向相关性（类似相似度搜索）
            - 0.5: 平衡相关性和多样性（推荐）
    
    返回:
        List[Document]: 文档列表
    """
    print(f"\n正在使用 MMR 搜索: {question} (lambda_mult={lambda_mult})")
    mmr_docs = vectordb.max_marginal_relevance_search(question, k=k, lambda_mult=lambda_mult)
    print(f"MMR 检索到的内容数：{len(mmr_docs)}")
    
    for i, doc in enumerate(mmr_docs):
        print(f"\nMMR 检索到的第{i+1}个内容:")
        print(f"页面: {doc.metadata.get('page', 'N/A')}")
        print(f"内容: {doc.page_content[:300]}...")
        print("-" * 50)
    
    return mmr_docs


# ==================== 初始化 Embedding 模型 ====================

def create_embedding_model(config: Config):
    """
    创建 SiliconFlow Qwen Embedding 模型
    
    使用 OpenAI 兼容的接口，因为 SiliconFlow 兼容 OpenAI API
    """
    print("\n正在初始化 Embedding 模型...")
    
    # 获取 API key
    api_key = config.api_key
    if api_key is None:
        api_key = os.environ.get('SILICONFLOW_API_KEY') or os.environ.get('OPENAI_API_KEY')
    
    if not api_key:
        raise ValueError(
            "API key 未找到。请设置 SILICONFLOW_API_KEY 或 OPENAI_API_KEY 环境变量，"
            "或者在配置中指定 api_key。\n"
            "提示: 可以在项目根目录创建 .env 文件，添加: SILICONFLOW_API_KEY=your_api_key"
        )
    
    print(f"模型: {config.embedding_model}")
    print(f"API Base: {config.api_base}")
    print(f"API Key: {api_key[:10]}...{api_key[-10:] if len(api_key) > 20 else ''}")
    
    # 使用 OpenAIEmbeddings，但设置自定义的 base_url
    # OpenAIEmbeddings 支持 openai_api_base 参数来指定自定义 API 端点
    embedding = OpenAIEmbeddings(
        model=config.embedding_model,
        openai_api_key=api_key,
        openai_api_base=config.api_base,
    )
    
    print(f"Embedding 模型已加载: {config.embedding_model}")
    print("提示: 这是通过 SiliconFlow API 调用的 Qwen Embedding 模型")
    
    return embedding


# ==================== 主程序 ====================

def build_vector_database(config: Config):
    """构建向量数据库"""
    # ========== 1. 加载文档 ==========
    pdf_pages = load_pdf_documents(config.pdf_path)
    
    # ========== 2. 数据清洗（可选） ==========
    if config.enable_cleaning:
        print("\n正在清洗文档...")
        pdf_pages = clean_documents(pdf_pages, config, is_pdf=True)
    
    # ========== 3. 文档分割 ==========
    text_splitter = create_text_splitter(config.chunk_size, config.chunk_overlap)
    split_docs = text_splitter.split_documents(pdf_pages)
    
    print(f"\n切分后的文件数量：{len(split_docs)}")
    print(f"切分后的字符数（可以用来大致评估 token 数）：{sum([len(doc.page_content) for doc in split_docs])}")
    
    # ========== 4. 初始化 Embedding 模型 ==========
    embedding = create_embedding_model(config)
    
    # ========== 5. 构建向量数据库 ==========
    vectordb = create_vector_database(split_docs, embedding, config.persist_directory, rebuild=config.rebuild)
    
    return vectordb, embedding


def interactive_search(vectordb, config: Config):
    """交互式搜索"""
    print("\n" + "="*50)
    print("进入交互式搜索模式（输入 'quit' 或 'exit' 退出）")
    print("="*50)
    
    while True:
        try:
            question = input("\n请输入搜索问题: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("退出交互式搜索模式")
                break
            
            if not question:
                continue
            
            # 询问搜索方式
            search_type = input("选择搜索方式 [1: 相似度搜索, 2: MMR搜索, 默认: 1]: ").strip()
            k = input(f"返回结果数量 [默认: {config.default_k}]: ").strip()
            k = int(k) if k.isdigit() else config.default_k
            
            if search_type == '2':
                mmr_search(vectordb, question, k=k, lambda_mult=config.mmr_lambda)
            else:
                similarity_search(vectordb, question, k=k)
                
        except KeyboardInterrupt:
            print("\n\n退出交互式搜索模式")
            break
        except Exception as e:
            print(f"搜索出错: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='使用 SiliconFlow API 的 Qwen Embedding 构建和检索 Chroma 向量数据库'
    )
    parser.add_argument('--pdf', type=str, help='PDF 文件路径')
    parser.add_argument('--rebuild', action='store_true', help='强制重建向量数据库')
    parser.add_argument('--interactive', action='store_true', help='交互式搜索模式')
    parser.add_argument('--question', type=str, help='搜索问题（非交互模式）')
    parser.add_argument('--k', type=int, default=5, help='返回结果数量')
    parser.add_argument('--api-key', type=str, help='SiliconFlow API Key（可选，也可通过环境变量设置）')
    parser.add_argument('--model', type=str, help='Embedding 模型名称（默认: Qwen/Qwen3-Embedding-4B）')
    parser.add_argument('--chunk-size', type=int, help='文档分割块大小')
    parser.add_argument('--chunk-overlap', type=int, help='文档分割重叠大小')
    
    args = parser.parse_args()
    
    # 创建配置
    config = Config()
    
    # 从命令行参数更新配置
    if args.pdf:
        config.pdf_path = args.pdf
    if args.api_key:
        config.api_key = args.api_key
    if args.model:
        config.embedding_model = args.model
    if args.chunk_size:
        config.chunk_size = args.chunk_size
    if args.chunk_overlap:
        config.chunk_overlap = args.chunk_overlap
    config.rebuild = args.rebuild
    config.default_k = args.k
    
    try:
        # 初始化 Embedding 模型
        embedding = create_embedding_model(config)
        
        # 检查向量数据库是否存在
        vectordb_exists = os.path.exists(config.persist_directory)
        
        if vectordb_exists and not config.rebuild:
            # 加载已有向量数据库
            print("\n检测到已有向量数据库，正在加载...")
            vectordb = load_vector_database(config.persist_directory, embedding)
        else:
            # 构建新的向量数据库
            print("\n正在构建向量数据库...")
            vectordb, embedding = build_vector_database(config)
        
        # 搜索模式
        if args.interactive:
            # 交互式搜索
            interactive_search(vectordb, config)
        elif args.question:
            # 命令行搜索
            print("\n" + "="*50)
            print("向量检索")
            print("="*50)
            similarity_search(vectordb, args.question, k=config.default_k)
        else:
            # 默认测试搜索
            print("\n" + "="*50)
            print("开始向量检索测试")
            print("="*50)
            
            question = "什么是机器学习"
            print("\n【相似度搜索 - 按相关性排序】")
            similarity_search(vectordb, question, k=config.default_k)
            
            print("\n【MMR 搜索 - 平衡相关性和多样性】")
            mmr_search(vectordb, question, k=config.default_k, lambda_mult=config.mmr_lambda)
        
        print("\n" + "="*50)
        print("完成！")
        print("="*50)
        print("\n提示:")
        print("1. 使用 --interactive 参数进入交互式搜索模式")
        print("2. 使用 --rebuild 参数强制重建向量数据库")
        print("3. 使用 --api-key 参数指定 API Key，或通过环境变量 SILICONFLOW_API_KEY 设置")
        print("4. 可以调整 chunk_size 和 chunk_overlap 参数优化分割效果")
        print("5. MMR 的 lambda_mult 参数可以调整多样性和相关性的平衡")
        
    except ValueError as e:
        print(f"配置错误: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
