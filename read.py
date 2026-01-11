from langchain_community.document_loaders import PyMuPDFLoader

# 创建一个 PyMuPDFLoader Class 实例，输入为待加载的 pdf 文档路径
loader = PyMuPDFLoader("/home/user/文档/RL4LLM/Rag_data/pumpkin_book.pdf")

# 调用 PyMuPDFLoader Class 的函数 load 对 pdf 文件进行加载
pdf_pages = loader.load()

# 打印摘要信息而不是全部内容
print(f"成功加载 {len(pdf_pages)} 页")
print(f"\n第一页预览:")
print(pdf_pages[0].page_content[:500])  # 只打印前500个字符
print(f"\n... 省略其余 {len(pdf_pages) - 1} 页内容")
print(len(pdf_pages))
#print(f"载入后的变量类型为：{type(pdf_pages)}，",  f"print(f"载入后的变量类型为：{type(pdf_pages)}，",  f"该 PDF 一共包含 {len(pdf_pages)} 页")
#print(f"载入后的变量类型为：{type(pdf_pages)}，",  f"该 PDF 一共包含 {len(pdf_pages)} 页")
#该 PDF 一共包含 {len(pdf_pages)} 页")
#print(f"载入后的变量类型为：{type(pdf_pages)}，",  f"该 PDF 一共包含 {len(pdf_pages)} 页")
