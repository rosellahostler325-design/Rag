from langchain_community.document_loaders import PyMuPDFLoader

# 创建一个 PyMuPDFLoader Class 实例，输入为待加载的 pdf 文档路径
loader = PyMuPDFLoader("/home/user/文档/RL4LLM/Rag_data/pumpkin_book.pdf")

# 调用 PyMuPDFLoader Class 的函数 load 对 pdf 文件进行加载
pdf_pages = loader.load()

# 查看详细内容
pdf_page = pdf_pages[1]
print(f"每一个元素的类型：{type(pdf_page)}.", 
    f"该文档的描述性数据：{pdf_page.metadata}", 
    f"查看该文档的内容:\n{pdf_page.page_content}", 
    sep="\n------\n")
