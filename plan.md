# 可执行计划：用 LangChain 核心组件完成 LLM + RAG（保留 OpenAI API → SiliconFlow → Qwen）

参考教程章节（第四章：构建 RAG 应用）：`https://datawhalechina.github.io/llm-universe/#/C4/C4`

## 0. 目标与边界

- **目标**：实现一个可用的 RAG（检索增强生成）问答：向量检索 → 组装上下文 → LLM 生成答案。
- **边界**：去掉多余的 LLM 接入（文心/星火/GLM 等），只保留 **OpenAI API 兼容**这一条路线，并保留通过 **SiliconFlow 调用 Qwen** 的可能性。
- **仓库现状可复用**：
  - `Rag_data/chroma.py`：本地 embedding + Chroma 的构建/检索脚本（结构参考）
  - `Rag_data/chroma_qwen_embedding.py`：SiliconFlow(Qwen Embedding) + Chroma 构建/检索（可直接用）

---

## 1. 环境与密钥（一次性）

### 1.1 准备依赖（按需）

- 文档加载：`langchain-community` + `pymupdf`（`PyMuPDFLoader`）
- 向量库：`chromadb`
- LangChain 核心链组件：`langchain` / `langchain-core`
- 如果做 Web Demo：`streamlit`

### 1.2 配置 SiliconFlow Key（保留 Qwen 的关键）

- `.env`（推荐）：设置 `SILICONFLOW_API_KEY=...`
- 也可设置 `OPENAI_API_KEY`（脚本中已做兜底）

**验收**：运行 `Rag_Embedding/test_api_key.py` 返回 embedding 成功。

---

## 2. 文档准备（PDF → Document）

### 2.1 加载 PDF

- 输入：PDF 路径（例如 `Rag_data/pumpkin_book.pdf`）
- 输出：`List[Document]`（包含 `page_content` + `metadata.page`）

**验收**：随机打印 1-2 页内容，确认可读。

### 2.2 数据清洗（“少做”优先）

原则：清洗是为了“更好检索”，不要破坏段落结构。

- **建议默认关闭**，只在检索很差时才开启清洗
- 可选清洗项：
  - 修复非中文字符之间的断行（教程里用正则处理）
  - 删除 `•`（目录/列表符号噪声）
  - 不建议全删空格（可读性会变差）

**验收**：清洗前后对比 1 页，确认没有把句子粘连到不可读。

---

## 3. 文档切分（Document → chunks）

### 3.1 选择切分器

使用 `RecursiveCharacterTextSplitter`：

- 建议初始参数：
  - `chunk_size = 1000`
  - `chunk_overlap = 200`

### 3.2 生成 chunks

- 输入：`List[Document]`
- 输出：`split_docs = splitter.split_documents(pages)`

**验收**：
- `len(split_docs)` 在合理范围（本项目 pdf 大概 400+ chunks）
- 总字符数统计合理（用于估算 token / 成本）

---

## 4. Embedding（优先走 SiliconFlow-Qwen）

### 4.1 路线 A（推荐）：SiliconFlow（OpenAI 兼容）→ Qwen Embedding

- base_url：`https://api.siliconflow.cn/v1`
- embedding model：`Qwen/Qwen3-Embedding-4B`

**关键规则**：构建向量库用的 embedding = 加载向量库检索时用的 embedding（必须一致）。

### 4.2 路线 B（备用）：本地 HF embedding

仅在无 API 或成本敏感时使用，效果需自行验证。

---

## 5. 构建 / 加载 Chroma 向量库

### 5.1 首次构建（或需要重建）

- `Chroma.from_documents(documents=split_docs, embedding=embedding, persist_directory=...)`
- 若要重建：先删掉 persist 目录再建（避免混库）

### 5.2 加载已有库

- `Chroma(persist_directory=..., embedding_function=embedding)`

**验收**：
- `vectordb._collection.count()` == `len(split_docs)`

---

## 6. 检索（Retriever）

### 6.1 构建 retriever

- `retriever = vectordb.as_retriever(search_kwargs={"k": 3})`
- `docs = retriever.invoke(question)`

### 6.2 检查检索质量（必须做）

打印：
- `doc.metadata.get("page")`
- `doc.page_content[:200]`

**验收问题建议**：
- “南瓜书是什么？”
- “西瓜书是什么？”

> 如果经常检到目录页：优先考虑 **清洗策略/切分策略**，以及过滤“纯目录块”再入库。

---

## 7. RAG 问答链（LCEL 串联核心组件）

### 7.1 组件清单（只用核心）

- Retriever（来自 Chroma）
- PromptTemplate / ChatPromptTemplate
- LLM（OpenAI API 兼容；保留 SiliconFlow-Qwen）
- OutputParser（例如 `StrOutputParser`）
- LCEL 串联（`|`）

### 7.2 实现最小 QA Chain（单轮）

1. `combine_docs(docs) -> "\n\n".join(doc.page_content for doc in docs)`
2. Prompt：必须包含 `{context}` 与 `{input}`，并要求“不知道就说不知道”
3. 组装：
   - `retriever -> combiner` 得到 context
   - `RunnableParallel({"context": retrieval_chain, "input": RunnablePassthrough()}) | prompt | llm | StrOutputParser()`

**验收**：对上述 2 个问题能给出“基于文本”的回答，而不是纯胡编。

---

## 8. 多轮对话（可选但推荐）

### 8.1 传递聊天记录（ChatPromptTemplate）

- 模板结构：
  - system：规则 + `{context}`
  - placeholder：`{chat_history}`
  - human：`{input}`

### 8.2 信息压缩（强烈推荐）

解决“它/他/这个”指代问题：

- 无 history：直接用 input 检索
- 有 history：先让 LLM **改写成完整问题**，再去检索
- 用 `RunnableBranch` 实现分支

**验收对话**：
1. 用户：西瓜书是什么？
2. 用户：你能介绍一下它吗？

应能把“它”还原为“西瓜书/周志华老师的机器学习”并检索到相关内容。

---

## 9. Demo（可选：Streamlit）

### 9.1 最小界面

- `st.session_state.messages` 存聊天历史
- `st.chat_input` 输入
- `st.chat_message` 输出
- 把 RAG 链的输出做成流式（可选）

### 9.2 运行

- `streamlit run streamlit_app.py`

**验收**：能连续对话，且回答来源于检索上下文（而不是纯模型记忆）。

---

## 10. 优化顺序（按优先级）

1. **先优化检索**：切分参数、清洗开关、过滤目录噪声、MMR/多样性检索
2. **再优化回答**：Prompt 收紧、长度限制、引用约束、拒答策略
3. **最后做工程化**：缓存、批 embedding、评估集、日志与可观测性

