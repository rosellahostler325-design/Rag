# Rag

## Streamlit Cloud 部署（RAG Demo）

在 Streamlit Cloud 的「部署应用程序」页面填：
- **存储库**：`rosellahostler325-design/Rag`
- **分支**：`main`
- **主文件路径**：`streamlit_app.py`

然后在 Streamlit Cloud → App settings → **Secrets** 里添加（推荐 SiliconFlow）：

```toml
SILICONFLOW_API_KEY = "你的key"
```

也支持用 OpenAI 兼容的环境变量名：

```toml
OPENAI_API_KEY = "你的key"
```

注意：
- `streamlit_app.py` 默认会加载仓库内的向量库目录：`vector_db/chroma_qwen`
- 如果你没有把 `vector_db/chroma_qwen` 提交到仓库，需要先在本地运行向量库构建脚本生成它，然后再推送。

