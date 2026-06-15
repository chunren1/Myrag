# 🧠 高品質 RAG 知識庫系统

基于 **Agentic RAG** 架构的零成本知识库问答系统，使用硅基流动免费 API，本地轻量部署。

## 架构概览

```
用户提问
  │
  ▼
┌─────────────────────────────────────────────┐
│  Agentic RAG 闭环 (backend/core/)            │
│                                              │
│  Planner        →  拆解为 2~3 个子问题       │
│     ↓                                        │
│  Retriever      →  粗排(Qdrant) + 精排(Rerank)│
│     ↓                                        │
│  Reflector      →  评估充分性 → 补充检索(≤1轮)│
│     ↓                                        │
│  Generator      →  <thinking>打草稿 + 引用标注 │
└─────────────────────────────────────────────┘
  │
  ▼
SSE 流式输出 (过滤 thinking 标签)
```

## 技术栈

| 层级 | 技术 |
|------|------|
| **API 提供商** | 硅基流动 (SiliconFlow) — 完全免费 |
| **Embedding** | `BAAI/bge-m3` (1024维) |
| **Rerank** | `BAAI/bge-reranker-v2-m3` |
| **Planner/Reflector** | `Qwen/Qwen2.5-72B-Instruct` |
| **Generator** | `deepseek-ai/DeepSeek-V2.5` |
| **向量数据库** | Qdrant (INT8 量化，内存仅需 ~25%) |
| **后端框架** | FastAPI + SSE 流式 |
| **限流重试** | tenacity 指数退避 + 随机抖动 |

## 快速开始

### 1. 环境要求

- Python 3.10+
- Docker Desktop

### 2. 安装

```powershell
# 克隆项目
cd d:\high-quality-rag

# 创建虚拟环境
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 安装依赖
pip install -r requirements.txt
```

### 3. 配置

编辑 `.env` 文件，填入硅基流动 API Key：

```
SILICONFLOW_API_KEY=sk-xxxxxxxxxxxxxxxx
```

> 获取地址: [https://cloud.siliconflow.cn/account/ak](https://cloud.siliconflow.cn/account/ak)

### 4. 启动服务

```powershell
# 启动 Qdrant 向量数据库
docker compose up -d

# 启动后端 API (开发模式 + 热重载)
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

访问 [http://localhost:8000/docs](http://localhost:8000/docs) 查看 API 文档。

### 5. 导入知识库

```powershell
# 将 Markdown 文件放入 workspace/raw_docs/
# 然后运行数据管道
python -m data_pipeline.main --dir ./workspace/raw_docs
```

### 6. 对话

```powershell
# 非流式
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "什么是RAG？", "stream": false}'

# 流式 (SSE)
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Transformer有哪些优势？", "stream": true}'
```

## 项目结构

```
high-quality-rag/
├── .env                          # 环境变量 (API Key)
├── .wslconfig                    # WSL2 内存限制
├── docker-compose.yml            # Qdrant 编排
├── requirements.txt              # Python 依赖
│
├── data_pipeline/                # 离线数据管道
│   ├── main.py                   # 管道入口
│   ├── markdown_splitter.py      # 标题层级切分
│   ├── llm_enrichment.py         # Doc2Query 增强
│   ├── embedder.py               # bge-m3 向量化
│   └── qdrant_uploader.py        # 入库 Qdrant
│
├── backend/                      # 在线推理服务
│   ├── main.py                   # FastAPI 入口
│   ├── api/chat.py               # SSE 流式 + thinking 过滤
│   ├── clients/
│   │   ├── siliconflow.py        # API 封装 (429 重试)
│   │   └── qdrant_client.py      # Qdrant (INT8 量化)
│   └── core/
│       ├── agent_flow.py         # Agent 总编排
│       ├── query_decomposer.py   # 问题拆解
│       ├── retriever.py          # 粗排 + 精排
│       ├── context_reflector.py  # 反思评估
│       └── prompt_builder.py     # Prompt 工程
│
├── dify/README.md                # Dify 集成说明
├── workspace/
│   ├── raw_docs/                 # 放原始 Markdown
│   └── logs/                     # 日志输出
└── qdrant_storage/               # Qdrant 数据持久化
```

## API 接口

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 系统信息 |
| `/api/health` | GET | 健康检查 |
| `/api/chat` | POST | 对话（支持 SSE 流式） |
| `/docs` | GET | Swagger 文档 |

### Chat 请求

```json
{
  "query": "你的问题",
  "stream": true,
  "mode": "agentic"
}
```

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `query` | string | 必填 | 用户问题 (1~5000字) |
| `stream` | bool | true | 流式输出 |
| `mode` | string | agentic | agentic(完整) / simple(简洁) |

### SSE 事件类型

| type | 说明 |
|------|------|
| `text` | 正文内容（已过滤 thinking） |
| `done` | 流结束 |
| `error` | 错误信息 |

## 核心特性

- **零成本**：全部使用硅基流动免费 API
- **Agentic RAG**：规划 → 检索 → 反思 → 生成 四阶段闭环
- **智能检索**：Doc2Query 增强 + 粗排 + bge-reranker 精排
- **流式输出**：实时 SSE 推送，过滤草稿内容
- **引用标注**：`[Doc_ID: xxx]` 格式追溯到源文档
- **防限流**：tenacity 指数退避 + 随机抖动重试
- **内存优化**：INT8 标量量化，节省 75% 内存
- **启动容错**：Qdrant 不可用时降级运行，不阻塞服务

## 常见问题

**Q: 提示 `ModuleNotFoundError`？**
A: 确保已激活虚拟环境 `.\\.venv\\Scripts\\Activate.ps1`

**Q: Qdrant 连接失败？**
A: 运行 `docker compose up -d` 启动 Qdrant

**Q: API 返回 429？**
A: 系统已内置自动重试，等待即可。频繁触发可能是免费 API 配额问题

**Q: 如何添加文档？**
A: 将 `.md` 文件放入 `workspace/raw_docs/`，运行 `python -m data_pipeline.main`
