# 部署指南（本地 / Docker / Hugging Face Spaces / 云平台）

本项目 Web 工作台（FastAPI + SSE + SQLite + Hybrid RAG）支持从零 API 费用的 `demo` 模式无缝切换到真实 DeepSeek `live` 模式，且容器镜像默认以非 root 用户运行并内置 `/api/health` 健康检查。

---

## 1. 本地运行（Windows PowerShell / Linux / macOS）

```powershell
# 1) 安装依赖（含 Hybrid RAG 与 Web 服务）
python -m pip install -r requirements-rag.txt -r requirements-web.txt

# 2) 构建本地向量知识库（首次会下载 BAAI/bge-small-zh-v1.5，无聊天 API 调用）
python -m customer_service.knowledge_cli build

# 3) 方式 A：零成本体验 UI、SQLite 读写与 Hybrid RAG
python web_server.py --mode demo --port 8000

# 4) 方式 B：接入真实 DeepSeek
$env:DEEPSEEK_API_KEY = "sk-你的密钥"
python web_server.py --mode live --port 8000
```

浏览器访问 `http://127.0.0.1:8000`。

---

## 2. Docker / Docker Compose 一键部署（VPS / 阿里云 ECS / 本地容器）

```bash
cp .env.example .env
# 在 .env 中填入 DEEPSEEK_API_KEY（留空则自动以零成本 demo 模式启动）
docker compose up -d --build
```

- 默认映射宿主机 `8000` 端口 → 容器 `7860` 端口，访问 `http://<服务器IP>:8000`。
- 若服务器内存较小且暂不需要本地 Embedding，可关闭 RAG 镜像层：
  ```bash
  INSTALL_RAG=false docker compose up -d --build
  ```
- 镜像构建阶段会自动预下载 `BAAI/bge-small-zh-v1.5` 并构建 `runtime/knowledge_demo`，容器重启冷启动在数秒内完成；`agent_runtime` 数据卷会持久保留演示库、审计日志与每日 Token 预算记录。

---

## 3. Hugging Face Spaces 免费公网部署（推荐简历附在线体验链接）

1. 在 [Hugging Face Spaces](https://huggingface.co/spaces) 新建 Space，SDK 选择 **Docker**（Blank）。
2. 推送本仓库代码（或绑定 GitHub 仓库）。
3. 在 Space 根目录的 `README.md` 顶部添加 Spaces Frontmatter（或直接在新建的 Space 仓库配置）：
   ```yaml
   ---
   title: E-Commerce AI Support Workbench
   emoji: 🧾
   colorFrom: blue
   colorTo: yellow
   sdk: docker
   app_port: 7860
   ---
   ```
4. 在 Space **Settings → Variables and secrets** 中配置：
   - **Secret**：`DEEPSEEK_API_KEY`（如不配置，站点会自动以 `demo` 模式运行，仍可完整体验页面、SQLite 取消确认与 Hybrid RAG）；
   - **Variable**（推荐开启成本保护）：`DAILY_TOKEN_BUDGET=200000`、`RATE_LIMIT_RPM=20`。
5. 构建完成后即可获得公网 HTTPS 体验链接，可直接贴入简历与 GitHub 仓库简介。

---

## 4. Google Cloud Run / Render / Fly.io

容器默认读取 `$PORT` 环境变量（未设置时默认 `7860`），可直接用于任意无服务器容器平台：

```bash
# Google Cloud Run 示例
gcloud run deploy ecommerce-support-agent \
  --source . \
  --region asia-east1 \
  --memory 2Gi \
  --set-env-vars APP_MODE=auto,RETRIEVAL_MODE=hybrid,DAILY_TOKEN_BUDGET=300000,RATE_LIMIT_RPM=20 \
  --set-secrets DEEPSEEK_API_KEY=deepseek-api-key:latest \
  --allow-unauthenticated
```

> 若使用 Cloud Run 多实例水平扩展，请注意当前演示版使用单机本地 SQLite 与进程内 `WebSessionStore`；真实多副本生产部署应启用粘性会话（session affinity）并将 SQLite / 限流状态外置至 PostgreSQL 与 Redis。

---

## 5. 上线前安全与成本自检清单

- **密钥隔离**：API Key 仅通过平台 Secrets / 环境变量注入，`.dockerignore` 与 `.gitignore` 均已屏蔽 `.env`、`data/auth/` 与 `*.db`。
- **成本熔断**：务必按需设置 `DAILY_TOKEN_BUDGET`（默认 `500000` tokens/天）与 `RATE_LIMIT_RPM`（默认 `30` 次/分钟/会话），配合静态 `system` Prompt 保持 ~90%+ Prompt Cache 命中率。
- **权限边界**：公开演示时 `STRICT_AUTH=false` 允许访客一键切换 `alice / bob / charlie` 体验跨客户隔离；若用于受限内部演示，可设 `STRICT_AUTH=true` 强制本地密码校验。
- **可选 Langfuse 观测**：在环境变量填入 `LANGFUSE_PUBLIC_KEY` 与 `LANGFUSE_SECRET_KEY`（并安装 `langfuse`），即可零改动将工具链路与耗时指标上报至 Langfuse。
