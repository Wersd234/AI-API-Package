# AI-RP-Proxy

纯透传两阶段 API 中继：把两台设备上的两个 llama-server 打包成一个 OpenAI 兼容 API 给 SillyTavern 用。

```
SillyTavern ──→ 本代理 (:5000) ──→ Stage 1 设备A (内容生成, 非流式)
                    │                      │ raw_text
                    │                      ▼
                    └────────────── Stage 2 设备B (文风打磨, 流式) ──→ SSE 逐字回 ST
```

**Stage 1 零注入**：收到 ST 的原样请求（含角色卡），所有采样参数原样透传。

**Stage 2 带润色提示词**：系统提示词在 `stage2_prompt.txt`（每次请求实时读取，改完不用重启）。它约束模型：只润色正文、删除思考碎片和元评论、`<UpdateVariable>`/`<scene>`/`<details>`/`<StatusPlaceHolderImpl/>` 等功能块逐字节原样保留。

**Stage 1 思考块清理**：如果 Stage 1 模型把思维链漏进 content（没用 --reasoning-format 时会这样），代理会先用正则剥掉 `<think>`/`<｜begin▁of▁thinking｜>` 等标签块再送给 Stage 2（`STAGE1_STRIP_REASONING=true`，默认开）。元评论类无标签垃圾由提示词约束 Gemma 删除。

> 建议：Stage 1 的 llama-server 加上 `--reasoning-format deepseek`（或对应格式），让思维链进 `reasoning_content` 字段从源头分离，代理的正则只是双保险。

## Docker 部署（推荐）

```bash
git clone https://github.com/Wersd234/AI-API-Package.git
cd AI-API-Package
cp .env.example .env   # 编辑两台后端设备的 IP 和模型名

docker compose up -d --build
# 或纯 docker：
# docker build -t ai-rp-proxy .
# docker run -d --name ai-rp-proxy --env-file .env -p 5000:5000 ai-rp-proxy
```

- 后端在局域网其他机器上即可，容器默认桥接网络可以访问 LAN IP
- `stage2_prompt.txt` 以只读卷挂载进容器，**改提示词立即生效**，不用重建不用重启
- 日志：`docker logs -f ai-rp-proxy`（两阶段进度、tok/s、耗时都在里面）

## 裸机部署

```bash
pip install -r requirements.txt
cp .env.example .env   # 编辑两台设备的 IP
python main.py         # 或 uvicorn main:app --host 0.0.0.0 --port 5000
```

SillyTavern 设置：Chat Completion 源 → `http://<本机IP>:5000/v1`，模型 `rp-two-stage`。

## 无 GPU 本地测试

```bash
python mock_backends.py   # 终端1：起两个假后端 (8080/8081)
python main.py            # 终端2：代理默认指向 localhost 这两个端口
# 终端3：
curl -N http://localhost:5000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"stream":true,"messages":[{"role":"user","content":"hi"}]}'
MOCK_STAGE1_DELAY=12 python mock_backends.py  # 慢速 Stage 1，验证心跳注释
```

## 行为说明

| 场景 | 行为 |
|------|------|
| ST 流式请求（默认） | Stage 1 非流式后台生成（期间每 5s 发 SSE 心跳注释）→ Stage 2 流式逐字输出 |
| ST 非流式请求 | 两级都非流式，Stage 2 为空时回退 Stage 1 原文 |
| 用户中途取消 | Stage 1 任务被取消；Stage 2 上游连接关闭，不再浪费 GPU |
| Stage 1 挂了 | 流式：SSE error 事件 + [DONE]；非流式：HTTP 502 带后端报错原文 |
| ST 没发 max_tokens | Stage 1 兑底用 `STAGE1_MAX_TOKENS`（防 llama-server 无限生成） |
| Stage 1 输出含思考块 | 代理正则剥除标签块后日志记录剥除字符数；元评论由 Stage 2 提示词删除 |
| Stage 2 token 上限 | `max(STAGE2_MAX_TOKENS, ST的max_tokens)`，防润色扩写被截断 |

## 配置项

见 `.env.example`。`STAGE2_TEMPERATURE` / `STAGE2_TOP_P` 留空（注释掉）则继承 ST 发来的值。