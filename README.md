# Zex

Zex 是一个极简、可单二进制运行的 AI Agent Harness 核心。它提供科研与工程工作流能够依赖的最小 Agent 循环，而不是大而全的 coding 产品。

Zex 的第一版刻意只包含 OpenAI 兼容模型接入、普通终端 REPL、一次性任务、四个本地工具、事件流和简单会话落盘。它不包含 MCP、子 Agent、Plan Mode、插件系统、Web UI、复杂 TUI、向量库、RAG 或长期记忆。

## 构建

需要 Rust 1.85 或更新版本（edition 2024）。

```bash
cargo build --release
```

生成的二进制为 `target/release/zex`；Windows 上为 `target/release/zex.exe`。

## 配置

至少配置 API Key 和模型名。`ZEX_*` 变量优先，未设置时读取对应的 `OPENAI_*` 变量。

| 变量 | 备用变量 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `ZEX_API_KEY` | `OPENAI_API_KEY` | 无 | 必填 API Key |
| `ZEX_BASE_URL` | `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI 兼容 API 根地址 |
| `ZEX_MODEL` | `OPENAI_MODEL` | 无 | 必填模型名 |
| `ZEX_OPENAI_API` | 无 | `chat-completions` | API 协议：`chat-completions` 或 `responses` |
| `ZEX_BASH_TIMEOUT_SECONDS` | 无 | `60` | 单次 `bash` 工具超时 |
| `ZEX_AGENT_TIMEOUT_SECONDS` | 无 | `600` | 单轮 Agent 总超时 |
| `ZEX_MAX_STEPS` | 无 | `12` | 单轮最大模型调用步数 |
| `ZEX_MAX_TOOL_OUTPUT_CHARS` | 无 | `32000` | `read`/`bash` 返回内容上限 |
| `ZEX_SESSION_DIR` | 无 | `.zex/sessions` | 会话 JSON 保存目录 |

PowerShell 示例：

```powershell
$env:ZEX_API_KEY = "your-api-key"
$env:ZEX_BASE_URL = "https://api.openai.com/v1"
$env:ZEX_MODEL = "gpt-4.1-mini"
$env:ZEX_OPENAI_API = "responses"
```

POSIX shell 示例：

```bash
export ZEX_API_KEY="your-api-key"
export ZEX_BASE_URL="https://api.openai.com/v1"
export ZEX_MODEL="gpt-4.1-mini"
export ZEX_OPENAI_API="responses"
```

`base_url` 应指向包含 `/v1` 的 API 根地址。Zex 根据 `ZEX_OPENAI_API` 请求：

- `chat-completions`：`${base_url}/chat/completions`
- `responses`：`${base_url}/responses`

如果 Provider 只支持 Responses API，PowerShell 最小配置为：

```powershell
$env:ZEX_API_KEY = "your-api-key"
$env:ZEX_BASE_URL = "https://your-gateway.example/v1"
$env:ZEX_MODEL = "your-model"
$env:ZEX_OPENAI_API = "responses"
```

不要把 `/responses` 写进 `ZEX_BASE_URL`，Zex 会自动拼接端点。

## 使用

### 一次性任务

```bash
zex -p "读取 README 并总结成三句话"
```

从源码运行：

```bash
cargo run -- -p "读取 README 并总结成三句话"
```

### 交互式 REPL

```bash
zex
```

终端出现 `zex>` 后可连续输入消息。Unix 上按 Ctrl-D，Windows 上按 Ctrl-Z 后回车退出。

### 继续最近会话

```bash
zex --continue-session
```

也可继续最近会话后立即提交新任务：

```bash
zex --continue-session -p "继续上一轮工作并给出结论"
```

每次运行结束后，Zex 会把完整消息列表保存为 `.zex/sessions/<timestamp>.json`。`--continue-session` 加载文件名时间戳最大的会话；目录为空时从新会话开始。

## 内置工具

工具通过统一的 `Tool` trait 注册，Agent loop 不包含工具名称分支。

- `read`：读取 UTF-8 文件；相对路径基于启动 Zex 时的工作目录。长内容会截断。
- `write`：创建或完整覆盖 UTF-8 文件，并自动创建父目录。
- `edit`：在 UTF-8 文件中执行一次精确文本替换；目标缺失或出现多次时拒绝修改，避免含糊编辑。
- `bash`：在启动工作目录中通过系统 shell 执行命令。Windows 使用 `cmd /D /S /C`，其他平台使用 `sh -c`。命令有超时，stdout/stderr 会合并为结构化文本并截断。

## Agent 循环

每轮用户输入进入统一消息列表。Provider 返回普通文本时结束该轮；返回 tool calls 时，Zex 逐个执行已注册工具，将每个结果作为 `tool` 消息回灌，再请求模型继续。循环受到最大步数和整轮超时限制。

OpenAI 兼容 Provider 支持 Chat Completions 和 Responses 两种协议。两种协议都优先请求流式响应，并兼容网关忽略 `stream` 后返回普通 JSON。Responses 模式使用扁平 function tool 定义、`function_call`/`function_call_output` 输入项，并保留 Provider 输出项以支持推理模型的连续工具调用。

内部事件 channel 发出：

- 文本增量
- 工具开始
- 工具结束及成功状态
- 错误
- 单轮结束

当前 CLI 只负责打印事件；后续可在不改动 Agent loop 的情况下替换为 TUI 或其他消费者。

## 安全边界

Zex 第一版信任本地用户，不提供 OS 级沙箱、权限弹窗或命令审核。模型能够通过 `write`、`edit` 和 `bash` 修改文件或运行危险命令，其权限与当前操作系统用户相同。请只在可信目录与可接受的账户权限下运行，并自行检查重要数据备份。

`bash` 超时和输出截断用于限制挂起进程与上下文膨胀，不构成安全隔离。

## 最小自测

以下验证需要一个支持 OpenAI Chat Completions 或 Responses function calls 的可用模型或兼容网关。

1. 验证普通对话：

   ```bash
   cargo run -- -p "只回答：Zex 已连接"
   ```

   预期：终端输出模型回复并正常退出。

2. 验证 `read` 与工具结果回灌：

   ```bash
   cargo run -- -p "必须使用 read 工具读取 Cargo.toml，然后告诉我 package name"
   ```

   预期：终端显示 `[tool] read`、`[tool] read: done`，随后模型基于工具结果回答 `zex`。

3. 验证 `bash` 与工具结果回灌：

   Windows：

   ```powershell
   cargo run -- -p "必须使用 bash 工具执行 cd，然后报告当前目录"
   ```

   Linux/macOS：

   ```bash
   cargo run -- -p "必须使用 bash 工具执行 pwd，然后报告当前目录"
   ```

   预期：终端显示 `bash` 工具开始和完成，模型读取命令输出后继续回答。

4. 验证 REPL 连续对话：

   ```bash
   cargo run --
   ```

   先输入 `记住数字 37`，再输入 `刚才的数字是什么？`。预期第二轮回答保留同一会话上下文。

5. 验证会话恢复：退出 REPL 后运行：

   ```bash
   cargo run -- --continue-session -p "复述上一轮记住的数字"
   ```

   预期：Zex 加载 `.zex/sessions` 中最近的 JSON 会话并回答 `37`。

## 模块

- `src/provider`：Provider 抽象、OpenAI 兼容 Chat Completions/Responses 与流式解析
- `src/agent`：消息类型、事件、带最大步数和超时的 Agent loop
- `src/tools`：统一 Tool trait、注册表和四个内置工具
- `src/session.rs`：简单 JSON 会话保存与最近恢复
- `src/cli.rs`：clap 命令行参数
- `src/config.rs`：环境变量配置

Zex 保持 minimal core：能力通过清晰边界继续扩展，而不是提前把扩展系统耦合进核心。
