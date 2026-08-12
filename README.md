# Zex

Zex 是一个极简、可单二进制运行的通用 coding harness。它提供最小 Agent 循环与可观测终端界面，而不是大而全的 IDE 或实验管理平台。

Zex 当前只包含 OpenAI 兼容模型接入、ReAct 循环、四个本地工具、核心事件流、ratatui TUI、headless 模式、TOML 配置和 JSONL 会话管理。它不包含 MCP、子 Agent、Plan Mode、插件市场、IDE、科研指标、实验记录、向量库、RAG 或长期记忆。

## 构建

需要 Rust 1.85 或更新版本（edition 2024）。

```bash
cargo build --release
```

生成的二进制为 `target/release/zex`；Windows 上为 `target/release/zex.exe`。

## 目录与文件格式

Zex 按以下顺序合并配置，后者覆盖前者：

1. 全局 `config.toml`
2. 当前工作目录的 `.zex/config.toml`
3. 环境变量

全局目录遵循操作系统标准：

- Windows：`%APPDATA%\zex\config.toml`
- macOS：`~/Library/Application Support/zex/config.toml`
- Linux：`$XDG_CONFIG_HOME/zex/config.toml`，未设置时为 `~/.config/zex/config.toml`

可用 `ZEX_CONFIG_DIR` 覆盖整个全局目录，便于便携安装和隔离测试。

会话默认保存在同一全局目录下的 `sessions/<id>.jsonl`。每个文件第一行是格式版本、会话 ID、创建与更新时间；后续每行是一条 Agent 消息。会话只保存消息，不保存 API Key 或其他运行配置。

项目配置示例：

```toml
# .zex/config.toml
model = "gpt-4.1-mini"
base_url = "https://api.openai.com/v1"
openai_api = "responses"
max_turns = 12
bash_timeout_seconds = 60
agent_timeout_seconds = 600
max_tool_output_chars = 32000
```

支持的 TOML 字段：

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `api_key` | 无 | API Key；允许配置，但推荐只用环境变量 |
| `model` | 无 | 必填模型名 |
| `base_url` | `https://api.openai.com/v1` | OpenAI 兼容 API 根地址 |
| `openai_api` | `chat-completions` | `chat-completions` 或 `responses` |
| `max_turns` | `12` | 单轮最大 Provider 调用次数 |
| `bash_timeout_seconds` | `60` | 单次 `bash` 工具超时 |
| `agent_timeout_seconds` | `600` | 单轮 Agent 总超时 |
| `max_tool_output_chars` | `32000` | `read`/`bash` 返回内容上限 |
| `session_dir` | 全局 `sessions` 目录 | 自定义会话目录；相对路径基于项目工作目录 |

环境变量优先级高于两个 TOML 文件。API Key 使用 `ZEX_API_KEY`，未设置或为空时再读 `OPENAI_API_KEY`；因此可把非敏感配置提交到项目配置，同时保证密钥不进入仓库。

| 变量 | 备用变量 | 对应字段 |
| --- | --- | --- |
| `ZEX_CONFIG_DIR` | 无 | 全局配置与默认会话目录根路径 |
| `ZEX_API_KEY` | `OPENAI_API_KEY` | `api_key` |
| `ZEX_MODEL` | `OPENAI_MODEL` | `model` |
| `ZEX_BASE_URL` | `OPENAI_BASE_URL` | `base_url` |
| `ZEX_OPENAI_API` | 无 | `openai_api` |
| `ZEX_MAX_TURNS` | 无 | `max_turns` |
| `ZEX_BASH_TIMEOUT_SECONDS` | 无 | `bash_timeout_seconds` |
| `ZEX_AGENT_TIMEOUT_SECONDS` | 无 | `agent_timeout_seconds` |
| `ZEX_MAX_TOOL_OUTPUT_CHARS` | 无 | `max_tool_output_chars` |
| `ZEX_SESSION_DIR` | 无 | `session_dir` |

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

`--prompt` / `-p` 始终使用 headless 输出，不进入 TUI。

### 交互式 TUI

在 stdin 和 stdout 都连接终端时，不带 `--prompt` 启动 TUI：

```bash
zex
```

主区显示用户/助手对话与工具执行过程；宽终端使用右侧状态栏，窄终端使用底部状态栏，展示当前状态、当前 tool 与最近错误摘要。

- 输入文本后按 Enter 提交。
- 空输入时按 Esc 或 `q` 退出。
- 当前轮执行期间按 `q` 或 Ctrl-C 退出。

stdin 或 stdout 不是 TTY 时自动保留普通终端 REPL：

```bash
echo "解释当前目录" | zex
```

非 TTY REPL 沿用 `zex>` 提示；Unix 上 Ctrl-D，Windows 上 Ctrl-Z 后回车退出。

### 会话列表与恢复

列出会话；输出包含 ID、最后更新时间、消息数和首条用户消息摘要：

```bash
zex sessions
```

恢复最近更新的会话并进入交互模式：

```bash
zex resume
```

恢复指定会话：

```bash
zex resume 20260812-143012-1a2b3c4d
```

恢复后立即运行一次性任务：

```bash
zex resume 20260812-143012-1a2b3c4d -p "继续上一轮工作并给出结论"
```

也可省略 ID，直接恢复最近会话：

```bash
zex resume -p "继续上一轮工作并给出结论"
```

新运行在退出时创建一个 JSONL 文件；恢复已有会话时原位更新同一文件，不复制分叉会话。若命令、Provider 或工具报错，Zex 仍会保存当前消息历史，再返回错误。

## 内置工具

工具通过统一的 `Tool` trait 注册，Agent loop 不包含工具名称分支。

- `read`：读取 UTF-8 文件；相对路径基于启动 Zex 时的工作目录。长内容会截断。
- `write`：创建或完整覆盖 UTF-8 文件，并自动创建父目录。
- `edit`：在 UTF-8 文件中执行一次精确文本替换；目标缺失或出现多次时拒绝修改，避免含糊编辑。
- `bash`：在启动工作目录中通过系统 shell 执行命令。Windows 使用 `cmd /D /S /C`，其他平台使用 `sh -c`。命令有超时，stdout/stderr 会合并为结构化文本并截断。

## Agent 循环

每轮用户输入进入统一消息列表。Provider 返回普通文本时结束该轮；返回 tool calls 时，Zex 逐个执行已注册工具，将每个结果作为 `tool` 消息回灌，再请求模型继续。循环受到最大步数和整轮超时限制。

OpenAI 兼容 Provider 支持 Chat Completions 和 Responses 两种协议。两种协议都优先请求流式响应，并兼容网关忽略 `stream` 后返回普通 JSON。Responses 模式使用扁平 function tool 定义、`function_call`/`function_call_output` 输入项，并保留 Provider 输出项以支持推理模型的连续工具调用。

## 事件设计与模块划分

核心通过 `tokio::sync::mpsc::UnboundedSender<AgentEvent>` 单向推送状态，不依赖 ratatui 或终端类型：

- `MessageDelta { role, delta }`：用户消息或助手文本增量。
- `ToolStart { call_id, name }`：工具开始；`call_id` 用于关联完成事件。
- `ToolEnd { call_id, name, output, is_error }`：工具完成、输出与失败状态。
- `Error { message }`：Provider、超时、步数上限等轮次错误。
- `TurnEnd`：一轮正常结束。

模块边界：

- `src/agent/event.rs`：与 UI 无关的事件契约。
- `src/agent/loop.rs`、`src/provider`：生产事件，完全不引用 TUI。
- `src/tui.rs`：消费事件并维护只用于渲染的视图状态；使用 ratatui + crossterm，与 tokio `select!` 配合处理 Agent 事件、键盘输入和重绘。
- `src/headless.rs`：同一事件流的纯文本消费者，供 `-p` 和无 TTY 场景使用。
- `src/main.rs`：仅负责检测模式并装配 core、TUI 或 headless 消费者。

TUI 不调用工具、不解析 Provider 响应，也不持有 Agent 业务状态；core 不知道事件由 TUI、headless 或其他消费者渲染。

## 安全边界

Zex 第一版信任本地用户，不提供 OS 级沙箱、权限弹窗或命令审核。模型能够通过 `write`、`edit` 和 `bash` 修改文件或运行危险命令，其权限与当前操作系统用户相同。请只在可信目录与可接受的账户权限下运行，并自行检查重要数据备份。

`bash` 超时和输出截断用于限制挂起进程与上下文膨胀，不构成安全隔离。

## 最小自测

第 1 项可离线运行；第 2–8 项需要一个支持 OpenAI Chat Completions 或 Responses function calls 的可用模型或兼容网关。

1. 运行自动化检查：

   ```bash
   cargo fmt --check
   cargo test
   cargo clippy --all-targets --all-features -- -D warnings
   ```

2. 验证 `-p` 保持 headless：

   ```bash
   cargo run -- -p "只回答：Zex 已连接"
   ```

   预期：即使在交互终端中也不进入 alternate screen，直接输出模型回复并正常退出。

3. 验证 `read` 与 headless 工具事件：

   ```bash
   cargo run -- -p "必须使用 read 工具读取 Cargo.toml，然后告诉我 package name"
   ```

   预期：终端显示 `[tool] read`、`[tool] read: done`，随后模型基于工具结果回答 `zex`。

4. 验证 `bash` 与工具结果回灌：

   Windows：

   ```powershell
   cargo run -- -p "必须使用 bash 工具执行 cd，然后报告当前目录"
   ```

   Linux/macOS：

   ```bash
   cargo run -- -p "必须使用 bash 工具执行 pwd，然后报告当前目录"
   ```

   预期：终端显示 `bash` 工具开始和完成，模型读取命令输出后继续回答。

5. 验证 TUI 连续对话与状态：

   ```bash
   cargo run --
   ```

   预期进入 TUI。先输入 `记住数字 37`，再输入 `必须使用 read 读取 Cargo.toml，然后告诉我刚才的数字`。主区应显示两轮对话与 `read` 的 running/done 过程；状态区应在 idle、thinking、running tool 间切换，第二轮回答保留上下文。

6. 验证错误摘要：使用错误 API Key 启动 TUI 并提交一句话。预期主区出现错误，状态区显示最近错误摘要，终端仍可退出并恢复。

7. 验证无 TTY 回退：

   Windows PowerShell：

   ```powershell
   "只回答：headless" | cargo run --
   ```

   Linux/macOS：

   ```bash
   printf '只回答：headless\n' | cargo run --
   ```

   预期不进入 TUI，使用普通 REPL/事件输出。

8. 验证会话列表与恢复：退出 TUI 后运行：

   ```bash
   cargo run -- sessions
   cargo run -- resume -p "复述上一轮记住的数字"
   ```

   预期：第一条命令显示刚保存的会话 ID；第二条命令加载最近的 JSONL 会话并回答 `37`。

## 模块

- `src/provider`：Provider 抽象、OpenAI 兼容 Chat Completions/Responses 与流式解析
- `src/agent`：消息类型、事件、带最大 Provider 轮次和超时的 Agent loop
- `src/tools`：统一 Tool trait、注册表和四个内置工具
- `src/tui.rs`：ratatui/crossterm 可观测界面
- `src/headless.rs`：一次性任务与无 TTY 的文本界面
- `src/session.rs`：版本化 JSONL 会话保存、列表与恢复
- `src/cli.rs`：clap 命令行参数
- `src/config.rs`：全局/项目 TOML 合并与环境变量覆盖

Zex 保持 minimal core：能力通过清晰边界继续扩展，而不是提前把扩展系统耦合进核心。
