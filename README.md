# Zex

Zex 是一个极简、可单二进制运行的通用 coding harness。它提供最小 Agent 循环与可观测终端界面，而不是大而全的 IDE 或实验管理平台。

Zex 当前只包含 OpenAI 兼容模型接入、ReAct 循环、六个开箱可用的本地工具、核心事件流、ratatui TUI、headless 模式、斜杠命令、规则式上下文 compact、TOML 配置和 JSONL 会话管理。它不包含 MCP、子 Agent、Plan Mode、权限审批流、插件市场、IDE、科研指标、实验记录、向量库、RAG 或长期记忆。

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

会话默认保存在同一全局目录下的 `sessions/<id>.jsonl`。每个文件第一行是格式版本、会话 ID、创建/更新时间、保存时的 model 和当前 `thinking_level`；后续每行是一条 Agent 消息。会话不保存 API Key、base URL 或其他敏感运行配置。恢复会话会恢复消息与思考级别，但不改变当前通过 `/model` 选择的模型。

项目配置示例：

```toml
# .zex/config.toml
active_model = { provider_id = "openai", model_id = "gpt-4.1-mini" }
max_turns = 12
tool_timeout_seconds = 60
agent_timeout_seconds = 600
max_tool_output_chars = 32000
max_context_chars = 120000
compact_keep_turns = 6
default_thinking_level = "medium"
hide_thinking_block = false

[[providers]]
id = "openai"
display_name = "OpenAI"
base_url = "https://api.openai.com/v1"
api_key = "your-api-key"
openai_api = "responses"

[[providers.models]]
id = "gpt-4.1-mini"
display_name = "GPT-4.1 Mini"

[providers.models.thinking]
min_level = "low"
max_level = "max"
supported = ["off", "low", "medium", "high", "xhigh", "max"]
mode = "effort"

[providers.models.compat]
supports_reasoning_effort = true
supports_interleaved_thinking = true

[providers.models.compat.reasoning_effort_map]
low = "low"
medium = "medium"
high = "high"
xhigh = "xhigh"
max = "max"
```

支持的 TOML 字段：

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `active_model` | 无 | 当前模型的 `{ provider_id, model_id }` 引用；由 `/model` 写入 |
| `providers` | `[]` | Provider 及其模型目录；由 `/provider` 管理 |
| `max_turns` | `12` | 单轮最大 Provider 调用次数 |
| `tool_timeout_seconds` | `60` | 所有内置工具的默认单次超时；每次调用可用 `timeout_seconds` 覆盖 |
| `agent_timeout_seconds` | `600` | 单轮 Agent 总超时 |
| `max_tool_output_chars` | `32000` | 所有内置工具返回内容的统一字符上限 |
| `max_context_chars` | `120000` | 上下文字符预算近似值；达到 85% 时自动 compact |
| `compact_keep_turns` | `6` | compact 时完整保留的最近用户轮次数 |
| `default_thinking_level` | `medium` | 新会话默认思考强度：`off`、`minimal`、`low`、`medium`、`high`、`xhigh`、`max` |
| `hide_thinking_block` | `false` | 是否隐藏 TUI 中默认折叠的思考卡片；隐藏不删除会话数据，也不影响模型实际思考 |
| `session_dir` | 全局 `sessions` 目录 | 自定义会话目录；相对路径基于项目工作目录 |
| `theme` | 内置配色 | TUI 配色覆盖，见下方 `[theme]` 说明 |

### `[theme]` 自定义配色

TUI 的全部颜色都可以在 `config.toml`（全局或项目，项目逐键覆盖全局）的 `[theme]` 段里覆盖。值为 `"#rgb"` 或 `"#rrggbb"` 十六进制颜色，或 `"default"` 表示跟随终端自身的颜色；未设置的键保持内置默认（青蓝主色 + 紫/琥珀点缀的中性灰配色）。

```toml
[theme]
accent_primary = "#7dcfff"    # 主强调色（天蓝）：输入提示、选中项
accent_secondary = "#bb9af7"  # 次强调色（紫）：思考块
command = "#ff9e64"           # 命令/工具执行（橙）
text = "#c0caf5"              # 正文
text_dim = "#737aa2"          # 次要文字
border = "#292e42"            # 边框
ok = "#9ece6a"                # 成功
bad = "#f7768e"               # 失败
background = "default"        # 透明底座：跟随终端背景
```

全部可配置键：`background`、`surface`、`surface_hover`、`surface_raised`、`text`、`text_strong`、`text_dim`、`text_faint`、`gray_dim`、`accent_primary`、`accent_secondary`、`accent_user`、`accent_thinking`、`accent_tool`、`border`、`border_active`、`ok`、`bad`、`command`、`running`、`model_accent`、`md_code`、`code_bg`、`diff_add_bg`、`diff_del_bg`、`wordmark_ink`。其中 `accent_thinking` 默认跟随 `accent_secondary`，`accent_tool` 默认跟随 `text_faint`，可单独覆盖。

Zex 启动时从 `https://models.dev/api.json` 刷新模型思考能力，并把原始响应缓存为全局目录下的 `models-dev-cache.json`。刷新失败时优先使用缓存；缓存也不可用时使用安全默认 `off/low/medium/high`。模型优先按 Provider ID 匹配；自定义 Provider ID 还会用配置的 API base URL 匹配 models.dev Provider；仍无法匹配时，只对全局唯一的模型 ID 使用发现结果。`reasoning_options` 中的 effort 值会映射到本地固定梯子；`reasoning = false` 强制为 `off`。只有 toggle 或 token-budget、但没有 effort 选项的模型不会被误发 `reasoning_effort`。Provider/模型的手动 `[thinking]` 与 `[compat]` 配置始终覆盖 models.dev 数据。

旧的顶层 `api_key`、`model`、`base_url`、`openai_api` 仍会在没有 `providers` 时作为单个 `default` Provider 载入；首次从 `/provider` 保存后会写入新结构并移除这些顶层字段。对应环境变量仅参与这个旧配置迁移路径。

| 变量 | 备用变量 | 对应字段 |
| --- | --- | --- |
| `ZEX_CONFIG_DIR` | 无 | 全局配置与默认会话目录根路径 |
| `ZEX_API_KEY` | `OPENAI_API_KEY` | `api_key` |
| `ZEX_MODEL` | `OPENAI_MODEL` | `model` |
| `ZEX_BASE_URL` | `OPENAI_BASE_URL` | `base_url` |
| `ZEX_OPENAI_API` | 无 | `openai_api` |
| `ZEX_MAX_TURNS` | 无 | `max_turns` |
| `ZEX_TOOL_TIMEOUT_SECONDS` | 无 | `tool_timeout_seconds` |
| `ZEX_AGENT_TIMEOUT_SECONDS` | 无 | `agent_timeout_seconds` |
| `ZEX_MAX_TOOL_OUTPUT_CHARS` | 无 | `max_tool_output_chars` |
| `ZEX_MAX_CONTEXT_CHARS` | 无 | `max_context_chars` |
| `ZEX_COMPACT_KEEP_TURNS` | 无 | `compact_keep_turns` |
| `ZEX_DEFAULT_THINKING_LEVEL` | 无 | `default_thinking_level` |
| `ZEX_HIDE_THINKING_BLOCK` | 无 | `hide_thinking_block` |
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

TUI 占满 terminal 工作区，但不铺自定义整屏背景。它保持单列时间流，不使用左侧对话、右侧工具的仪表盘分栏：

1. 主区：可滚动时间流，按发生顺序显示 user/assistant 文本、思考卡片和 tool 卡片。短时间流靠近输入区向上生长，长时间流占满主区并正常滚动；当前 core 未生产 planning/todo 事件，因此不会为不存在的数据预留面板。滚轮与翻页采用缓动平滑滚动，鼠标命中区域跟随视觉位置。
2. 斜杠补全：输入 `/` 时从输入框上方弹出，不打断 statusline，也不占永久分区。
3. 顶栏单行显示 git 分支、脏文件数与折叠后的 cwd；turn 运行中在主区下方出现一条状态行：80ms 节奏的 braille spinner、带扫光的活动词（thinking 紫 / working 绿 / error 红）、工具计数与耗时，右侧 `esc interrupt` 提示。
4. 固定底栏：圆角输入框顶边内嵌会话标题，底边内嵌 `model · think` 与真实 Provider 输出速率、上下文占用百分比；再往下是一条不换行的快捷键提示行，toast 在同一行淡入淡出（约 4 秒）。多行输入把框架增高到最多五行。输入框边框在聚焦/忙碌与空闲之间做 ~120ms 颜色过渡。

根区域、空状态、用户消息和 assistant 正文都使用终端原生背景；近黑 surface 只用于用户消息带、代码块、展开的 tool 输出和斜杠补全。界面使用 Ink Indigo 调色板：冷近黑底色、四级灰阶正文、柔和蓝主 accent 与紫色次 accent，低饱和 success/error 色，同一屏幕至多三个色相。用户消息使用 `❯` 引导加全宽 surface 带，assistant 正文与卡片使用低亮度 `┃` 左轨连成单一时间流；两者都不显示 `YOU` / `ASSISTANT` 标签。基础 Markdown 标题、列表、引用和代码围栏形成清楚但克制的层级。

动效语言集中在"有事发生"的时刻：assistant 流式回复末尾跟随一枚慢速呼吸的 `▍` 光标，运行中的 tool 卡头部带实时 spinner，landing 页 ZEX 字标有 4 秒对角扫光叠加 5 秒呼吸脉冲、hero 边框以 4.6 秒周期向主 accent 呼吸。工作区空闲时不运行任何常驻动画。所有环境动效由 wall clock 在渲染时推导，连续状态（滚动、焦点、toast 透明度）每帧指数趋近目标。

思考内容优先读取 Provider 的 `reasoning_content`、`reasoning`、`reasoning_details` / `thinking_blocks` 或 Responses API reasoning summary；缺少显式字段时再解析完整的 `<think>...</think>`。思考与 assistant 最终回答、tool call 分开保存，并在单一时间流中显示为默认折叠卡片。`/thinking show|hide` 控制卡片可见性并持久化，隐藏期间数据仍保留，因此再次显示或恢复会话时可重新渲染；该开关独立于 `/think` 的模型思考强度。

Provider 与模型都可声明 `[thinking]`（`min_level`、`max_level`、可选精确 `supported`、`mode = "effort"`）和 `[compat]`（`supports_reasoning_effort`、`supports_interleaved_thinking`、`reasoning_effort_map`）。合并优先级为安全默认 → models.dev → Provider 手动配置 → 模型手动配置。缺失能力时使用安全范围 `off/low/medium/high`，因此 `xhigh` 和 `max` 会降级到 `high`，不会原样发送未知值。工具调用轮次仅在目标模型声明支持 interleaved thinking 时回传 reasoning；最终回答及不支持该能力的模型历史会在请求层剥离 reasoning，但会话中的可查看内容仍保留。

思考和 tool 卡片共用 Zex 摘要语法。思考卡为 `think · level · active|done · summary`；tool 卡为 `tool · subject · result · duration`，其中 `bash` 显示 `exit N`，`read` 显示行数，`grep` / `glob` 显示匹配数，`write` / `edit` 显示真实 diff 计数 `+a −b`。默认折叠态严格占一行；点击标题行或选中后按 Enter / Space 可展开或折叠。`write` / `edit` 的展开卡片展示基于真实文件内容的 diff——覆盖写也能看到被删除的行（`+` / `-` 全宽色带）；超过 512 KiB 的文件或恢复的旧会话回退为按工具参数推导的 diff。`Ctrl-O` 批量展开或折叠全部卡片。错误默认只显示首行摘要，`Ctrl-E` 展开或收起详情。Assistant 流式增量合并到当前消息，工具结果保留在卡片内；assistant 结论继续作为普通 Markdown 正文呈现。TUI 按固定帧率差分重绘：状态变化、输入与新事件立即触发，进行中的动效（滚动缓动、spinner、扫光、toast 淡化）在活动期间逐帧推进，完全静止时不产生重绘。

| 快捷键 | idle 模式 | turn 运行中 |
| --- | --- | --- |
| Enter | 发送非空输入 | — |
| Shift-Enter / Alt-Enter | 插入换行 | — |
| Ctrl-C | 退出 TUI | 中断当前 turn，返回 idle |
| Esc | 关闭补全、tool 详情、取消选择、清空草稿或回到底部；无可取消状态时退出 | 关闭当前 UI 选择，不中断 turn |
| 鼠标滚轮 | 平滑滚动时间流 | 平滑滚动时间流 |
| PageUp / PageDown | 对话历史翻页 | 对话历史翻页 |
| Home / End | 跳到历史顶部 / 底部 | 跳到历史顶部 / 底部 |
| Tab / Shift-Tab | 补全打开时接受当前命令；输入为空时选择下一个 / 上一个思考或 tool 卡片 | 选择下一个 / 上一个思考或 tool 卡片 |
| Up / Down | 补全打开时选择上一项 / 下一项；否则浏览已发送输入并恢复草稿 | — |
| Space | 激活当前列表项或已选卡片 | 激活已选卡片 |
| Ctrl-O | 批量展开 / 折叠全部思考和 tool 卡片 | 同左 |
| Ctrl-E | 展开 / 折叠最近一条错误详情 | 同左 |
| Ctrl-T | 循环当前模型声明的可用级别并持久化 | — |

粘贴使用终端 bracketed paste，允许直接粘贴多行内容。当前 turn 运行时输入区锁定，避免草稿与执行中状态混淆；Ctrl-C 中断后，已输入的用户消息保留，未完成的 assistant/tool 状态不会进入后续 Provider 上下文。

### 斜杠命令

TUI 输入框、非 TTY REPL 和一次性 `zex -p` 使用同一个命令注册表与解析模块；`/help` 和补全列表因此不会漂移。输入 `/` 后按前缀过滤，例如 `/se` 只显示 `/sessions`。Up/Down 选择，Tab 补全，Enter 在前缀未完整时先补全、命令完整时执行，Esc 关闭。命中的斜杠命令不会作为普通用户消息发给模型；未知命令返回可读错误。`/model`、`/provider`、`/resume` 和 `/help` 都替换主区显示干净列表或表单，使用统一左侧选中指示，Esc 退出后精确恢复原时间流滚动位置。模型、thinking、compact、新建与恢复会话等短暂状态反馈只更新底栏或显示约 4 秒 toast，不写入主 feed；`/sessions` 仍作为需要阅读的结果进入时间流。`/thinking` 是 TUI 显示设置，headless 模式会返回明确错误。

| 命令 | 行为 |
| --- | --- |
| `/help` | 打开有限高度的命令面板；Esc 关闭，不写入对话时间流 |
| `/model` | 用主区轻量列表选择已配置模型；Enter 立即切换并持久化，Esc/q 取消 |
| `/provider` | 打开双栏 Provider 配置页，管理 Provider、端点、密钥及模型列表；选中 Provider 后按 `f` 从其 OpenAI-compatible `/models` 接口导入模型 |
| `/clear` | 清空当前 TUI/REPL 上下文；TUI 同时清空对话视图。不删除磁盘会话，下一条普通消息创建新会话 |
| `/sessions` | 查看保存的会话；复用 `SessionStore::list` 列出 ID、更新时间、消息数和预览 |
| `/resume [id]` | 无参数时打开历史会话选择列表；有参数时直接恢复指定会话。只恢复消息，不改变当前模型 |
| `/compact` | 立即压缩旧上下文，显示压缩前后字符数、约释放字符数、保留轮次和摘要数量 |
| `/think [off\|minimal\|low\|medium\|high\|xhigh\|max]` | 无参数时显示当前请求值、有效值与模型可用级别；有参数时设置并自动 clamp/map。写入项目默认值及活跃会话 |
| `/thinking [show\|hide]` | 无参数时显示当前思考卡片可见性；有参数时独立设置并写入项目 `.zex/config.toml` |

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

新运行在退出时创建一个 JSONL 文件；恢复已有会话时原位更新同一文件，不复制分叉会话。若 Provider 或工具报错，Zex 仍会保存当前消息历史，再返回错误；无效斜杠命令只在当前界面显示错误，不写入模型上下文。

## 内置工具

工具通过统一的 `Tool` trait 注册，Agent loop 不包含工具名称分支。六个工具全部编译进 Zex，不调用 `rg`、`fd` 或其他可选外部二进制。每个 schema 都接受可选正整数 `timeout_seconds`；未指定时使用 `tool_timeout_seconds`。Registry 对所有成功输出统一执行 `max_tool_output_chars` 截断，并给参数、超时、I/O 和执行错误补充工具名上下文。

- `read`：读取 UTF-8 文件；相对路径基于启动 Zex 时的工作目录。长内容会截断。
- `write`：创建或完整覆盖 UTF-8 文件，并自动创建父目录。写入前捕获旧内容，供 TUI/headless 展示真实 diff（覆盖可见被删行）；变更记录不进入模型上下文。
- `edit`：在 UTF-8 文件中执行一次精确文本替换；目标缺失或出现多次时拒绝修改，避免含糊编辑。与 `write` 一样捕获修改前后内容用于 diff 展示。
- `grep`：使用 Rust `regex` + `ignore` 递归搜索 UTF-8 文件内容，返回 `path:line:content`。主要 schema：`pattern`、`path`、`case_sensitive`、`file_glob`、`hidden`、`max_results`。默认尊重 `.gitignore`、全局 gitignore 和 `.git/info/exclude`；二进制或非 UTF-8 文件跳过。
- `glob`：使用 Rust `globset` + `ignore` 按路径 glob 查找文件或目录。主要 schema：`pattern`、`path`、`hidden`、`max_results`。无 `/` 的模式在任意深度匹配，目录结果带 `/`；默认尊重 Git ignore。
- `bash`：仅用于其他系统命令。在启动工作目录中通过系统 shell 执行；Windows 使用 `cmd /D /S /C`，其他平台使用 `sh -c`。stdout/stderr 合并为结构化文本后执行统一截断。

工具描述明确约定：搜文件内容用 `grep`；找文件或目录用 `glob`；其他系统命令才用 `bash`。

## Agent 循环

每轮用户输入进入统一消息列表。Provider 返回普通文本时结束该轮；返回 tool calls 时，Zex 逐个执行已注册工具，将每个结果作为 `tool` 消息回灌，再请求模型继续。循环受到最大步数和整轮超时限制。

### 上下文 compact

Compact 是 core 的确定性规则，不调用外部总结模型：

1. system prompt 始终完整保留。
2. 最近 `compact_keep_turns` 个用户轮次及其 assistant/tool 消息完整保留。
3. 更早轮次压成一条 system 摘要：用户和 assistant 文本保留首尾关键片段；旧 tool 输出优先压成工具名、首尾各 180 字符、原省略长度。
4. 上下文字符数达到 `max_context_chars` 的 85% 时，在 TUI 和 headless 共用的 Agent core 中自动 compact；`/compact` 可随时手动触发。若保留配置轮数后仍超过预算，会逐步减少完整保留轮次，但至少保留最近 1 轮。
5. TUI/REPL 反馈约释放字符数、compact 前后字符数、保留完整轮次、摘要旧轮次和 tool 输出数量。Compact 后的消息直接用于后续 Provider 请求和会话持久化。

OpenAI 兼容 Provider 支持 Chat Completions 和 Responses 两种协议。两种协议都优先请求流式响应，并兼容网关忽略 `stream` 后返回普通 JSON。Responses 模式使用扁平 function tool 定义、`function_call`/`function_call_output` 输入项，并保留 Provider 输出项以支持推理模型的连续工具调用。

## 事件设计与模块划分

核心通过 `tokio::sync::mpsc::UnboundedSender<AgentEvent>` 单向推送状态，不依赖 ratatui 或终端类型：

- `MessageDelta { role, delta }`：用户消息或助手文本增量。
- `ToolStart { call_id, name, arguments }`：工具开始；`call_id` 用于关联完成事件，参数仅由消费者决定是否展示。
- `ToolEnd { call_id, name, output, is_error }`：工具完成、输出与失败状态；`write` / `edit` 成功时附带 `change`（修改前后内容），供消费者渲染真实 diff，不进入模型上下文。
- `Error { message }`：Provider、超时、步数上限等轮次错误。
- `ContextCompacted { stats }`：core 自动 compact 后的字符数、释放量和保留轮次统计。
- `TurnCancelled`：调用方中断当前轮；核心丢弃未完成的 assistant/tool 上下文并恢复可继续输入状态。
- `TurnEnd`：一轮正常结束。

模块边界：

- `src/agent/event.rs`：与 UI 无关的事件契约。
- `src/agent/loop.rs`、`src/provider`：生产事件，完全不引用 TUI。
- `src/tui.rs`：消费事件并维护只用于渲染的视图状态；使用 ratatui + crossterm，与 tokio `select!` 配合处理 Agent 事件、键盘输入和重绘。
- `src/headless.rs`：同一事件流的纯文本消费者，供 `-p` 和无 TTY 场景使用；`write` / `edit` 完成后额外打印一行 `[change] path: +a −b` 变更统计。
- `src/main.rs`：仅负责检测模式并装配 core、TUI 或 headless 消费者。

TUI 不调用工具、不解析 Provider 响应，也不持有 Agent 业务状态；core 不知道事件由 TUI、headless 或其他消费者渲染。

## 安全边界

Zex 第一版信任本地用户，不提供 OS 级沙箱、权限弹窗或命令审核。模型能够通过 `write`、`edit` 和 `bash` 修改文件或运行危险命令，其权限与当前操作系统用户相同。请只在可信目录与可接受的账户权限下运行，并自行检查重要数据备份。

工具超时、输出截断与 compact 用于限制挂起执行和上下文膨胀，不构成安全隔离。

## 最小自测

第 1 项可离线运行；第 2–9 项需要一个支持 OpenAI Chat Completions 或 Responses function calls 的可用模型或兼容网关。

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

   预期进入 TUI。先输入 `记住数字 37`，再输入 `必须使用 read 读取 Cargo.toml，然后告诉我刚才的数字`。主区应显示两轮单一时间流、默认折叠的 `think · level · state · summary` 卡片，以及默认折叠的 `read · Cargo.toml · N lines · duration` 卡片；底栏应在 `idle`、`thinking`、`running` 间切换，并按 model → think → context → status 显示右侧信息。点击卡片标题可展开/折叠；Tab 选择卡片后可按 Enter / Space 激活；Ctrl-O 批量展开/折叠全部卡片。鼠标滚轮或 PageUp/PageDown 滚动历史。第二轮回答应保留上下文。

6. 验证多行输入与中断：用 Shift-Enter 或 Alt-Enter 输入两行后按 Enter 发送。预期中央输入在固定两行高度内滚动，底栏和主区不跳动。再提交一个会运行较久的请求，并在 `thinking` 或 `running` 状态按 Ctrl-C。预期底部短暂显示 interrupted toast，运行中的 tool 标记为 stopped，状态恢复 `idle`，可立即发送下一条消息。

7. 验证错误摘要：使用错误 API Key 启动 TUI 并提交一句话。预期主区只出现一条可读错误，不重复刷屏；Esc 或 Ctrl-C 可正常退出并恢复终端。

8. 验证无 TTY 回退：

   Windows PowerShell：

   ```powershell
   "只回答：headless" | cargo run --
   ```

   Linux/macOS：

   ```bash
   printf '只回答：headless\n' | cargo run --
   ```

   预期不进入 TUI，使用普通 REPL/事件输出。

9. 验证会话列表与恢复：退出 TUI 后运行：

   ```bash
   cargo run -- sessions
   cargo run -- resume -p "复述上一轮记住的数字"
   ```

   预期：第一条命令显示刚保存的会话 ID；第二条命令加载最近的 JSONL 会话并回答 `37`。

10. 验证内置搜索：在 TUI 或非 TTY REPL 中要求模型“必须用 `grep` 搜索 `Cargo.toml` 中的 `name`，再用 `glob` 查找 `src/**/*.rs`”。预期出现两个内置 tool 事件，不调用 `rg`/`fd`，并且 `.gitignore` 中排除的路径不出现在结果里。

11. 验证配置命令：输入 `/provider`，配置 Provider 的 base URL 和 API Key 后按 `f` 请求 `${base_url}/models`；预期保留已有模型及其手动 thinking/compat 设置，只按 ID 导入新模型。新增或编辑完成后按 `Ctrl-S` 保存；退出后输入 `/model`，用 Up/Down 或 j/k 选择另一个模型并按 Enter。预期配置页与模型页都替换主区、不写入对话 feed；状态栏立即更新，项目 `.zex/config.toml` 持久化 `providers` 与 `active_model`。输入 `/resume` 恢复历史会话后，当前模型保持不变。

12. 验证 `/compact` 前后上下文变化：

    1. 临时设置 `compact_keep_turns = 2`，进行至少 4 轮对话，其中早期一轮让模型读取一个较长文件。
    2. 输入 `/compact`。预期底部 toast 显示类似 `freed approximately N chars (before → after); kept 2 recent turn(s)`，主 feed 不新增配置行；其中有足够旧内容时 `N > 0`。
    3. 再询问最近两轮的信息，预期能完整回答；询问早期任务时应基于 compact 摘要回答。退出后检查会话 JSONL，可看到一条以 `[Compacted earlier conversation:` 开头的 system 消息，旧的大段 tool 输出不再完整保存。
    4. 临时把 `max_context_chars` 调低后重复长输出，预期无需输入 `/compact` 即出现自动 compact 反馈；TUI 与非 TTY REPL 行为一致。

13. 验证本次 TUI 交互：

    1. 输入 `/se`，预期输入框上方只出现 `/sessions` 与说明；Up/Down 选择，Tab 补全，Esc 关闭。
    2. 要求模型连续执行 `git status` 和 `git rev-parse --short HEAD`。预期同一时间流内出现 `bash · git status · exit 0 · duration` 和 `bash · git rev-parse --short HEAD · exit 0 · duration` 两张单行卡片；展开后才显示完整 output、参数和 timeout。
    3. 点击工具/思考卡标题展开或折叠；按 Tab 选中卡片后按 Enter / Space 激活；按 Ctrl-O 批量展开/折叠全部卡片。
    4. 输入 `/think high`，再连续按 Ctrl-T。预期状态栏 think 更新，项目 `.zex/config.toml` 写入最新偏好；每次只更新 toast，不向主 feed 追加消息。不支持推理强度的模型显示 `n/a`，不崩溃。
    5. 输入 `/thinking hide` 后提交会返回思考内容的请求。预期不显示思考卡片，但最终回答和 tool 卡片不受影响；输入 `/thinking show` 后新返回的思考内容恢复为默认折叠卡片，配置写入 `hide_thinking_block`。
    6. 分别打开 `/model`、`/resume`、`/help`、`/provider`，确认页面替换主区、选中态使用统一左侧指示，`/help` 每条命令独占一行；点击一行会选中，再点一次或按 Enter / Space 确认。点击输入框恢复输入焦点；点击状态栏 model 打开模型选择器，点击 think 循环级别。退出后时间流精确回到原滚动位置。

## 模块

- `src/provider`：Provider 抽象、OpenAI 兼容 Chat Completions/Responses 与流式解析
- `src/agent`：消息类型、事件、带最大 Provider 轮次和超时的 Agent loop
- `src/tools`：统一 Tool trait、注册表和六个纯 Rust/本地内置工具
- `src/command.rs`：TUI 与 headless REPL 共用的斜杠命令解析和执行
- `src/tui.rs`：ratatui/crossterm 可观测界面
- `src/headless.rs`：一次性任务与无 TTY 的文本界面
- `src/session.rs`：版本化 JSONL 会话保存、列表与恢复
- `src/cli.rs`：clap 命令行参数
- `src/config.rs`：全局/项目 TOML 合并与环境变量覆盖

Zex 保持 minimal core：能力通过清晰边界继续扩展，而不是提前把扩展系统耦合进核心。
