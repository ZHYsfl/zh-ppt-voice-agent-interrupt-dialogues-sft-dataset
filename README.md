# zh-ppt-voice-agent-interrupt-dialogues-sft-dataset

> **Language**: Chinese (zh)  
> **Task**: Supervised Fine-Tuning (SFT) for Voice-based Conversational Agents  
> **Domain**: PowerPoint (PPT) Creation Assistant with Real-time Interruption Handling

---

## 1. 数据集简介

本数据集用于训练一个**语音交互式 PPT 助手**（Voice Agent）。该 Agent 通过自然语音对话帮助用户完成 PPT 制作，核心能力包括：

- **需求收集**：通过多轮对话收集 PPT 的主题、风格、页数、受众等关键信息。
- **进度沟通**：在 PPT 生成过程中，作为用户与后端 PPT Agent 之间的沟通桥梁，传递反馈、汇报进度。
- **实时打断处理**：支持用户在 Agent 语音播放过程中随时打断（barge-in），Agent 能正确理解上下文并作出连贯响应。
- **工具调用**：通过结构化动作标签（`<action>`）与后端系统交互，同步执行需求更新、消息拉取、状态转发等操作。

数据集全部为**中文多轮对话**，采用 OpenAI 风格的消息格式（`system` / `user` / `assistant` / `tool`），可直接用于基于 Qwen、Llama 等模型的 SFT 训练。

---

## 2. 核心设计：为什么这个数据集"不一样"

### 2.1 不是简单的问答，而是"语音助手 + 工具调用"

传统对话数据集通常是 "用户问 → 助手答"。本数据集的 assistant 输出分为两部分：

1. **TTS 文本**：说给用户听的自然口语。
2. **`<action>` 标签**：在说话结束后，输出给后端系统的结构化指令，用于触发真实工具执行。

例如：

```
好的，请问风格偏好是什么？<action>update_requirements|topic:数学</action>
```

其中 `"好的，请问风格偏好是什么？"` 会被转成语音播放给用户，而 `<action>...` 会被后端静默执行，用户听不到。

### 2.2 不是一轮对话，而是"两回合推理架构"

本系统存在一种特殊的 **Round 2 汇报机制**：

- **Round 1**：模型输出 "TTS + action"。
- **Round 2（条件触发）**：只有当 action 是 `fetch_from_ppt_message_queue`（拉取 PPT 消息队列）时，后端会把 tool 执行结果拼入历史，然后**主动发起第二次推理**，让模型生成一段"汇报 TTS"向用户说明拉取到的内容（如"新版 PPT 已经生成完毕"）。

其他 action（如 `send_to_ppt_agent`）执行完后直接进入 `turn_end`，不会触发 Round 2。

这意味着数据集中存在**连续的 assistant → tool → assistant** 三段式结构，这是与普通多轮对话数据最显著的区别。

### 2.3 不是"说完再听"，而是"随时打断"

语音交互的核心特征是：**用户可以在 Agent 说话的任何时刻打断**。

本数据集系统性地构造了打断场景，并严格遵循以下工程约束：

- **打断只能发生在 TTS 播放阶段**：一旦 `<action>` 标签开始输出，action 序列将在后台静默执行到结束，不可能被打断。
- `**</interrupted>` 标记**：只有当用户的语音打断了正在播放的 TTS 时，下一条 `user` 消息才会以 `</interrupted>` 开头。
- **TTS 已播完、仅 action 在后台跑时用户说话**：**不算打断**，不加 `</interrupted>`。

因此，数据集中大量存在 assistant 消息被截断的场景（如 `"好的，请问风格偏好是"`），考验模型在残缺上下文下的推理能力。

### 2.4 不是"一个萝卜一个坑"，而是"延迟调用与累积"

用户可能连续多次在 TTS 阶段打断 Agent，导致 Agent 一直想调用的 action 一直被推迟。直到某一轮终于不再打断时，Agent 需要把之前累积应调用的 action 一次性补齐。这种"延迟调用"机制让数据集的行为模式比单轮调用更加复杂。

---

## 3. 数据格式详解

每条样本是一个 JSON 对象，包含一个 `messages` 数组，采用标准的 OpenAI ChatML 格式：

```json
{
  "messages": [
    { "role": "system", "content": "..." },
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." },
    { "role": "tool", "content": "..." },
    ...
  ]
}
```

### 3.1 四种 Role


| Role        | 作用     | 说明                                                  |
| ----------- | ------ | --------------------------------------------------- |
| `system`    | 阶段定义   | 由训练脚本统一注入，数据集中不手写。Phase 1 和 Phase 2 的系统提示词不同。       |
| `user`      | 用户输入   | 固定格式，包含队列状态和转录文本。                                   |
| `assistant` | 模型输出   | 只包含口语文本 + `<action>` 标签，不包含工具结果。                    |
| `tool`      | 工具执行结果 | **独立的消息**，紧跟在对应的 `assistant` 消息之后，`content` 为纯文本结果。 |


**⚠️ 注意**：`tool` 是一个独立的 `role`，**不是** assistant 内容里的 XML 包装。当前协议已完全移除 `<tool>...</tool>` 标签。

### 3.2 `user` 消息格式

```
<status>{empty | not empty}</status>
<user>{用户说的话}</user>
```

- `status` 表示**用户说完话瞬间** PPT 消息队列的状态：
  - `empty`：队列为空。
  - `not empty`：队列中有 PPT Agent 发来的消息（如新版本完成、问题、冲突等）。
- 打断场景下，在开头追加 `</interrupted>`：
  ```
  </interrupted>
  <status>{empty | not empty}</status>
  <user>{用户说的话}</user>
  ```

### 3.3 `assistant` 消息格式

两种形态：

**A. 纯对话（无 action）**

```text
好的，我明白了。
```

**B. 对话 + action 标签**

```text
好的，请问风格偏好是什么？<action>update_requirements|topic:数学</action>
```

规则：

- assistant 的**说话文本中不允许出现换行符**（`\n`）。
- Phase 1 的基本顺序是：**TTS 文本 → action(s)**。action 在所有说话文本之后输出。
- Phase 2 中，`fetch_from_ppt_message_queue` 拿到结果后，允许在同一 turn 内继续输出一段 TTS 向用户汇报结果，顺序为：**TTS 文本 → action(s) → [可选] TTS 文本**。
- 一个 turn 内允许出现**多个 action**，直接拼接：
  ```text
  <action>fetch_from_ppt_message_queue</action><action>send_to_ppt_agent|data:...</action>
  ```

### 3.4 `tool` 消息格式

`tool` 是**独立的消息**，`content` 为纯文本结果，无 XML 包装。

```json
{
  "role": "tool",
  "content": "all fields are updated"
}
```

历史顺序规则：

1. `user` 消息
2. `assistant` 消息（包含该 turn 的所有 `<action>`）
3. 按顺序排列的 `tool` 消息（每个 `<action>` 对应一个 `tool`）
4. **如果该 turn 有汇报 TTS（post-action TTS），它是一个独立的 `assistant` 消息，紧跟在所有 `tool` 之后。**

### 3.5 Action 空间


| Action                         | 格式                                              | 出现阶段        | 说明                  |
| ------------------------------ | ----------------------------------------------- | ----------- | ------------------- |
| `update_requirements`          | `update_requirements`                            | topic:...   | style:...           |
| `require_confirm`              | `<action>require_confirm</action>`              | Phase 1     | 四个字段收集完毕后，要求用户确认。   |
| `send_to_ppt_agent`            | `send_to_ppt_agent`                              | data:...   | Phase 1 & 2         |
| `fetch_from_ppt_message_queue` | `<action>fetch_from_ppt_message_queue</action>` | Phase 1 & 2 | 拉取 PPT Agent 的消息队列。 |


注意：`update_requirements` 和 `require_confirm` 在**第一次调用 `send_to_ppt_agent` 之后永久消失**。

---

## 4. 两阶段任务定义

### Phase 1：需求收集阶段

从对话开始到**第一次 `send_to_ppt_agent` 成功调用**为止。

- **目标**：收集 4 个必要字段 —— `topic`（主题）、`style`（风格）、`total_pages`（总页数）、`audience`（受众）。
- **可用 action**：`update_requirements`、`require_confirm`、`send_to_ppt_agent`。
- **status 约束**：Phase 1 期间 PPT Agent 尚未启动，所有 `user` 消息的 `status` 必须为 `empty`。
- **场景覆盖**：
  - 正常逐步收集（一问一答）。
  - 用户"抢答"（一次性提供多个字段）。
  - 用户连续打断导致 action 延迟多轮。
  - 纯闲聊（无 action，保持自然对话）。

### Phase 2：反馈与通信阶段

从**第一次 `send_to_ppt_agent` 调用之后**开始。

- **目标**：作为用户与 PPT Agent 之间的沟通桥梁。
- **可用 action**：`send_to_ppt_agent|data:...`、`fetch_from_ppt_message_queue`。
- **status 约束**：`status` 可以是 `empty` 或 `not empty`，取决于用户说完话瞬间队列状态。
- **场景覆盖**：
  - 队列为空时的自然闲聊。
  - 队列不空时拉取消息并汇报（触发 Round 2）。
  - 用户反馈转发给 PPT Agent。
  - fetch + send 同 turn 的混合动作。
  - 汇报 TTS（Round 2）被打断后的延迟汇报。
  - PPT 消息类型多样性：新版本完成、向用户提问、冲突/错误告警。

---

## 5. 典型数据示例

### 示例 1：Phase 1 正常收集（未打断）

```json
[
  { "role": "system", "content": "You are a helpful voice assistant..." },
  { "role": "user", "content": "<status>empty</status>\n<user>我想做一个关于数学的 PPT</user>" },
  { "role": "assistant", "content": "好的，请问风格偏好是什么？<action>update_requirements|topic:数学</action>" },
  { "role": "tool", "content": "we now still missing style, total_pages, audience" },
  { "role": "user", "content": "<status>empty</status>\n<user>风格要简洁优雅，15页，面向中学生</user>" },
  { "role": "assistant", "content": "好的，请确认以下需求是否正确。<action>update_requirements|style:简洁优雅|total_pages:15|audience:中学生</action><action>require_confirm</action>" },
  { "role": "tool", "content": "all fields are updated" },
  { "role": "tool", "content": "data is sent to the frontend successfully" },
  { "role": "user", "content": "<status>empty</status>\n<user>确认</user>" },
  { "role": "assistant", "content": "好的，开始为您生成 PPT。<action>send_to_ppt_agent|data:topic:数学, style:简洁优雅, total_pages:15, audience:中学生</action>" },
  { "role": "tool", "content": "data is sent to the ppt agent successfully" }
]
```

### 示例 2：TTS 阶段被打断

```json
[
  { "role": "system", "content": "You are a helpful voice assistant..." },
  { "role": "user", "content": "<status>empty</status>\n<user>我想做一个 PPT</user>" },
  { "role": "assistant", "content": "好的，请问主题是什" },
  { "role": "user", "content": "</interrupted>\n<status>empty</status>\n<user>主题是数学</user>" },
  { "role": "assistant", "content": "请问风格偏好是什么？<action>update_requirements|topic:数学</action>" },
  { "role": "tool", "content": "we now still missing style, total_pages, audience" }
]
```

注意：assistant 的消息 `"好的，请问主题是什"` 是被截断的口语文本，action 尚未输出。

### 示例 3：Phase 2 拉取队列 + Round 2 汇报

```json
[
  { "role": "system", "content": "You are a helpful voice assistant..." },
  { "role": "user", "content": "<status>not empty</status>\n<user>PPT 做得怎么样了</user>" },
  { "role": "assistant", "content": "我去帮您看看进度。<action>fetch_from_ppt_message_queue</action>" },
  { "role": "tool", "content": "the ppt message is: the new version of the ppt is generated successfully" },
  { "role": "assistant", "content": "新版 PPT 已经生成完毕，您需要我展示出来吗？" }
]
```

注意：最后一条 assistant 是 **Round 2 汇报**，它前面是 `tool` 消息，且**不含任何 action**。

### 示例 4：Round 2 汇报被打断 → 延迟汇报

```json
[
  { "role": "user", "content": "<status>not empty</status>\n<user>PPT 怎么样了</user>" },
  { "role": "assistant", "content": "我去帮您看看进度。<action>fetch_from_ppt_message_queue</action>" },
  { "role": "tool", "content": "the ppt message is: the new version of the ppt is generated successfully" },
  { "role": "assistant", "content": "新版 PPT 已经生" },
  { "role": "user", "content": "</interrupted>\n<status>empty</status>\n<user>直接发我邮箱</user>" },
  { "role": "assistant", "content": "好的，新版 PPT 已经生成完毕，我这就帮您发送。<action>send_to_ppt_agent|data:用户要求发到邮箱</action>" },
  { "role": "tool", "content": "data is sent to the ppt agent successfully" }
]
```

注意：Round 2 的 `"新版 PPT 已经生"` 被打断后，下一轮 assistant 重新说出了完整的汇报语句并执行了新的 action。

---

## 6. 数据集使用建议

### 6.1 基础模型

本数据集基于 **Qwen/Qwen3-4B-Instruct-2507** 构建，兼容 qwen3 系列模型。

### 6.2 训练配置参考（单卡 RTX 4090 24GB）

- **方法**：QLoRA SFT（4-bit 量化底座 + LoRA 适配器）
- **基础模型**：`Qwen/Qwen3-4B-Instruct-2507`
- **LoRA 参数**：`r=8`, `alpha=16`, `dropout=0`
- **目标模块**：`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- **最大序列长度**：1536
- **数据类型**：`bfloat16`
- **批次**：`per_device_train_batch_size=1`, `gradient_accumulation_steps=8`（有效 batch = 8）
- **学习率**：`2e-4`
- **调度器**：`cosine`，`warmup_ratio=0.1`
- **正则**：`weight_decay=0.01`, `max_grad_norm=0.3`
- **训练轮数**：3 epochs
- **packing**：`False`（长样本不打包，避免 OOM）
- **显存优化建议**：训练前设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 以降低显存碎片导致的偶发 OOM

### 6.3 系统提示词

**Phase 1 系统提示词**：强调需求收集、4 个字段、`update_requirements` / `require_confirm` / `send_to_ppt_agent` 的动作规则，以及打断处理。

**Phase 2 系统提示词**：强调沟通桥梁角色、`fetch_from_ppt_message_queue` 的 Round 2 汇报机制、`send_to_ppt_agent` 的反馈转发，以及 `update_requirements` / `require_confirm` 已失效。

为了保持最佳效果，推理脚本和训练数据集里的system prompt最好保持一致。

---

## 7. 相关资源

- **项目仓库**：[https://github.com/ZHYsfl/EducationAgent](https://github.com/ZHYsfl/EducationAgent)
- **该数据集基于Qwen/Qwen3-4B-Instruct-2507进行SFT训练后的lora权重：**[https://huggingface.co/ZaneSFL/zh-ppt-voice-agent-model-lora-support-interrupt](https://huggingface.co/ZaneSFL/zh-ppt-voice-agent-model-lora-support-interrupt)
- **本数据集的Hugging Face地址**：https://huggingface.co/datasets/ZaneSFL/zh-ppt-voice-agent-interrupt-dialogues

## 8. 许可与引用

本项目数据集遵循 Apache 2.0 许可证开源。

若本数据集对您的研究有帮助，欢迎引用：

```bibtex
@dataset{zh-ppt-voice-agent-sft-2026,
  title={Chinese PPT Voice Agent Interrupt Dialogues SFT Dataset},
  author={Zane},
  year={2026},
  url={https://huggingface.co/datasets/ZaneSFL/zh-ppt-voice-agent-interrupt-dialogues}
}
```
