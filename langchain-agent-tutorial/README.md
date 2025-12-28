# 🤖 LangChain + LangGraph 智能体教程

使用 **阿里千问 (Qwen)** 模型，从零开始学习构建 AI 智能体。

## 📚 教程目录

| 课程 | 内容 | Notebook |
|------|------|----------|
| 第一课 | 连接阿里千问模型 | `01_连接千问模型.ipynb` |
| 第二课 | Prompt Template 和 Chain | `02_Prompt和Chain.ipynb` |
| 第三课 | 添加工具 (Tools) 能力 | `03_工具Tools.ipynb` |
| 第四课 | 使用 LangGraph 构建智能体 | `04_LangGraph智能体.ipynb` |
| 第五课 | 带记忆的多轮对话智能体 | `05_带记忆的智能体.ipynb` |

## 🚀 快速开始

### 1. 安装依赖

```bash
cd langchain-agent-tutorial
pip install -r requirements.txt
```

### 2. 配置 API Key

获取 API Key: https://dashscope.console.aliyun.com/apiKey

### 3. 启动 JupyterLab

```bash
# 启动 JupyterLab
jupyter lab

# 或者使用 Jupyter Notebook
jupyter notebook
```

### 4. 开始学习

1. 打开 `01_连接千问模型.ipynb`
2. 在第一个代码单元格中设置你的 API Key
3. 按顺序运行每个单元格，学习每个概念
4. 完成练习，巩固所学知识

---

## 📖 详细教程

### 第一课：连接阿里千问模型

学习如何使用 LangChain 连接阿里千问模型。

**核心代码：**

```python
from langchain_openai import ChatOpenAI

# 千问支持 OpenAI 兼容接口
chat = ChatOpenAI(
    model="qwen-plus",
    openai_api_key="your_api_key",
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 发送消息
response = chat.invoke("你好！")
print(response.content)
```

**关键概念：**
- `ChatOpenAI`: LangChain 提供的 OpenAI 兼容客户端
- `openai_api_base`: 指向千问的 API 地址
- 支持的模型: `qwen-turbo`, `qwen-plus`, `qwen-max`

---

### 第二课：Prompt Template 和 Chain

学习如何创建可复用的提示词模板和处理链。

**Prompt Template 示例：**

```python
from langchain_core.prompts import ChatPromptTemplate

# 创建模板
prompt = ChatPromptTemplate.from_template(
    "将以下内容翻译成{language}：{text}"
)

# 使用模板
messages = prompt.format_messages(language="英文", text="你好世界")
```

**LCEL Chain 示例：**

```python
from langchain_core.output_parsers import StrOutputParser

# 使用 | 操作符连接组件
chain = prompt | model | StrOutputParser()

# 调用链
result = chain.invoke({"language": "英文", "text": "你好"})
```

**关键概念：**
- `ChatPromptTemplate`: 聊天提示词模板
- `LCEL (|)`: LangChain Expression Language，使用管道连接组件
- `StrOutputParser`: 将 AI 回复转换为字符串

---

### 第三课：添加工具 (Tools) 能力

学习如何让 AI 能够调用工具获取信息或执行操作。

**创建工具：**

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """获取城市天气信息。"""
    # 实现获取天气的逻辑
    return f"{city}：晴，25°C"
```

**绑定工具到模型：**

```python
tools = [get_weather, calculate]
model_with_tools = model.bind_tools(tools)
```

**处理工具调用：**

```python
response = model_with_tools.invoke("北京天气怎么样？")

if response.tool_calls:
    for call in response.tool_calls:
        # 执行工具
        result = tools[call["name"]].invoke(call["args"])
        # 将结果返回给 AI
```

**关键概念：**
- `@tool` 装饰器：将函数转换为工具
- `bind_tools`: 将工具绑定到模型
- `tool_calls`: AI 返回的工具调用请求
- `ToolMessage`: 工具执行结果消息

---

### 第四课：使用 LangGraph 构建智能体

学习使用 LangGraph 构建有状态的智能体。

**智能体图结构：**

```
START → agent → [判断] → tools → agent → ...
                  ↓
                 END
```

**定义状态：**

```python
from typing import TypedDict, Annotated, Sequence
import operator

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
```

**构建图：**

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))

# 设置入口
workflow.set_entry_point("agent")

# 添加边
workflow.add_conditional_edges("agent", should_continue, 
    {"tools": "tools", "end": END})
workflow.add_edge("tools", "agent")

# 编译
agent = workflow.compile()
```

**使用预构建 Agent：**

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(model, tools)
result = agent.invoke({"messages": [HumanMessage(content="你好")]})
```

**关键概念：**
- `StateGraph`: 状态图，定义工作流
- `Node`: 节点，执行具体逻辑
- `Edge`: 边，定义节点间连接
- `Conditional Edge`: 条件边，根据状态决定走向
- `ToolNode`: 预构建的工具执行节点

---

### 第五课：带记忆的多轮对话智能体

学习如何让智能体记住对话历史。

**添加记忆：**

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
agent = workflow.compile(checkpointer=memory)
```

**使用 thread_id 区分会话：**

```python
config = {"configurable": {"thread_id": "user_123"}}

# 第一轮
agent.invoke({"messages": [HumanMessage("我叫小明")]}, config=config)

# 第二轮 - 智能体记得用户叫小明
agent.invoke({"messages": [HumanMessage("我叫什么？")]}, config=config)
```

**查看对话历史：**

```python
state = agent.get_state(config)
messages = state.values["messages"]
```

**关键概念：**
- `MemorySaver`: 内存检查点存储
- `thread_id`: 会话标识，相同 ID 共享历史
- `get_state`: 获取当前会话状态

---

## 🏗️ 架构图

```
┌─────────────────────────────────────────────────────────┐
│                      智能体 (Agent)                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   ┌─────────┐    ┌─────────┐    ┌─────────────────┐    │
│   │  用户   │───▶│   LLM   │───▶│  工具调用决策   │    │
│   │  输入   │    │ (千问)  │    │                 │    │
│   └─────────┘    └─────────┘    └────────┬────────┘    │
│                                          │              │
│                       ┌──────────────────┼──────┐      │
│                       ▼                  ▼      ▼      │
│                 ┌─────────┐        ┌─────────┐ ...     │
│                 │  工具1  │        │  工具2  │         │
│                 │ (天气)  │        │ (计算)  │         │
│                 └────┬────┘        └────┬────┘         │
│                      │                  │              │
│                      └────────┬─────────┘              │
│                               ▼                        │
│                        ┌─────────┐                     │
│                        │ 最终回复 │                     │
│                        └─────────┘                     │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                    记忆 (Memory)                        │
│   ┌─────────────────────────────────────────────────┐  │
│   │  thread_001: [msg1, msg2, msg3, ...]            │  │
│   │  thread_002: [msg1, msg2, ...]                  │  │
│   └─────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 💡 最佳实践

### 1. 工具设计
- 工具函数的 docstring 要清晰描述功能
- 参数名要有意义，方便 AI 理解
- 返回值要包含足够信息

### 2. 提示词工程
- 使用系统消息设定 AI 的角色和行为
- 明确列出可用的工具
- 给出清晰的任务指令

### 3. 错误处理
- 工具函数中要处理异常
- 提供有用的错误信息
- 设置合理的超时时间

### 4. 记忆管理
- 生产环境使用持久化存储 (SQLite/PostgreSQL)
- 定期清理过长的对话历史
- 考虑对话历史的摘要压缩

---

## 📦 依赖说明

| 包名 | 说明 |
|------|------|
| `langchain` | LangChain 核心库 |
| `langchain-openai` | OpenAI 兼容接口支持 |
| `langgraph` | 构建智能体工作流 |
| `dashscope` | 阿里云 DashScope SDK |
| `python-dotenv` | 环境变量管理 |

---

## 🔗 相关资源

- [LangChain 官方文档](https://python.langchain.com/)
- [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/)
- [阿里云 DashScope](https://dashscope.console.aliyun.com/)
- [千问模型介绍](https://help.aliyun.com/zh/dashscope/developer-reference/model-introduction)

---

## 📝 许可证

MIT License
