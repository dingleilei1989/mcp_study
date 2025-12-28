"""
第五课：带记忆的多轮对话智能体

本课学习目标：
1. 理解 Checkpointer（检查点）机制
2. 实现多轮对话记忆
3. 构建一个完整的、可以记住上下文的智能体

为什么需要记忆？
- 之前的智能体每次调用都是独立的，不记得之前的对话
- 有了记忆，智能体可以：
  - 记住用户说过的话
  - 理解上下文，给出更好的回答
  - 跨多轮对话完成复杂任务
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from typing import TypedDict, Annotated, Sequence
import operator
from datetime import datetime

# LangGraph 核心组件
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()


def create_qwen_chat():
    """创建千问聊天模型"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量")
    
    return ChatOpenAI(
        model="qwen-plus",
        openai_api_key=api_key,
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.7,
    )


# ============================================
# 定义工具
# ============================================

@tool
def get_current_time() -> str:
    """获取当前的日期和时间。"""
    return datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")


@tool
def calculate(expression: str) -> str:
    """计算数学表达式。"""
    try:
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return "错误：表达式包含不允许的字符"
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误：{str(e)}"


@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息。"""
    weather_data = {
        "北京": {"temp": 15, "condition": "晴"},
        "上海": {"temp": 18, "condition": "多云"},
        "广州": {"temp": 25, "condition": "小雨"},
        "深圳": {"temp": 26, "condition": "阴"},
    }
    
    if city in weather_data:
        data = weather_data[city]
        return f"{city}：{data['condition']}，{data['temp']}°C"
    return f"暂无 {city} 的天气数据"


@tool
def remember_note(note: str) -> str:
    """
    记住用户的笔记或重要信息。
    
    Args:
        note: 要记住的内容
    """
    return f"已记住：{note}"


# ============================================
# 1. 理解 Checkpointer（检查点）
# ============================================

def understand_checkpointer():
    """
    理解 Checkpointer 的作用
    
    Checkpointer 是 LangGraph 的核心概念：
    - 它保存图执行过程中的状态快照
    - 每个状态快照用一个唯一的 thread_id 标识
    - 下次调用时，传入相同的 thread_id 就能恢复之前的状态
    
    常用的 Checkpointer：
    - MemorySaver: 保存在内存中（重启后丢失）
    - SqliteSaver: 保存在 SQLite 数据库
    - PostgresSaver: 保存在 PostgreSQL 数据库
    """
    print("=" * 50)
    print("1. 理解 Checkpointer（检查点）")
    print("=" * 50)
    
    print("""
Checkpointer 的工作原理：

    对话 1 (thread_id="user_123")        对话 2 (thread_id="user_456")
    ┌─────────────────────┐              ┌─────────────────────┐
    │ 用户: 我叫小明       │              │ 用户: 你好           │
    │ AI: 你好小明！       │              │ AI: 你好！           │
    │ ─────────────────   │              │ ─────────────────   │
    │ 用户: 我叫什么？     │              │ 用户: 我叫什么？     │
    │ AI: 你叫小明。       │              │ AI: 你还没告诉我     │
    └─────────────────────┘              └─────────────────────┘

每个 thread_id 有独立的对话历史！
    """)
    
    print("\n常用的 Checkpointer 类型：")
    print("1. MemorySaver - 内存存储，简单快速，重启后丢失")
    print("2. SqliteSaver - SQLite 存储，持久化到本地文件")
    print("3. PostgresSaver - PostgreSQL 存储，适合生产环境")


# ============================================
# 2. 构建带记忆的智能体
# ============================================

class AgentState(TypedDict):
    """智能体状态"""
    messages: Annotated[Sequence[HumanMessage | AIMessage | ToolMessage], operator.add]


def build_memory_agent():
    """
    构建带记忆的智能体
    """
    print("\n" + "=" * 50)
    print("2. 构建带记忆的智能体")
    print("=" * 50)
    
    model = create_qwen_chat()
    tools = [get_current_time, calculate, get_weather, remember_note]
    model_with_tools = model.bind_tools(tools)
    
    system_message = SystemMessage(content="""你是一个友好的智能助手，具有以下特点：
1. 你会记住用户在对话中提到的信息
2. 你可以使用工具来帮助用户
3. 你的回答简洁友好

可用工具：
- get_current_time: 获取当前时间
- calculate: 数学计算
- get_weather: 查询天气
- remember_note: 记住重要信息

请根据对话上下文，给出最合适的回答。""")
    
    def agent_node(state: AgentState) -> dict:
        messages = [system_message] + list(state["messages"])
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def should_continue(state: AgentState) -> str:
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "end"
    
    tool_node = ToolNode(tools)
    
    # 构建图
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    workflow.add_edge("tools", "agent")
    
    # 🔑 关键：添加 MemorySaver 作为检查点
    memory = MemorySaver()
    agent = workflow.compile(checkpointer=memory)
    
    print("\n✅ 带记忆的智能体构建完成！")
    
    return agent


# ============================================
# 3. 多轮对话演示
# ============================================

def multi_turn_conversation():
    """
    多轮对话演示
    
    关键点：
    - 使用相同的 thread_id 保持对话连续性
    - 不同的 thread_id 是独立的对话
    """
    print("\n" + "=" * 50)
    print("3. 多轮对话演示")
    print("=" * 50)
    
    agent = build_memory_agent()
    
    # 配置：使用 thread_id 标识对话
    config = {"configurable": {"thread_id": "conversation_001"}}
    
    # 模拟多轮对话
    conversation = [
        "你好！我叫张三，今年 25 岁。",
        "我喜欢编程和阅读。",
        "你还记得我叫什么名字吗？",
        "我的年龄和爱好呢？",
        "帮我查一下北京的天气。",
        "如果我去北京旅行，应该注意什么？",
    ]
    
    print("\n开始对话（同一个 thread_id）：")
    print("-" * 40)
    
    for user_input in conversation:
        print(f"\n👤 用户: {user_input}")
        
        # 调用智能体，传入配置
        result = agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config  # 🔑 关键：传入配置以使用记忆
        )
        
        # 获取最后的 AI 回复
        ai_response = result["messages"][-1].content
        print(f"🤖 AI: {ai_response}")
    
    # 演示不同 thread_id 的独立性
    print("\n" + "=" * 40)
    print("新的对话（不同的 thread_id）：")
    print("-" * 40)
    
    new_config = {"configurable": {"thread_id": "conversation_002"}}
    
    result = agent.invoke(
        {"messages": [HumanMessage(content="你知道我叫什么名字吗？")]},
        config=new_config
    )
    
    print(f"\n👤 用户: 你知道我叫什么名字吗？")
    print(f"🤖 AI: {result['messages'][-1].content}")


# ============================================
# 4. 查看对话历史
# ============================================

def view_conversation_history():
    """
    查看保存的对话历史
    """
    print("\n" + "=" * 50)
    print("4. 查看对话历史")
    print("=" * 50)
    
    agent = build_memory_agent()
    config = {"configurable": {"thread_id": "history_demo"}}
    
    # 进行一些对话
    messages_to_send = [
        "记住：我的生日是 3 月 15 日",
        "我最喜欢的颜色是蓝色",
        "现在几点了？",
    ]
    
    for msg in messages_to_send:
        agent.invoke({"messages": [HumanMessage(content=msg)]}, config=config)
    
    # 获取状态快照
    state = agent.get_state(config)
    
    print("\n当前对话状态：")
    print(f"消息数量: {len(state.values['messages'])}")
    print("\n完整对话历史：")
    
    for i, msg in enumerate(state.values["messages"]):
        if isinstance(msg, HumanMessage):
            print(f"  [{i}] 👤 用户: {msg.content}")
        elif isinstance(msg, AIMessage):
            if msg.content:
                print(f"  [{i}] 🤖 AI: {msg.content[:80]}...")
            if msg.tool_calls:
                print(f"  [{i}] 🔧 工具调用: {[tc['name'] for tc in msg.tool_calls]}")
        elif isinstance(msg, ToolMessage):
            print(f"  [{i}] 📋 工具结果: {msg.content}")


# ============================================
# 5. 使用预构建的 ReAct Agent + 记忆
# ============================================

def prebuilt_agent_with_memory():
    """
    使用预构建的 ReAct Agent 并添加记忆
    """
    print("\n" + "=" * 50)
    print("5. 预构建 ReAct Agent + 记忆")
    print("=" * 50)
    
    model = create_qwen_chat()
    tools = [get_current_time, calculate, get_weather]
    memory = MemorySaver()
    
    # 使用预构建的 ReAct Agent
    agent = create_react_agent(
        model=model,
        tools=tools,
        checkpointer=memory,  # 添加记忆
        state_modifier="你是一个友好的助手，会记住用户的信息。回答简洁。"
    )
    
    print("\n✅ 预构建 Agent + 记忆 创建完成！")
    
    # 测试
    config = {"configurable": {"thread_id": "prebuilt_001"}}
    
    test_messages = [
        "我住在深圳，帮我查下天气",
        "比北京冷还是热？",
    ]
    
    for msg in test_messages:
        print(f"\n👤 用户: {msg}")
        result = agent.invoke({"messages": [HumanMessage(content=msg)]}, config=config)
        print(f"🤖 AI: {result['messages'][-1].content}")
    
    return agent


# ============================================
# 6. 交互式对话（完整示例）
# ============================================

def interactive_chat():
    """
    创建一个可以交互式聊天的智能体
    
    这是一个完整的智能体示例，可以：
    - 记住对话历史
    - 使用工具
    - 多轮对话
    """
    print("\n" + "=" * 50)
    print("6. 交互式对话示例")
    print("=" * 50)
    
    model = create_qwen_chat()
    tools = [get_current_time, calculate, get_weather, remember_note]
    memory = MemorySaver()
    
    agent = create_react_agent(
        model=model,
        tools=tools,
        checkpointer=memory,
        state_modifier="""你是一个智能助手，名叫"小智"。

你的特点：
1. 友好、热情、有耐心
2. 会记住用户在对话中分享的信息
3. 可以使用工具获取时间、天气、进行计算
4. 回答简洁但有帮助

开始对话时，先友好地打招呼，询问用户想要什么帮助。"""
    )
    
    config = {"configurable": {"thread_id": "interactive_session"}}
    
    print("\n模拟交互式对话：")
    print("(输入 'quit' 退出)")
    print("-" * 40)
    
    # 模拟用户输入
    simulated_inputs = [
        "你好",
        "我叫小王，在北京工作",
        "今天天气怎么样？",
        "帮我算一下 1500 * 12 等于多少",
        "你还记得我在哪里工作吗？",
    ]
    
    for user_input in simulated_inputs:
        print(f"\n👤 用户: {user_input}")
        
        result = agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config
        )
        
        response = result["messages"][-1].content
        print(f"🤖 小智: {response}")
    
    print("\n[对话结束]")


# ============================================
# 7. 完整的智能体类封装
# ============================================

class SmartAssistant:
    """
    封装好的智能助手类
    
    这是一个可以直接使用的智能体封装，包含：
    - 多轮对话记忆
    - 工具调用
    - 简洁的 API
    """
    
    def __init__(self, name: str = "小智"):
        self.name = name
        self.model = create_qwen_chat()
        self.tools = [get_current_time, calculate, get_weather, remember_note]
        self.memory = MemorySaver()
        
        self.agent = create_react_agent(
            model=self.model,
            tools=self.tools,
            checkpointer=self.memory,
            state_modifier=f"""你是一个智能助手，名叫"{name}"。
友好、热情、简洁地回答用户问题。
会记住用户分享的信息，可以使用工具帮助用户。"""
        )
    
    def chat(self, message: str, session_id: str = "default") -> str:
        """
        发送消息并获取回复
        
        Args:
            message: 用户消息
            session_id: 会话 ID，相同 ID 共享对话历史
        
        Returns:
            AI 的回复
        """
        config = {"configurable": {"thread_id": session_id}}
        result = self.agent.invoke(
            {"messages": [HumanMessage(content=message)]},
            config=config
        )
        return result["messages"][-1].content
    
    def get_history(self, session_id: str = "default") -> list:
        """获取对话历史"""
        config = {"configurable": {"thread_id": session_id}}
        state = self.agent.get_state(config)
        return state.values.get("messages", [])
    
    def clear_history(self, session_id: str = "default"):
        """清除对话历史（通过使用新的 session_id）"""
        print(f"提示：要清除历史，请使用新的 session_id")


def demo_smart_assistant():
    """演示封装好的智能助手"""
    print("\n" + "=" * 50)
    print("7. 使用封装好的 SmartAssistant 类")
    print("=" * 50)
    
    # 创建助手实例
    assistant = SmartAssistant(name="小助")
    
    print("\n与小助对话：")
    print("-" * 40)
    
    # 对话
    messages = [
        "你好！我叫李华",
        "我想学习 Python 编程",
        "现在几点了？",
        "你还记得我想学什么吗？",
    ]
    
    for msg in messages:
        print(f"\n👤 用户: {msg}")
        response = assistant.chat(msg, session_id="user_lihua")
        print(f"🤖 小助: {response}")
    
    # 查看历史
    print(f"\n对话历史共有 {len(assistant.get_history('user_lihua'))} 条消息")


# ============================================
# 运行示例
# ============================================

if __name__ == "__main__":
    print("\n🚀 第五课：带记忆的多轮对话智能体\n")
    
    try:
        # 1. 理解 Checkpointer
        understand_checkpointer()
        
        # 2. 构建带记忆的智能体（在后面的示例中展示）
        
        # 3. 多轮对话演示
        multi_turn_conversation()
        
        # 4. 查看对话历史
        view_conversation_history()
        
        # 5. 预构建 Agent + 记忆
        prebuilt_agent_with_memory()
        
        # 6. 交互式对话
        interactive_chat()
        
        # 7. 封装好的智能助手
        demo_smart_assistant()
        
        print("\n" + "=" * 50)
        print("✅ 第五课完成！")
        print("=" * 50)
        print("\n📚 关键概念回顾：")
        print("1. Checkpointer: 保存图执行状态的机制")
        print("2. thread_id: 标识不同对话会话的唯一 ID")
        print("3. MemorySaver: 内存存储，适合开发和测试")
        print("4. get_state: 获取当前对话的完整状态")
        print("5. 记忆让智能体能够进行有上下文的多轮对话")
        print("\n🎉 恭喜！你已经学会了如何构建一个完整的智能体！")
        
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
