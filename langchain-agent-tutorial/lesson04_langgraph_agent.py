"""
第四课：使用 LangGraph 构建智能体

本课学习目标：
1. 理解 LangGraph 的核心概念
2. 理解智能体的工作原理
3. 构建一个能够自主决策的智能体

什么是 LangGraph？
- LangGraph 是一个用于构建有状态、多步骤 AI 应用的库
- 它使用图（Graph）来表示工作流
- 非常适合构建需要循环、条件判断的复杂 AI 应用

什么是智能体（Agent）？
- 智能体 = 大语言模型 + 工具 + 自主决策能力
- 它可以根据任务自主选择使用哪些工具
- 可以在多个步骤中持续工作，直到完成任务
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
from langgraph.prebuilt import ToolNode

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
        temperature=0,
    )


# ============================================
# 1. 定义工具
# ============================================

@tool
def get_current_time() -> str:
    """获取当前的日期和时间。"""
    return datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")


@tool
def calculate(expression: str) -> str:
    """计算数学表达式，支持加减乘除和幂运算。"""
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
        "北京": {"temp": 15, "condition": "晴", "humidity": 45},
        "上海": {"temp": 18, "condition": "多云", "humidity": 60},
        "广州": {"temp": 25, "condition": "小雨", "humidity": 80},
        "深圳": {"temp": 26, "condition": "阴", "humidity": 75},
    }
    
    if city in weather_data:
        data = weather_data[city]
        return f"{city}天气：{data['condition']}，温度 {data['temp']}°C，湿度 {data['humidity']}%"
    return f"暂无 {city} 的天气数据"


@tool  
def search_web(query: str) -> str:
    """搜索网络获取信息（模拟）。"""
    # 模拟搜索结果
    results = {
        "python": "Python 是最流行的编程语言之一，特别适合 AI 和数据科学。",
        "langchain": "LangChain 是构建 LLM 应用的流行框架，提供丰富的工具和抽象。",
        "langgraph": "LangGraph 用于构建有状态的多步骤 AI 工作流和智能体。",
    }
    
    for key, value in results.items():
        if key.lower() in query.lower():
            return value
    return f"搜索 '{query}' 未找到相关结果。"


# ============================================
# 2. 定义状态 (State)
# ============================================

class AgentState(TypedDict):
    """
    智能体状态
    
    状态是 LangGraph 的核心概念：
    - 状态在图的各个节点之间传递
    - 每个节点可以读取和修改状态
    - messages 使用 Annotated 和 operator.add 实现追加而非替换
    """
    # 消息列表，使用 operator.add 表示新消息追加到列表末尾
    messages: Annotated[Sequence[HumanMessage | AIMessage | ToolMessage], operator.add]


# ============================================
# 3. 构建智能体图
# ============================================

def build_basic_agent():
    """
    构建一个基础智能体
    
    图的结构：
    
    START → agent → should_continue? → tools → agent → ...
                         ↓
                        END
    
    - agent 节点：调用 LLM 进行推理
    - tools 节点：执行工具
    - should_continue：条件边，决定是否继续
    """
    print("=" * 50)
    print("构建基础智能体")
    print("=" * 50)
    
    # 准备模型和工具
    model = create_qwen_chat()
    tools = [get_current_time, calculate, get_weather, search_web]
    model_with_tools = model.bind_tools(tools)
    
    # 系统提示词
    system_message = SystemMessage(content="""你是一个智能助手，可以使用工具来帮助用户解决问题。

你可以使用的工具：
- get_current_time: 获取当前时间
- calculate: 进行数学计算
- get_weather: 查询天气
- search_web: 搜索网络

请根据用户的问题，决定是否需要使用工具。如果需要，请调用相应的工具。
回答要准确、简洁。""")
    
    # 定义 agent 节点
    def agent_node(state: AgentState) -> dict:
        """
        智能体节点：调用 LLM 进行推理
        
        输入：当前状态
        输出：新的消息（追加到 messages）
        """
        # 在消息列表开头添加系统消息
        messages = [system_message] + list(state["messages"])
        
        # 调用模型
        response = model_with_tools.invoke(messages)
        
        # 返回新消息（会追加到状态的 messages 中）
        return {"messages": [response]}
    
    # 定义条件函数：决定下一步去哪里
    def should_continue(state: AgentState) -> str:
        """
        条件函数：检查是否需要继续执行工具
        
        返回值：
        - "tools": 需要执行工具
        - "end": 结束
        """
        last_message = state["messages"][-1]
        
        # 如果 AI 请求调用工具，则继续到 tools 节点
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        
        # 否则结束
        return "end"
    
    # 创建工具节点
    tool_node = ToolNode(tools)
    
    # 构建图
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    
    # 设置入口点
    workflow.set_entry_point("agent")
    
    # 添加条件边
    workflow.add_conditional_edges(
        "agent",  # 从 agent 节点出发
        should_continue,  # 使用 should_continue 函数决定下一步
        {
            "tools": "tools",  # 如果返回 "tools"，去 tools 节点
            "end": END,  # 如果返回 "end"，结束
        }
    )
    
    # tools 执行完后，回到 agent
    workflow.add_edge("tools", "agent")
    
    # 编译图
    agent = workflow.compile()
    
    print("\n✅ 智能体构建完成！")
    print("\n图的结构：")
    print("  START → agent → [条件判断]")
    print("                     ├→ tools → agent (循环)")
    print("                     └→ END")
    
    return agent


# ============================================
# 4. 运行智能体
# ============================================

def run_agent_example():
    """
    运行智能体示例
    """
    print("\n" + "=" * 50)
    print("运行智能体示例")
    print("=" * 50)
    
    # 构建智能体
    agent = build_basic_agent()
    
    # 测试问题
    test_queries = [
        "你好！",
        "现在几点了？",
        "帮我计算 123 * 456 + 789",
        "北京和上海今天哪个城市更热？",
        "先告诉我现在的时间，然后查一下北京的天气",
    ]
    
    for query in test_queries:
        print(f"\n{'='*40}")
        print(f"用户: {query}")
        print(f"{'='*40}")
        
        # 运行智能体
        result = agent.invoke({
            "messages": [HumanMessage(content=query)]
        })
        
        # 打印执行过程
        print("\n执行过程：")
        for i, msg in enumerate(result["messages"]):
            if isinstance(msg, HumanMessage):
                print(f"  [{i}] 用户: {msg.content}")
            elif isinstance(msg, AIMessage):
                if msg.tool_calls:
                    print(f"  [{i}] AI 调用工具: {[tc['name'] for tc in msg.tool_calls]}")
                else:
                    print(f"  [{i}] AI 回复: {msg.content[:100]}...")
            elif isinstance(msg, ToolMessage):
                print(f"  [{i}] 工具结果: {msg.content}")
        
        # 最终回答
        final_answer = result["messages"][-1].content
        print(f"\n最终回答: {final_answer}")


# ============================================
# 5. 流式输出
# ============================================

def streaming_agent_example():
    """
    流式运行智能体
    
    使用 stream 方法可以实时看到智能体的执行过程
    """
    print("\n" + "=" * 50)
    print("流式输出示例")
    print("=" * 50)
    
    agent = build_basic_agent()
    
    query = "帮我查一下北京天气，然后告诉我现在的时间"
    print(f"\n用户: {query}")
    print("\n执行过程 (流式)：")
    
    # 使用 stream 方法
    for event in agent.stream({"messages": [HumanMessage(content=query)]}):
        # event 是一个字典，键是节点名，值是该节点的输出
        for node_name, node_output in event.items():
            print(f"\n--- {node_name} 节点 ---")
            if "messages" in node_output:
                for msg in node_output["messages"]:
                    if isinstance(msg, AIMessage):
                        if msg.tool_calls:
                            print(f"  AI 调用: {[tc['name'] for tc in msg.tool_calls]}")
                        elif msg.content:
                            print(f"  AI: {msg.content}")
                    elif isinstance(msg, ToolMessage):
                        print(f"  工具结果: {msg.content}")


# ============================================
# 6. 使用 LangGraph 预构建的 Agent
# ============================================

def prebuilt_agent_example():
    """
    使用 LangGraph 预构建的 ReAct Agent
    
    LangGraph 提供了预构建的智能体，可以快速创建
    """
    from langgraph.prebuilt import create_react_agent
    
    print("\n" + "=" * 50)
    print("使用预构建的 ReAct Agent")
    print("=" * 50)
    
    model = create_qwen_chat()
    tools = [get_current_time, calculate, get_weather, search_web]
    
    # 使用预构建的 ReAct Agent
    # ReAct = Reasoning + Acting，推理与行动结合
    agent = create_react_agent(
        model=model,
        tools=tools,
        state_modifier="你是一个智能助手，可以使用工具帮助用户。回答要简洁准确。"
    )
    
    print("\n✅ ReAct Agent 创建完成！")
    
    # 测试
    query = "计算一下 2 的 10 次方是多少"
    print(f"\n用户: {query}")
    
    result = agent.invoke({"messages": [HumanMessage(content=query)]})
    
    print(f"\n回答: {result['messages'][-1].content}")
    
    return agent


# ============================================
# 7. 可视化智能体图（可选）
# ============================================

def visualize_agent():
    """
    可视化智能体的图结构
    
    需要安装: pip install pygraphviz
    """
    print("\n" + "=" * 50)
    print("智能体图结构")
    print("=" * 50)
    
    agent = build_basic_agent()
    
    # 获取图的 ASCII 表示
    try:
        print("\n图的 Mermaid 表示：")
        print(agent.get_graph().draw_mermaid())
    except Exception as e:
        print(f"无法生成图的可视化: {e}")
        print("\n图的结构描述：")
        print("""
        ┌─────────┐
        │  START  │
        └────┬────┘
             │
             ▼
        ┌─────────┐
        │  agent  │◄───────┐
        └────┬────┘        │
             │             │
             ▼             │
        ┌─────────┐        │
        │ 条件判断 │        │
        └────┬────┘        │
             │             │
        ┌────┴────┐        │
        │         │        │
        ▼         ▼        │
    ┌──────┐  ┌──────┐     │
    │ END  │  │tools │─────┘
    └──────┘  └──────┘
        """)


# ============================================
# 运行示例
# ============================================

if __name__ == "__main__":
    print("\n🚀 第四课：使用 LangGraph 构建智能体\n")
    
    try:
        # 1. 运行智能体示例
        run_agent_example()
        
        # 2. 流式输出示例
        streaming_agent_example()
        
        # 3. 预构建 Agent
        prebuilt_agent_example()
        
        # 4. 可视化
        visualize_agent()
        
        print("\n" + "=" * 50)
        print("✅ 第四课完成！")
        print("=" * 50)
        print("\n📚 关键概念回顾：")
        print("1. State: 状态是在图节点间传递的数据")
        print("2. Node: 节点是执行具体逻辑的地方")
        print("3. Edge: 边定义了节点之间的连接")
        print("4. Conditional Edge: 条件边根据状态决定下一步")
        print("5. ToolNode: 预构建的工具执行节点")
        print("6. ReAct Agent: 推理与行动结合的智能体模式")
        print("\n📚 下一课我们将添加记忆功能，实现多轮对话！")
        
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
