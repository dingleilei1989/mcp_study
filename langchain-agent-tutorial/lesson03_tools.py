"""
第三课：添加工具 (Tools) 能力

本课学习目标：
1. 了解什么是 Tool（工具）
2. 学会创建自定义工具
3. 让 AI 能够调用工具

工具是智能体的核心能力之一，它让 AI 能够：
- 获取实时信息（天气、新闻等）
- 执行计算
- 与外部系统交互
- 访问数据库
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import json
from datetime import datetime

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
        temperature=0,  # 使用工具时建议设为 0，让输出更确定
    )


# ============================================
# 1. 创建自定义工具
# ============================================

# 使用 @tool 装饰器是创建工具最简单的方式
# 函数的 docstring 会被用作工具的描述，AI 会根据描述决定何时使用这个工具

@tool
def get_current_time() -> str:
    """获取当前的日期和时间。当用户询问现在几点或今天日期时使用此工具。"""
    now = datetime.now()
    return now.strftime("%Y年%m月%d日 %H:%M:%S")


@tool
def calculate(expression: str) -> str:
    """
    计算数学表达式。支持加减乘除、幂运算等。
    
    Args:
        expression: 要计算的数学表达式，例如 "2 + 3 * 4" 或 "2 ** 10"
    
    Returns:
        计算结果
    """
    try:
        # 安全地评估数学表达式
        # 注意：在生产环境中应该使用更安全的方式
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return "错误：表达式包含不允许的字符"
        
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误：{str(e)}"


@tool
def get_weather(city: str) -> str:
    """
    获取指定城市的天气信息。
    
    Args:
        city: 城市名称，例如 "北京"、"上海"
    
    Returns:
        天气信息
    """
    # 这是模拟数据，实际应用中应该调用真实的天气 API
    weather_data = {
        "北京": {"temp": 15, "condition": "晴", "humidity": 45},
        "上海": {"temp": 18, "condition": "多云", "humidity": 60},
        "广州": {"temp": 25, "condition": "小雨", "humidity": 80},
        "深圳": {"temp": 26, "condition": "阴", "humidity": 75},
    }
    
    if city in weather_data:
        data = weather_data[city]
        return f"{city}天气：{data['condition']}，温度 {data['temp']}°C，湿度 {data['humidity']}%"
    else:
        return f"抱歉，暂无 {city} 的天气数据。支持的城市：北京、上海、广州、深圳"


@tool
def search_knowledge_base(query: str) -> str:
    """
    搜索知识库获取信息。当需要查找特定知识或事实时使用。
    
    Args:
        query: 搜索关键词
    
    Returns:
        搜索结果
    """
    # 模拟知识库
    knowledge = {
        "python": "Python 是一种解释型、面向对象、动态数据类型的高级程序设计语言。由 Guido van Rossum 于 1989 年发明。",
        "langchain": "LangChain 是一个用于开发由语言模型驱动的应用程序的框架。它提供了模块化的组件和工具链。",
        "langgraph": "LangGraph 是 LangChain 团队开发的库，用于构建有状态的多步骤 AI 应用和智能体。",
        "agent": "智能体（Agent）是能够自主决策、执行任务的 AI 系统。它可以使用工具、推理和规划来完成复杂任务。",
    }
    
    query_lower = query.lower()
    for key, value in knowledge.items():
        if key in query_lower:
            return value
    
    return f"未找到关于 '{query}' 的信息。"


def show_tool_info():
    """展示工具的基本信息"""
    print("=" * 50)
    print("1. 工具基本信息")
    print("=" * 50)
    
    tools = [get_current_time, calculate, get_weather, search_knowledge_base]
    
    for t in tools:
        print(f"\n工具名称: {t.name}")
        print(f"工具描述: {t.description}")
        print(f"参数结构: {t.args}")


# ============================================
# 2. 将工具绑定到模型
# ============================================

def bind_tools_example():
    """
    将工具绑定到模型
    
    绑定后，模型就知道有哪些工具可用，
    并能在需要时生成工具调用请求
    """
    print("\n" + "=" * 50)
    print("2. 绑定工具到模型")
    print("=" * 50)
    
    model = create_qwen_chat()
    
    # 定义可用工具
    tools = [get_current_time, calculate, get_weather]
    
    # 将工具绑定到模型
    model_with_tools = model.bind_tools(tools)
    
    print("\n工具已绑定到模型！")
    print("模型现在可以调用以下工具：")
    for t in tools:
        print(f"  - {t.name}: {t.description[:50]}...")
    
    return model_with_tools


# ============================================
# 3. AI 决定是否使用工具
# ============================================

def tool_calling_example():
    """
    工具调用示例
    
    当用户提问时，AI 会判断是否需要使用工具：
    - 如果需要，返回 tool_calls
    - 如果不需要，直接返回回答
    """
    print("\n" + "=" * 50)
    print("3. AI 决定是否使用工具")
    print("=" * 50)
    
    model = create_qwen_chat()
    tools = [get_current_time, calculate, get_weather]
    model_with_tools = model.bind_tools(tools)
    
    # 测试不同的问题
    questions = [
        "你好，介绍一下你自己",  # 不需要工具
        "现在几点了？",  # 需要 get_current_time
        "计算一下 15 * 28 + 100",  # 需要 calculate
        "北京今天天气怎么样？",  # 需要 get_weather
    ]
    
    for question in questions:
        print(f"\n问题: {question}")
        response = model_with_tools.invoke(question)
        
        # 检查是否有工具调用
        if response.tool_calls:
            print("AI 决定调用工具：")
            for call in response.tool_calls:
                print(f"  - 工具: {call['name']}")
                print(f"    参数: {call['args']}")
        else:
            print(f"AI 直接回答: {response.content[:100]}...")


# ============================================
# 4. 完整的工具调用流程
# ============================================

def complete_tool_flow():
    """
    完整的工具调用流程
    
    流程：
    1. 用户提问
    2. AI 判断是否需要工具
    3. 如果需要，调用工具
    4. 将工具结果返回给 AI
    5. AI 生成最终回答
    """
    print("\n" + "=" * 50)
    print("4. 完整的工具调用流程")
    print("=" * 50)
    
    model = create_qwen_chat()
    tools = [get_current_time, calculate, get_weather, search_knowledge_base]
    model_with_tools = model.bind_tools(tools)
    
    # 创建工具映射，方便根据名称查找工具
    tool_map = {t.name: t for t in tools}
    
    def process_query(query: str) -> str:
        """处理用户查询的完整流程"""
        print(f"\n{'='*40}")
        print(f"用户: {query}")
        print(f"{'='*40}")
        
        # 第一步：发送问题给 AI
        messages = [HumanMessage(content=query)]
        response = model_with_tools.invoke(messages)
        
        # 第二步：检查是否需要调用工具
        if not response.tool_calls:
            # 不需要工具，直接返回回答
            print(f"\n[无需工具] AI 直接回答")
            return response.content
        
        # 第三步：执行工具调用
        print(f"\n[需要工具] AI 请求调用 {len(response.tool_calls)} 个工具")
        
        # 将 AI 的响应添加到消息列表
        messages.append(response)
        
        # 执行每个工具调用
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]
            
            print(f"\n  调用工具: {tool_name}")
            print(f"  参数: {tool_args}")
            
            # 执行工具
            tool = tool_map.get(tool_name)
            if tool:
                result = tool.invoke(tool_args)
                print(f"  结果: {result}")
            else:
                result = f"未知工具: {tool_name}"
            
            # 将工具结果添加到消息列表
            messages.append(ToolMessage(
                content=str(result),
                tool_call_id=tool_id
            ))
        
        # 第四步：让 AI 根据工具结果生成最终回答
        final_response = model_with_tools.invoke(messages)
        
        print(f"\n[最终回答]")
        return final_response.content
    
    # 测试几个问题
    test_queries = [
        "现在北京时间几点？",
        "帮我计算一下，如果我每月存 3000 元，一年能存多少？",
        "深圳和广州今天哪个城市更热？",
        "什么是 LangGraph？它和 LangChain 有什么关系？",
    ]
    
    for query in test_queries:
        answer = process_query(query)
        print(f"\n回答: {answer}")
        print("\n" + "-" * 50)


# ============================================
# 5. 使用 StructuredTool 创建复杂工具
# ============================================

def structured_tool_example():
    """
    使用 StructuredTool 创建更复杂的工具
    
    当工具需要更复杂的参数结构时，可以使用 Pydantic 模型
    """
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, Field
    
    print("\n" + "=" * 50)
    print("5. 使用 StructuredTool 创建复杂工具")
    print("=" * 50)
    
    # 定义参数模型
    class ConversionInput(BaseModel):
        """单位转换工具的输入参数"""
        value: float = Field(description="要转换的数值")
        from_unit: str = Field(description="原始单位，如 km, m, cm")
        to_unit: str = Field(description="目标单位，如 km, m, cm")
    
    def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
        """执行单位转换"""
        # 转换到基础单位（米）
        to_meter = {
            "km": 1000,
            "m": 1,
            "cm": 0.01,
            "mm": 0.001,
            "mile": 1609.34,
            "ft": 0.3048,
        }
        
        if from_unit not in to_meter or to_unit not in to_meter:
            return f"不支持的单位。支持的单位：{list(to_meter.keys())}"
        
        # 转换
        meters = value * to_meter[from_unit]
        result = meters / to_meter[to_unit]
        
        return f"{value} {from_unit} = {result:.4f} {to_unit}"
    
    # 创建结构化工具
    converter_tool = StructuredTool.from_function(
        func=unit_converter,
        name="unit_converter",
        description="长度单位转换工具。支持 km, m, cm, mm, mile, ft 之间的转换。",
        args_schema=ConversionInput
    )
    
    print(f"\n工具名称: {converter_tool.name}")
    print(f"工具描述: {converter_tool.description}")
    print(f"参数结构: {converter_tool.args}")
    
    # 测试工具
    result = converter_tool.invoke({
        "value": 5,
        "from_unit": "km",
        "to_unit": "mile"
    })
    print(f"\n测试结果: {result}")
    
    return converter_tool


# ============================================
# 运行示例
# ============================================

if __name__ == "__main__":
    print("\n🚀 第三课：添加工具 (Tools) 能力\n")
    
    try:
        # 1. 展示工具信息
        show_tool_info()
        
        # 2. 绑定工具到模型
        bind_tools_example()
        
        # 3. AI 决定是否使用工具
        tool_calling_example()
        
        # 4. 完整的工具调用流程
        complete_tool_flow()
        
        # 5. 结构化工具
        structured_tool_example()
        
        print("\n" + "=" * 50)
        print("✅ 第三课完成！")
        print("=" * 50)
        print("\n📚 关键概念回顾：")
        print("1. Tool: 工具是 AI 可以调用的函数，让 AI 能够与外部世界交互")
        print("2. @tool 装饰器: 最简单的创建工具的方式")
        print("3. bind_tools: 将工具绑定到模型")
        print("4. tool_calls: AI 返回的工具调用请求")
        print("5. ToolMessage: 将工具执行结果返回给 AI")
        print("\n📚 下一课我们将使用 LangGraph 构建真正的智能体！")
        
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
