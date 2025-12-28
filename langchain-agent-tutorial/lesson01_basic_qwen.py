"""
第一课：基础 - 连接阿里千问模型

本课学习目标：
1. 了解如何配置阿里千问 API
2. 使用 LangChain 连接千问模型
3. 发送第一条消息并获取回复

阿里千问支持两种调用方式：
- DashScope SDK：阿里云原生 SDK
- OpenAI 兼容接口：使用 OpenAI SDK 调用

本教程使用 OpenAI 兼容接口，因为 LangChain 对其支持更好。
"""

import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ============================================
# 方式一：使用 langchain-openai (推荐)
# ============================================

from langchain_openai import ChatOpenAI

def create_qwen_chat():
    """
    创建千问聊天模型实例
    
    千问支持 OpenAI 兼容接口，我们可以直接使用 ChatOpenAI
    只需要修改 base_url 指向千问的 API 地址
    """
    
    # 获取 API Key
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量")
    
    # 创建聊天模型
    # base_url: 千问的 OpenAI 兼容接口地址
    # model: 使用 qwen-plus 或 qwen-turbo 等模型
    chat = ChatOpenAI(
        model="qwen-plus",  # 可选: qwen-turbo, qwen-max, qwen-plus
        openai_api_key=api_key,
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.7,  # 控制回复的随机性，0-2，越高越随机
    )
    
    return chat


def basic_chat_example():
    """
    基础聊天示例：发送一条消息并获取回复
    """
    print("=" * 50)
    print("基础聊天示例")
    print("=" * 50)
    
    # 创建模型
    chat = create_qwen_chat()
    
    # 发送消息 - 最简单的方式，直接传入字符串
    response = chat.invoke("你好！请用一句话介绍一下你自己。")
    
    # 打印回复
    print(f"\n问题: 你好！请用一句话介绍一下你自己。")
    print(f"\n回复: {response.content}")
    print(f"\n回复类型: {type(response)}")
    
    return response


def message_types_example():
    """
    消息类型示例：LangChain 支持多种消息类型
    
    - SystemMessage: 系统消息，设定 AI 的角色和行为
    - HumanMessage: 用户消息
    - AIMessage: AI 的回复
    """
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    
    print("\n" + "=" * 50)
    print("消息类型示例")
    print("=" * 50)
    
    chat = create_qwen_chat()
    
    # 使用消息列表进行对话
    messages = [
        SystemMessage(content="你是一个专业的 Python 编程助手，回答要简洁明了。"),
        HumanMessage(content="Python 的列表推导式是什么？给个简单例子。")
    ]
    
    response = chat.invoke(messages)
    
    print(f"\n系统设定: 你是一个专业的 Python 编程助手，回答要简洁明了。")
    print(f"\n问题: Python 的列表推导式是什么？给个简单例子。")
    print(f"\n回复:\n{response.content}")
    
    return response


def streaming_example():
    """
    流式输出示例：实时显示 AI 的回复
    
    对于长回复，流式输出可以提升用户体验
    """
    print("\n" + "=" * 50)
    print("流式输出示例")
    print("=" * 50)
    
    chat = create_qwen_chat()
    
    print("\n问题: 用 Python 写一个快速排序的代码")
    print("\n回复 (流式输出):")
    
    # 使用 stream 方法进行流式输出
    for chunk in chat.stream("用 Python 写一个快速排序的代码，并添加注释"):
        print(chunk.content, end="", flush=True)
    
    print("\n")


# ============================================
# 运行示例
# ============================================

if __name__ == "__main__":
    print("\n🚀 第一课：连接阿里千问模型\n")
    
    try:
        # 1. 基础聊天
        basic_chat_example()
        
        # 2. 消息类型
        message_types_example()
        
        # 3. 流式输出
        streaming_example()
        
        print("\n✅ 第一课完成！")
        print("\n📚 下一课我们将学习如何使用 LangChain 的 Prompt Template。")
        
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        print("\n请检查:")
        print("1. 是否已设置 DASHSCOPE_API_KEY 环境变量")
        print("2. API Key 是否正确")
        print("3. 网络连接是否正常")
