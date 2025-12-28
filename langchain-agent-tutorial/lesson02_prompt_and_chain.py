"""
第二课：使用 LangChain 构建简单对话

本课学习目标：
1. 了解 Prompt Template（提示词模板）
2. 了解 Chain（链）的概念
3. 使用 LCEL (LangChain Expression Language) 构建工作流
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

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
# 1. Prompt Template（提示词模板）
# ============================================

def prompt_template_example():
    """
    Prompt Template 示例
    
    Prompt Template 让你可以：
    - 定义可复用的提示词结构
    - 动态插入变量
    - 保持提示词的一致性
    """
    print("=" * 50)
    print("1. Prompt Template 示例")
    print("=" * 50)
    
    # 创建一个简单的提示词模板
    # {topic} 是一个变量，稍后会被替换
    prompt = ChatPromptTemplate.from_template(
        "你是一位资深的技术专家。请用通俗易懂的语言解释什么是 {topic}，"
        "并给出一个实际应用的例子。回答控制在100字以内。"
    )
    
    # 查看模板结构
    print(f"\n模板变量: {prompt.input_variables}")
    
    # 格式化模板 - 将变量替换为实际值
    formatted = prompt.format(topic="机器学习")
    print(f"\n格式化后的提示词:\n{formatted}")
    
    # 使用模型回答
    chat = create_qwen_chat()
    response = chat.invoke(prompt.format_messages(topic="机器学习"))
    
    print(f"\n回复:\n{response.content}")
    
    return response


def chat_prompt_template_example():
    """
    聊天提示词模板示例
    
    ChatPromptTemplate 支持多种消息类型：
    - system: 系统消息
    - human: 用户消息  
    - ai: AI 消息
    """
    print("\n" + "=" * 50)
    print("2. 聊天提示词模板示例")
    print("=" * 50)
    
    # 创建带有系统消息的聊天模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个{role}，专门帮助用户解答{domain}相关的问题。回答要专业但通俗易懂。"),
        ("human", "{question}")
    ])
    
    print(f"\n模板变量: {prompt.input_variables}")
    
    # 使用模板
    chat = create_qwen_chat()
    messages = prompt.format_messages(
        role="Python 导师",
        domain="Python 编程",
        question="什么是装饰器？"
    )
    
    response = chat.invoke(messages)
    
    print(f"\n问题: 什么是装饰器？")
    print(f"\n回复:\n{response.content}")
    
    return response


# ============================================
# 2. Chain（链）和 LCEL
# ============================================

def simple_chain_example():
    """
    简单链示例
    
    LCEL (LangChain Expression Language) 使用 | 操作符
    将多个组件连接成一个链：
    
    prompt | model | output_parser
    
    这类似于 Unix 管道，数据从左到右流动。
    """
    print("\n" + "=" * 50)
    print("3. 简单链 (Chain) 示例")
    print("=" * 50)
    
    # 创建组件
    prompt = ChatPromptTemplate.from_template(
        "将以下内容翻译成{language}：\n\n{text}"
    )
    model = create_qwen_chat()
    output_parser = StrOutputParser()  # 将 AIMessage 转换为字符串
    
    # 使用 LCEL 创建链
    # | 操作符连接各个组件
    chain = prompt | model | output_parser
    
    # 调用链
    result = chain.invoke({
        "language": "英文",
        "text": "人工智能正在改变我们的生活方式。"
    })
    
    print(f"\n原文: 人工智能正在改变我们的生活方式。")
    print(f"翻译 (英文): {result}")
    
    # 再试一次翻译成日语
    result_ja = chain.invoke({
        "language": "日语",
        "text": "今天天气真好！"
    })
    
    print(f"\n原文: 今天天气真好！")
    print(f"翻译 (日语): {result_ja}")
    
    return result


def chain_with_multiple_steps():
    """
    多步骤链示例
    
    可以创建多个链，然后组合它们
    """
    print("\n" + "=" * 50)
    print("4. 多步骤链示例")
    print("=" * 50)
    
    model = create_qwen_chat()
    
    # 第一步：生成故事大纲
    outline_prompt = ChatPromptTemplate.from_template(
        "请为一个关于{theme}的短故事生成一个简短的大纲（3个要点）。"
    )
    
    # 第二步：根据大纲写故事
    story_prompt = ChatPromptTemplate.from_template(
        "根据以下大纲，写一个100字左右的短故事：\n\n{outline}"
    )
    
    # 创建链
    outline_chain = outline_prompt | model | StrOutputParser()
    story_chain = story_prompt | model | StrOutputParser()
    
    # 执行第一步
    print("\n主题: 一只勇敢的小猫")
    outline = outline_chain.invoke({"theme": "一只勇敢的小猫"})
    print(f"\n故事大纲:\n{outline}")
    
    # 执行第二步
    story = story_chain.invoke({"outline": outline})
    print(f"\n完整故事:\n{story}")
    
    return story


def runnable_passthrough_example():
    """
    RunnablePassthrough 示例
    
    有时我们需要在链中传递原始输入，
    RunnablePassthrough 可以帮助我们做到这一点
    """
    from langchain_core.runnables import RunnablePassthrough, RunnableParallel
    
    print("\n" + "=" * 50)
    print("5. RunnablePassthrough 示例")
    print("=" * 50)
    
    model = create_qwen_chat()
    
    # 创建一个同时返回原文和翻译的链
    prompt = ChatPromptTemplate.from_template(
        "将以下中文翻译成英文，只返回翻译结果：\n{text}"
    )
    
    # RunnableParallel 允许并行执行多个操作
    chain = RunnableParallel(
        original=RunnablePassthrough(),  # 传递原始输入
        translated=prompt | model | StrOutputParser()  # 翻译
    )
    
    result = chain.invoke({"text": "学习编程很有趣！"})
    
    print(f"\n原文: {result['original']['text']}")
    print(f"翻译: {result['translated']}")
    
    return result


# ============================================
# 3. 实战：创建一个代码审查助手
# ============================================

def code_review_assistant():
    """
    实战：代码审查助手
    
    这个例子展示如何创建一个实用的代码审查工具
    """
    print("\n" + "=" * 50)
    print("6. 实战：代码审查助手")
    print("=" * 50)
    
    model = create_qwen_chat()
    
    # 创建代码审查提示词模板
    review_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一位经验丰富的代码审查专家。
请审查用户提供的代码，从以下几个方面给出建议：
1. 代码质量和可读性
2. 潜在的 bug 或问题
3. 性能优化建议
4. 最佳实践

请用简洁的中文回答。"""),
        ("human", "请审查以下 {language} 代码：\n\n```{language}\n{code}\n```")
    ])
    
    # 创建链
    review_chain = review_prompt | model | StrOutputParser()
    
    # 测试代码
    test_code = '''
def find_user(users, name):
    for i in range(len(users)):
        if users[i]["name"] == name:
            return users[i]
    return None

def get_user_emails(users):
    emails = []
    for user in users:
        emails.append(user["email"])
    return emails
'''
    
    print(f"\n待审查的代码:\n{test_code}")
    
    review = review_chain.invoke({
        "language": "python",
        "code": test_code
    })
    
    print(f"\n审查结果:\n{review}")
    
    return review


# ============================================
# 运行示例
# ============================================

if __name__ == "__main__":
    print("\n🚀 第二课：Prompt Template 和 Chain\n")
    
    try:
        # 1. Prompt Template 基础
        prompt_template_example()
        
        # 2. 聊天提示词模板
        chat_prompt_template_example()
        
        # 3. 简单链
        simple_chain_example()
        
        # 4. 多步骤链
        chain_with_multiple_steps()
        
        # 5. RunnablePassthrough
        runnable_passthrough_example()
        
        # 6. 实战：代码审查助手
        code_review_assistant()
        
        print("\n" + "=" * 50)
        print("✅ 第二课完成！")
        print("=" * 50)
        print("\n📚 关键概念回顾：")
        print("1. Prompt Template: 可复用的提示词模板，支持变量替换")
        print("2. LCEL: 使用 | 操作符连接组件，创建数据处理流水线")
        print("3. Chain: 多个组件的组合，实现复杂的处理逻辑")
        print("\n📚 下一课我们将学习如何给 AI 添加工具（Tools）能力！")
        
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
