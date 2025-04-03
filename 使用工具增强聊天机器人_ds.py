import getpass
import os
from typing import Dict, List

if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API key:\n")

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage

# 初始化工具和模型
tool = TavilySearchResults(max_results=1)   #max_results检索放回条数
llm = ChatOllama(
    model="deepseek-r1:7b",
    # base_url="http://10.10.14.3:11434",
    base_url="http://127.0.0.1:11434",
    temperature=0.5
)

def search_and_answer(user_query: str) -> str:
    """执行搜索并生成回答"""
    # 1. 执行搜索
    search_results = tool.invoke(user_query)
    
    # 2. 构建带上下文的提示词
    context = "\n".join([f"来源 {i+1}: {res['content']}" for i, res in enumerate(search_results)])
    prompt = f"""请基于以下信息回答问题：
{context}

问题：{user_query}
答案："""
    
    # 3. 调用模型生成回答
    response = llm.invoke(prompt)
    return response.content

def normal_answer(user_query: str) -> str:
    """直接生成回答"""
    response = llm.invoke(user_query)
    return response.content

def process_input(user_input: str) -> str:
    """处理用户输入"""
    if user_input.lower().startswith("search"):
        # 提取实际查询内容
        query = user_input[len("search"):].strip()
        return search_and_answer(query)
    else:
        return normal_answer(user_input)

# 交互循环
while True:
    try:
        user_input = input("\n用户（输入'search: 问题'进行联网搜索，或输入exit退出）: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("再见！")
            break
            
        response = process_input(user_input)
        print(f"\n助手：{response}")
        
    except Exception as e:
        print(f"出错：{str(e)}")
        break