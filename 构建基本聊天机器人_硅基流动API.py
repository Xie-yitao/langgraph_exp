import getpass
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Any
from langchain.schema.messages import HumanMessage
import requests
import os

# 设置 SiliconFlow API 的 URL 和密钥
SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/chat/completions"
API_KEY = os.getenv("SILICONFLOW_API_KEY") or getpass.getpass("SiliconFlow API key:\n")
print("API_KEY:", API_KEY)  # 打印 API 密钥以供调试

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

def convert_message(message: Any) -> dict:
    """将 LangChain 消息对象转换为普通字典"""
    if isinstance(message, HumanMessage):
        return {"role": "user", "content": message.content}
    # 如果有其他消息类型，可以在这里添加转换逻辑
    return {"role": "system", "content": str(message)}  # 默认处理未知类型

def chatbot(state: State):
    # 从 State 中提取用户消息
    messages = state["messages"]
    # 将消息转换为普通字典
    converted_messages = [convert_message(msg) for msg in messages]
    
    # 构建请求 payload
    payload = {
        # "model": "Qwen/Qwen2.5-7B-Instruct",
        "model":"Qwen/Qwen3-8B",
        "messages": converted_messages,
        "stream": False,
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "frequency_penalty": 0.5,
        "response_format": {"type": "text"}
    }
    
    # 设置请求头
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        # print("Sending request to:", SILICONFLOW_API_URL)
        # print("Payload:", payload)
        # print("Headers:", headers)
        
        response = requests.post(SILICONFLOW_API_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()  # 检查请求是否成功
        
        # print("Response status code:", response.status_code)
        # print("Response content:", response.content)
        
        response_data = response.json()
        
        # 提取响应中的消息内容
        if "choices" in response_data and len(response_data["choices"]) > 0:
            assistant_message = response_data["choices"][0]["message"]["content"]
            return {"messages": [{"role": "assistant", "content": assistant_message}]}
        else:
            return {"messages": [{"role": "assistant", "content": "No response from SiliconFlow API"}]}
    
    except requests.exceptions.RequestException as e:
        error_message = f"Request failed: {str(e)}"
        print(error_message)  # 打印错误信息
        return {"messages": [{"role": "assistant", "content": error_message}]}
    except Exception as e:
        error_message = f"Error: {str(e)}"
        print(error_message)  # 打印错误信息
        return {"messages": [{"role": "assistant", "content": error_message}]}

# 添加 chatbot 节点
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            if "messages" in value and len(value["messages"]) > 0:
                last_message = value["messages"][-1]
                if isinstance(last_message, dict) and "content" in last_message:
                    print("Assistant:", last_message["content"])  # 修正输出格式

def main():
    print("Chatbot started. Type 'quit', 'exit', or 'q' to end the conversation.")
    
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            stream_graph_updates(user_input)
        
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            break

if __name__ == "__main__":
    main()