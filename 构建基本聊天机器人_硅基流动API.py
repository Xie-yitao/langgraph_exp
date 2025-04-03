from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

import requests
import os



# 设置 SiliconFlow API 的 URL 和密钥
SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/chat/completions"
API_KEY = os.getenv("SILICONFLOW_API_KEY", "your_api_key_here")

def chatbot(state: State):
    # 从 State 中提取用户消息
    messages = state["messages"]
    
    # 构建请求 payload
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": messages,
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
        "Authorization": f"Bearer {API_KEY}", #API
        "Content-Type": "application/json"
    }
    
    # 发送请求
    try:
        response = requests.post(SILICONFLOW_API_URL, json=payload, headers=headers)
        response.raise_for_status()  # 检查请求是否成功
        response_data = response.json()
        
        # 提取响应中的消息内容
        if "choices" in response_data and len(response_data["choices"]) > 0:
            assistant_message = response_data["choices"][0]["message"]["content"]
            return {"messages": [{"role": "assistant", "content": assistant_message}]}
        else:
            return {"messages": [{"role": "assistant", "content": "No response from SiliconFlow API"}]}
    
    except Exception as e:
        return {"messages": [{"role": "assistant", "content": f"Error: {str(e)}"}]}

# 添加 chatbot 节点
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()
from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

# def stream_graph_updates(user_input: str):
#     for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
#         for value in event.values():
#             print("Assistant:", value["messages"][-1].content)

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        print("Event:", event)  # 打印事件内容
        for value in event.values():
            print("Value:", value)  # 打印值
            if "messages" in value and len(value["messages"]) > 0:
                last_message = value["messages"][-1]
                if hasattr(last_message, "content"):
                    print("Assistant:", last_message.content)


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break