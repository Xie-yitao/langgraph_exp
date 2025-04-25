import getpass
import os
import json
import uuid
import requests
from typing import Any, List, TypedDict
from typing_extensions import Annotated
from langchain.schema.messages import HumanMessage, AIMessage, ToolMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver

# —————— API Key 配置 ——————
if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API key:\n")
SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/chat/completions"
API_KEY = os.getenv("SILICONFLOW_API_KEY") or getpass.getpass("SiliconFlow API key:\n")

# —————— 创建搜索工具 ——————
@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]

tool = TavilySearchResults(name="web_search", max_results=1)
tools = [tool, human_assistance]

# —————— 定义 State 类型 ——————
class State(TypedDict):
    messages: Annotated[List[dict], add_messages]

graph_builder = StateGraph(State)

# —————— 消息转换（保证 content 为字符串，工具结果标为 tool） ——————
def convert_message(message: Any) -> dict:
    if isinstance(message, HumanMessage):
        return {"role": "user", "content": message.content}
    if isinstance(message, AIMessage):
        return {"role": "assistant", "content": message.content}
    if isinstance(message, ToolMessage):
        return {
            "role": "tool",
            "content": message.content,
            "tool_call_id": message.tool_call_id
        }
    return {"role": "system", "content": str(message)}

# —————— 聊天节点 ——————
def chatbot(state: State):
    messages = [convert_message(m) for m in state["messages"]]
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": messages,
        "stream": False,
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "frequency_penalty": 0.5,
        "n": 1,
        "response_format": {"type": "text"},
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for information",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"]
                    },
                    "strict": False
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "human_assistance",
                    "description": "Request assistance from a human",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"]
                    },
                    "strict": False
                }
            }
        ],
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    resp = requests.post(SILICONFLOW_API_URL, json=payload, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    choice = data["choices"][0]["message"]
    if choice.get("tool_calls"):
        return {"messages": [choice]}
    return {"messages": [{"role": "assistant", "content": choice.get("content", "")}]}

graph_builder.add_node("chatbot", chatbot)

# —————— 工具执行节点 ——————
class BasicToolNode:
    def __init__(self, tools):
        self.tools_by_name = {t.name: t for t in tools}

    def __call__(self, inputs):
        last = inputs["messages"][-1]
        outputs = []
        for tc in getattr(last, "tool_calls", []):
            result = self.tools_by_name[tc["name"]].invoke(tc["args"])
            outputs.append(ToolMessage(
                content=json.dumps(result, ensure_ascii=False),
                name=tc["name"],
                tool_call_id=tc["id"]
            ))
        return {"messages": outputs}

graph_builder.add_node("tools", BasicToolNode(tools))

# —————— 路由：有 tool_calls 则走 tools ——————
def route_tools(state: State):
    last = state["messages"][-1]
    return "tools" if hasattr(last, "tool_calls") and last.tool_calls else END

graph_builder.add_conditional_edges("chatbot", route_tools, {"tools": "tools", END: END})
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

# —————— 配置 MemorySaver 和 线程 ID ——————
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# 为会话生成一个固定 thread_id，用于持久化
THREAD_ID = str(uuid.uuid4())

# —————— 安全打印流式响应（包含 thread_id 配置） ——————
def stream_graph_updates(user_input: str):
    # 每次调用都带上 thread_id，确保 MemorySaver 能正确存取状态
    config = {"configurable": {"thread_id": THREAD_ID}}
    inputs = {"messages": [{"role": "user", "content": user_input}]}
    for event in graph.stream(inputs, config=config):
        for v in event.values():
            for msg in v.get("messages", []):
                text = msg.content if hasattr(msg, "content") else msg.get("content")
                print("Assistant:", text)

if __name__ == "__main__":
    print(f"Using THREAD_ID = {THREAD_ID}")
    while True:
        u = input("User: ")
        if u.lower() in ("q","quit","exit"):
            print("Goodbye!")
            break
        try:
            stream_graph_updates(u)
        except Exception as e:
            print("Error during chat:", e)
