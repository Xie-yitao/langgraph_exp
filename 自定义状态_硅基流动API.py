import getpass
import os
import json
import uuid
import requests
from typing import Any, List, TypedDict
from typing_extensions import Annotated
from langchain.schema.messages import HumanMessage, AIMessage, ToolMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool, InjectedToolCallId
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
# @tool
# def human_assistance(query: str) -> str:
#     """Request assistance from a human."""
#     human_response = interrupt({"query": query})
#     return human_response["data"]

@tool
# Note that because we are generating a ToolMessage for a state update, we
# generally require the ID of the corresponding tool call. We can use
# LangChain's InjectedToolCallId to signal that this argument should not
# be revealed to the model in the tool's schema.
def human_assistance(
    name: str, birthday: str, tool_call_id: Annotated[str, InjectedToolCallId]
) -> str:
    """Request assistance from a human."""
    human_response = interrupt(
        {
            "question": "Is this correct?",
            "name": name,
            "birthday": birthday,
        },
    )
    # If the information is correct, update the state as-is.
    if human_response.get("correct", "").lower().startswith("y"):
        verified_name = name
        verified_birthday = birthday
        response = "Correct"
    # Otherwise, receive information from the human reviewer.
    else:
        verified_name = human_response.get("name", name)
        verified_birthday = human_response.get("birthday", birthday)
        response = f"Made a correction: {human_response}"

    # This time we explicitly update the state with a ToolMessage inside
    # the tool.
    state_update = {
        "name": verified_name,
        "birthday": verified_birthday,
        "messages": [ToolMessage(response, tool_call_id=tool_call_id)],
    }
    # We return a Command object in the tool to update our state.
    return Command(update=state_update)

tool = TavilySearchResults(name="web_search", max_results=1)
tools = [tool, human_assistance]

# —————— 定义 State 类型 ——————
class State(TypedDict):
    messages: Annotated[List[dict], add_messages]
    name: str
    birthday:str

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
        "model": "Qwen/Qwen2.5-72B-Instruct",
        "messages": messages,
        "stream": False,
        # 自动考虑调用工具
        "function_call": "auto",
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
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
                    "strict": False
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "human_assistance",
                    "description": "Request assistance from a human",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
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

# —————— 系统提示：何时调用 human_assistance ——————
SYSTEM_PROMPT = """
你是一个智能助理。当你面对的问题满足下列任一情形时：
  1. 你调用了检索工具还无法确定正确答案；
  2. 你需要更多上下文或专家确认；
请发起对工具 human_assistance 的调用，格式如下（严格遵循 JSON schema）：

✿FUNCTION✿: human_assistance
✿ARGS✿: {\"query\": \"<你的查询文本>\"}

等待工具返回后，再基于返回结果继续生成最终回答。
"""

# —————— 安全打印流式响应（包含 system prompt 和 thread_id） ——————
def stream_graph_updates(user_input: str):
    config = {"configurable": {"thread_id": THREAD_ID}}
    # 将 SYSTEM_PROMPT 注入为第一条 system 消息
    inputs = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input}
        ]
    }
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