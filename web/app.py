# from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
# from langgraph.graph import StateGraph, START, END
# from langgraph.graph.message import add_messages
# from langgraph.types import Command, interrupt
# from langgraph.checkpoint.memory import MemorySaver
# import getpass
# import os
# import json
# import uuid
# import requests
# from typing import Any, List, TypedDict
# from typing_extensions import Annotated
# from langchain.schema.messages import HumanMessage, AIMessage, ToolMessage
# from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain_core.tools import tool
# from pydantic import BaseModel

# # —————— API Key 配置 ——————
# if not os.environ.get("TAVILY_API_KEY"):
#     os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API key:\n")
# SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/chat/completions"
# API_KEY = os.getenv("SILICONFLOW_API_KEY") or getpass.getpass("SiliconFlow API key:\n")

# # —————— 创建搜索工具 ——————
# @tool
# def human_assistance(query: str) -> str:
#     """Request assistance from a human."""
#     human_response = interrupt({"query": query})
#     return human_response["data"]

# tool = TavilySearchResults(name="web_search", max_results=1)
# tools = [tool, human_assistance]

# # —————— 定义 State 类型 ——————
# class State(TypedDict):
#     messages: Annotated[List[dict], add_messages]

# graph_builder = StateGraph(State)

# # —————— 消息转换（保证 content 为字符串，工具结果标为 tool） ——————
# def convert_message(message: Any) -> dict:
#     if isinstance(message, HumanMessage):
#         return {"role": "user", "content": message.content}
#     if isinstance(message, AIMessage):
#         return {"role": "assistant", "content": message.content}
#     if isinstance(message, ToolMessage):
#         return {
#             "role": "tool",
#             "content": message.content,
#             "tool_call_id": message.tool_call_id
#         }
#     return {"role": "system", "content": str(message)}

# # —————— 聊天节点 ——————
# def chatbot(state: State):
#     messages = [convert_message(m) for m in state["messages"]]
#     payload = {
#         "model": "Qwen/Qwen2.5-7B-Instruct",
#         "messages": messages,
#         "stream": False,
#         "max_tokens": 512,
#         "temperature": 0.7,
#         "top_p": 0.7,
#         "top_k": 50,
#         "frequency_penalty": 0.5,
#         "n": 1,
#         "response_format": {"type": "text"},
#         "tools": [
#             {
#                 "type": "function",
#                 "function": {
#                     "name": "web_search",
#                     "description": "Search the web for information",
#                     "parameters": {
#                         "type": "object",
#                         "properties": {"query": {"type": "string"}},
#                         "required": ["query"]
#                     },
#                     "strict": False
#                 }
#             },
#             {
#                 "type": "function",
#                 "function": {
#                     "name": "human_assistance",
#                     "description": "Request assistance from a human",
#                     "parameters": {
#                         "type": "object",
#                         "properties": {"query": {"type": "string"}},
#                         "required": ["query"]
#                     },
#                     "strict": False
#                 }
#             }
#         ],
#     }
#     headers = {
#         "Authorization": f"Bearer {API_KEY}",
#         "Content-Type": "application/json"
#     }
#     resp = requests.post(SILICONFLOW_API_URL, json=payload, headers=headers)
#     resp.raise_for_status()
#     data = resp.json()
#     choice = data["choices"][0]["message"]
#     # if choice.get("tool_calls"):
#     #     return {"messages": [choice]}
#     if hasattr(choice, "tool_calls") and choice.tool_calls:
#         return {"messages": [choice]}
#     return {"messages": [{"role": "assistant", "content": choice.get("content", "")}]}

# graph_builder.add_node("chatbot", chatbot)

# # —————— 工具执行节点 ——————
# class BasicToolNode:
#     def __init__(self, tools):
#         self.tools_by_name = {t.name: t for t in tools}

#     def __call__(self, inputs):
#         last = inputs["messages"][-1]
#         outputs = []
#         for tc in getattr(last, "tool_calls", []):
#             result = self.tools_by_name[tc["name"]].invoke(tc["args"])
#             outputs.append(ToolMessage(
#                 content=json.dumps(result, ensure_ascii=False),
#                 name=tc["name"],
#                 tool_call_id=tc["id"]
#             ))
#         return {"messages": outputs}

# graph_builder.add_node("tools", BasicToolNode(tools))

# # —————— 路由：有 tool_calls 则走 tools ——————
# def route_tools(state: State):
#     last = state["messages"][-1]
#     return "tools" if hasattr(last, "tool_calls") and last.tool_calls else END

# graph_builder.add_conditional_edges("chatbot", route_tools, {"tools": "tools", END: END})
# graph_builder.add_edge("tools", "chatbot")
# graph_builder.add_edge(START, "chatbot")

# # —————— 配置 MemorySaver 和 线程 ID ——————
# memory = MemorySaver()
# graph = graph_builder.compile(checkpointer=memory)

# app = FastAPI()
# app.mount("/static", StaticFiles(directory="static"), name="static")
# templates = Jinja2Templates(directory="templates")

# class ChatRequest(BaseModel):
#     message: str
#     thread_id: str
#     show_thinking: bool

# @app.get("/")
# async def read_root(request: Request):
#     thread_id = str(uuid.uuid4())
#     return templates.TemplateResponse("index.html", {"request": request, "thread_id": thread_id})

# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     thread_id = None
    
#     while True:
#         try:
#             data = await websocket.receive_json()
#             message = data.get("message")
#             thread_id = data.get("thread_id")
#             show_thinking = data.get("show_thinking", False)
            
#             if not thread_id:
#                 thread_id = str(uuid.uuid4())
            
#             config = {"configurable": {"thread_id": thread_id}}
#             inputs = {"messages": [{"role": "user", "content": message}]}
            
#             response_history = []
#             async for event in graph.astream(inputs, config=config):
#                 for v in event.values():
#                     for msg in v.get("messages", []):
#                         content = msg.content if hasattr(msg, "content") else msg.get("content", "")
#                         role = msg.role if hasattr(msg, "role") else msg.get("role", "assistant")
                        
#                         if show_thinking or role != "tool":
#                             response_history.append({
#                                 "role": role,
#                                 "content": content
#                             })
            
#             await websocket.send_json({
#                 "thread_id": thread_id,
#                 "history": response_history,
#                 "final_response": response_history[-1]["content"] if response_history else ""
#             })
            
#         except WebSocketDisconnect:
#             break
#         except Exception as e:
#             await websocket.send_json({"error": str(e)})




# from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, status
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
# from langgraph.graph import StateGraph, START, END
# from langgraph.graph.message import add_messages
# from langgraph.types import Command, interrupt
# from langgraph.checkpoint.memory import MemorySaver
# import getpass
# import os
# import json
# import uuid
# import requests
# from typing import Any, List, TypedDict, Optional
# from typing_extensions import Annotated
# from langchain.schema.messages import HumanMessage, AIMessage, ToolMessage
# from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain_core.tools import tool
# from pydantic import BaseModel

# # —————— API Key 配置 ——————
# if not os.environ.get("TAVILY_API_KEY"):
#     os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API key:\n")
    
# SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/chat/completions"
# API_KEY = os.getenv("SILICONFLOW_API_KEY") or getpass.getpass("SiliconFlow API key:\n")

# # —————— 创建搜索工具 ——————
# @tool
# def human_assistance(query: str) -> str:
#     """Request assistance from a human."""
#     human_response = interrupt({"query": query})
#     return human_response["data"]

# web_search = TavilySearchResults(name="web_search", max_results=1)
# tools = [web_search, human_assistance]

# # —————— 定义 State 类型 ——————
# class State(TypedDict):
#     messages: Annotated[List[dict], add_messages]

# graph_builder = StateGraph(State)

# # —————— 消息转换（保证 content 为字符串，工具结果标为 tool） ——————
# def convert_message(message: Any) -> dict:
#     if isinstance(message, HumanMessage):
#         return {"role": "user", "content": message.content}
#     if isinstance(message, AIMessage):
#         return {"role": "assistant", "content": message.content}
#     if isinstance(message, ToolMessage):
#         return {
#             "role": "tool",
#             "content": message.content,
#             "tool_call_id": message.tool_call_id
#         }
#     return {"role": "system", "content": str(message)}

# # —————— 聊天节点 ——————
# def chatbot(state: State):
#     messages = [convert_message(m) for m in state["messages"]]
#     payload = {
#         "model": "Qwen/Qwen3-8B",
#         "messages": messages,
#         "stream": False,
#         "max_tokens": 512,
#         "temperature": 0.7,
#         "top_p": 0.7,
#         "top_k": 50,
#         "frequency_penalty": 0.5,
#         "n": 1,
#         "response_format": {"type": "text"},
#         "tools": [
#             {
#                 "type": "function",
#                 "function": {
#                     "name": "web_search",
#                     "description": "Search the web for information",
#                     "parameters": {
#                         "type": "object",
#                         "properties": {"query": {"type": "string"}},
#                         "required": ["query"]
#                     },
#                     "strict": False
#                 }
#             },
#             {
#                 "type": "function",
#                 "function": {
#                     "name": "human_assistance",
#                     "description": "Request assistance from a human",
#                     "parameters": {
#                         "type": "object",
#                         "properties": {"query": {"type": "string"}},
#                         "required": ["query"]
#                     },
#                     "strict": False
#                 }
#             }
#         ],
#     }
#     headers = {
#         "Authorization": f"Bearer {API_KEY}",
#         "Content-Type": "application/json"
#     }
#     try:
#         resp = requests.post(SILICONFLOW_API_URL, json=payload, headers=headers)
#         resp.raise_for_status()
#         data = resp.json()
#         choice = data["choices"][0]["message"]
#         return {"messages": [choice]}
#     except Exception as e:
#         return {"messages": [{"role": "assistant", "content": f"Error: {str(e)}"}]}

# graph_builder.add_node("chatbot", chatbot)

# # —————— 工具执行节点 ——————
# class BasicToolNode:
#     def __init__(self, tools):
#         self.tools_by_name = {t.name: t for t in tools}

#     def __call__(self, inputs):
#         last = inputs["messages"][-1]
#         outputs = []
#         tool_calls = getattr(last, "tool_calls", [])
        
#         for tc in tool_calls:
#             tool_name = tc.get("name")
#             tool_args = tc.get("args", {})
            
#             if tool_name in self.tools_by_name:
#                 try:
#                     result = self.tools_by_name[tool_name].invoke(tool_args)
#                     outputs.append(ToolMessage(
#                         content=json.dumps(result, ensure_ascii=False),
#                         name=tool_name,
#                         tool_call_id=tc.get("id")
#                     ))
#                 except Exception as e:
#                     outputs.append(ToolMessage(
#                         content=f"Error executing {tool_name}: {str(e)}",
#                         name=tool_name,
#                         tool_call_id=tc.get("id")
#                     ))
#         return {"messages": outputs}

# graph_builder.add_node("tools", BasicToolNode(tools))

# # —————— 路由：有 tool_calls 则走 tools ——————
# def route_tools(state: State):
#     last_message = state["messages"][-1]
#     return "tools" if hasattr(last_message, "tool_calls") and last_message.tool_calls else END

# graph_builder.add_conditional_edges("chatbot", route_tools, {"tools": "tools", END: END})
# graph_builder.add_edge("tools", "chatbot")
# graph_builder.add_edge(START, "chatbot")

# # —————— 配置 MemorySaver 和 线程 ID ——————
# memory = MemorySaver()
# graph = graph_builder.compile(checkpointer=memory)

# app = FastAPI()
# app.mount("/static", StaticFiles(directory="static"), name="static")
# templates = Jinja2Templates(directory="templates")

# class ChatRequest(BaseModel):
#     message: str
#     thread_id: str
#     show_thinking: bool

# @app.get("/")
# async def read_root(request: Request):
#     thread_id = str(uuid.uuid4())
#     return templates.TemplateResponse("index.html", {"request": request, "thread_id": thread_id})

# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     thread_id = None
#     message_history = []
    
#     try:
#         while True:
#             data = await websocket.receive_json()
#             message = data.get("message")
#             thread_id = data.get("thread_id") or str(uuid.uuid4())
#             show_thinking = data.get("show_thinking", False)
            
#             # 添加用户消息到历史记录
#             user_message = {"role": "user", "content": message}
#             message_history.append(user_message)
            
#             # 准备输入
#             config = {"configurable": {"thread_id": thread_id}}
#             inputs = {"messages": message_history.copy()}
            
#             # 获取模型响应
#             response_history = []
#             tool_invocation_count = 0
#             max_tool_invocations = 3  # 限制工具调用次数
            
#             async for event in graph.astream(inputs, config=config):
#                 for v in event.values():
#                     messages = v.get("messages", [])
                    
#                     for msg in messages:
#                         content = msg.get("content", "")
#                         role = msg.get("role", "assistant")
                        
#                         # 处理工具调用
#                         if role == "tool":
#                             tool_invocation_count += 1
#                             if tool_invocation_count > max_tool_invocations:
#                                 content = "Too many tool invocations, stopping."
#                                 role = "assistant"
                            
#                             if show_thinking:
#                                 response_history.append({
#                                     "role": "thinking",
#                                     "content": f"Tool {msg.get('name')} called with args: {msg.get('content')}"
#                                 })
                        
#                         # 添加到响应历史
#                         if show_thinking or role not in ["tool", "thinking"]:
#                             response_history.append({
#                                 "role": role,
#                                 "content": content
#                             })
                        
#                         # 添加到消息历史
#                         message_history.append(msg)
            
#             # 发送最终响应
#             await websocket.send_json({
#                 "thread_id": thread_id,
#                 "history": response_history,
#                 "final_response": response_history[-1]["content"] if response_history else ""
#             })
    
#     except WebSocketDisconnect:
#         print("WebSocket connection closed")
#     except Exception as e:
#         await websocket.send_json({
#             "error": str(e),
#             "thread_id": thread_id or str(uuid.uuid4())
#         })
#         raise

# # —————— 健康检查 ——————
# @app.get("/health")
# async def health_check():
#     return {"status": "ok"}



from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, status
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
import getpass
import os
import json
import uuid
import requests
from typing import Any, List, TypedDict, Optional
from typing_extensions import Annotated
from langchain.schema.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from pydantic import BaseModel

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

web_search = TavilySearchResults(name="web_search", max_results=1)
tools = [web_search, human_assistance]

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
            "tool_call_id": message.tool_call_id if hasattr(message, 'tool_call_id') else None
        }
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}
    return {"role": "system", "content": str(message)}

# —————— 聊天节点 ——————
def chatbot(state: State):
    messages = [convert_message(m) for m in state["messages"]]
    payload = {
        "model": "Qwen/Qwen3-8B",
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
    try:
        resp = requests.post(SILICONFLOW_API_URL, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]["message"]
        return {"messages": [choice]}
    except Exception as e:
        return {"messages": [{"role": "assistant", "content": f"Error: {str(e)}"}]}

graph_builder.add_node("chatbot", chatbot)

# —————— 工具执行节点 ——————
class BasicToolNode:
    def __init__(self, tools):
        self.tools_by_name = {t.name: t for t in tools}

    def __call__(self, inputs):
        last = inputs["messages"][-1]
        outputs = []
        tool_calls = getattr(last, "tool_calls", [])
        
        for tc in tool_calls:
            tool_name = tc.get("name")
            tool_args = tc.get("args", {})
            tool_id = tc.get("id")  # 获取tool_call_id
            
            if tool_name in self.tools_by_name:
                try:
                    result = self.tools_by_name[tool_name].invoke(tool_args)
                    # 添加思考过程消息，使用SystemMessage类型
                    outputs.append({
                        "role": "system",
                        "content": f"Calling tool {tool_name} with args: {json.dumps(tool_args)}"
                    })
                    # 确保ToolMessage包含tool_call_id
                    outputs.append({
                        "role": "tool",
                        "content": json.dumps(result, ensure_ascii=False),
                        "tool_call_id": tool_id
                    })
                except Exception as e:
                    outputs.append({
                        "role": "system",
                        "content": f"Error calling tool {tool_name}: {str(e)}"
                    })
                    outputs.append({
                        "role": "tool",
                        "content": f"Error executing {tool_name}: {str(e)}",
                        "tool_call_id": tool_id
                    })
        return {"messages": outputs}

graph_builder.add_node("tools", BasicToolNode(tools))

# —————— 路由：有 tool_calls 则走 tools ——————
def route_tools(state: State):
    last_message = state["messages"][-1]
    return "tools" if hasattr(last_message, "tool_calls") and last_message.tool_calls else END

graph_builder.add_conditional_edges("chatbot", route_tools, {"tools": "tools", END: END})
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

# —————— 配置 MemorySaver 和 线程 ID ——————
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class ChatRequest(BaseModel):
    message: str
    thread_id: str
    show_thinking: bool

@app.get("/")
async def read_root(request: Request):
    thread_id = str(uuid.uuid4())
    return templates.TemplateResponse("index.html", {"request": request, "thread_id": thread_id})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    thread_id = None
    message_history = []
    
    try:
        while True:
            data = await websocket.receive_json()
            message = data.get("message")
            thread_id = data.get("thread_id") or str(uuid.uuid4())
            show_thinking = data.get("show_thinking", False)
            
            # 添加用户消息到历史记录
            user_message = {"role": "user", "content": message}
            message_history.append(user_message)
            
            # 准备输入
            config = {"configurable": {"thread_id": thread_id}}
            inputs = {"messages": message_history.copy()}
            
            # 获取模型响应
            response_history = []
            tool_invocation_count = 0
            max_tool_invocations = 3  # 限制工具调用次数
            
            async for event in graph.astream(inputs, config=config):
                for v in event.values():
                    messages = v.get("messages", [])
                    
                    for msg in messages:
                        # 判断msg是否为字典格式，如果不是转换为字典
                        if isinstance(msg, dict):
                            content = msg.get("content", "")
                            role = msg.get("role", "assistant")
                            tool_call_id = msg.get("tool_call_id", None)
                        else:
                            content = msg.content if hasattr(msg, 'content') else str(msg)
                            role = msg.role if hasattr(msg, 'role') else "assistant"
                            tool_call_id = msg.tool_call_id if hasattr(msg, 'tool_call_id') else None
                        
                        # 处理工具调用
                        if role == "tool":
                            tool_invocation_count += 1
                            if tool_invocation_count > max_tool_invocations:
                                content = "Too many tool invocations, stopping."
                                role = "assistant"
                        
                        # 添加到响应历史
                        if show_thinking or role != "system":
                            response_history.append({
                                "role": role,
                                "content": content,
                                "tool_call_id": tool_call_id  # 传递tool_call_id
                            })
                        
                        # 添加到消息历史
                        message_history.append({
                            "role": role,
                            "content": content,
                            "tool_call_id": tool_call_id  # 传递tool_call_id
                        })
            
            # 发送最终响应
            await websocket.send_json({
                "thread_id": thread_id,
                "history": response_history,
                "final_response": response_history[-1]["content"] if response_history else ""
            })
    
    except WebSocketDisconnect:
        print("WebSocket connection closed")
    except Exception as e:
        await websocket.send_json({
            "error": str(e),
            "thread_id": thread_id or str(uuid.uuid4())
        })
        raise

# —————— 健康检查 ——————
@app.get("/health")
async def health_check():
    return {"status": "ok"}