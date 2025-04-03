import getpass
import os
from typing import Annotated, Any
from typing_extensions import TypedDict
import requests
import json
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.schema.messages import HumanMessage, AIMessage, ToolMessage  # 添加了消息类型导入

from typing import Annotated

from langchain_anthropic import ChatAnthropic
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain.llms import Ollama  # 请确保安装了 langchain-ollama 扩展


# 初始环境设置（保持不变）
if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API key:\n")

# 工具设置（保持不变）
tool = TavilySearchResults(name="web_search", max_results=1)
tools = [tool]


# 状态定义（保持不变）
class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

from typing import Annotated, Any
from typing_extensions import TypedDict
from langchain.schema.messages import HumanMessage
import requests
import os
import json


# 设置 SiliconFlow API 的 URL 和密钥
SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/chat/completions"
API_KEY = os.getenv("SILICONFLOW_API_KEY", "sk-iihecohtojjkbgwaqaoukdopasitiyneinjstosyadytfepl")

# 修改点1：增强消息转换函数
def convert_message(message: Any) -> dict:
    """将各种消息类型转换为API需要的字典格式"""
    if isinstance(message, HumanMessage):
        return {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        msg = {"role": "assistant", "content": message.content}
        if hasattr(message, "tool_calls"):
            msg["tool_calls"] = [{
                "id": call.get("id", ""),
                "type": "function",
                "function": {
                    "name": call.get("name", ""),
                    "arguments": json.dumps(call.get("args", {}))
                }
            } for call in getattr(message, "tool_calls", [])]
        return msg
    elif isinstance(message, ToolMessage):
        return {
            "role": "tool",
            "content": message.content,
            "tool_call_id": message.tool_call_id
        }
    # 处理普通字典格式的消息
    if isinstance(message, dict):
        return message
    raise ValueError(f"Unsupported message type: {type(message)}")

def chatbot(state: State):

    messages = [convert_message(msg) for msg in state["messages"]]
    
    # 构建请求 payload
    payload = {
        # "model": "Qwen/Qwen2.5-7B-Instruct",
        "model": "Qwen/Qwen2.5-72B-Instruct",
        
        "messages": messages,
        "stream": False,
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "frequency_penalty": 0.5,
        "response_format": {"type": "text"},
        "tools": [
            {
                "type": "function",
                "function": {
                    "description": "Search the web for information",
                    "name": "web_search",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                    "strict": False
                }
            }
        ]
    }
    
    # 设置请求头
    headers = {
        "Authorization": f"Bearer {API_KEY}",  # API
        "Content-Type": "application/json"
    }
    
    # 发送请求
    try:
        response = requests.post(SILICONFLOW_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        response_data = response.json()
        
        if "choices" in response_data and response_data["choices"]:
            msg = response_data["choices"][0]["message"]
            result = {"role": "assistant", "content": msg.get("content", "")}
            
            if "tool_calls" in msg:
                result["tool_calls"] = [{
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": call["function"]["name"],
                        "arguments": call["function"]["arguments"]
                    }
                } for call in msg["tool_calls"]]
            
            return {"messages": [result]}
        return {"messages": [{"role": "assistant", "content": "No response"}]}
    except Exception as e:
        return {"messages": [{"role": "assistant", "content": f"Error: {str(e)}"}]}


graph_builder.add_node("chatbot", chatbot)

import json

from langchain_core.messages import ToolMessage


# 修改点3：改进的ToolNode实现
class BasicToolNode:
    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        messages = inputs.get("messages", [])
        if not messages:
            raise ValueError("No message found in input")
        
        last_message = messages[-1]
        outputs = []
        
        # 处理字典格式的消息
        if isinstance(last_message, dict):
            for tool_call in last_message.get("tool_calls", []):
                tool_name = tool_call["function"]["name"]
                if tool_name not in self.tools_by_name:
                    continue
                try:
                    args = json.loads(tool_call["function"]["arguments"])
                except json.JSONDecodeError:
                    args = {"query": tool_call["function"]["arguments"]}
                
                tool_result = self.tools_by_name[tool_name].invoke(args)
                outputs.append({
                    "role": "tool",
                    "content": json.dumps(tool_result),
                    "tool_call_id": tool_call["id"]
                })
        
        return {"messages": outputs}
    
tool_node = BasicToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)




# 修改点4：增强路由判断
def route_tools(state: State):
    messages = state.get("messages", [])
    if not messages:
        return END
    
    last_message = messages[-1]
    if isinstance(last_message, dict):
        if last_message.get("role") == "assistant" and "tool_calls" in last_message:
            return "tools"
    return END


# The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if
# it is fine directly responding. This conditional routing defines the main agent loop.
# 如果聊天机器人要求使用工具，则 `tools_condition` 函数返回“tools”，如果直接响应则返回“END”。此条件路由定义了主代理循环。
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
    # It defaults to the identity function, but if you
    # want to use a node named something else apart from "tools",
    # You can update the value of the dictionary to something else
    # e.g., "tools": "my_tools"

    # 以下字典可让您告诉图表将条件的输出解释为特定节点
    # 它默认为身份函数，但如果您
    # 想要使用除“tools”之外的其他名称的节点，
    # 您可以将字典的值更新为其他名称
    # 例如，“tools”：“my_tools”
    {"tools": "tools", END: END},
)
# Any time a tool is called, we return to the chatbot to decide the next step
# 每次调用工具时，我们都会返回聊天机器人来决定下一步
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()



def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            if isinstance(value.get("messages"), list) and len(value["messages"]) > 0:
                last_message = value["messages"][-1]
                if isinstance(last_message, dict) and "content" in last_message:
                    print("Assistant:", last_message["content"])
                elif hasattr(last_message, "content"):
                    print("Assistant:", last_message.content)
                else:
                    print("Assistant: No content found in message")
            else:
                print("Assistant: No messages in response")

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