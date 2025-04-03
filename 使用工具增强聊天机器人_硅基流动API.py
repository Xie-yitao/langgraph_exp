import getpass
import os

if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API key:\n")

from langchain_community.tools.tavily_search import TavilySearchResults

# tool = TavilySearchResults(max_results=1)
tool = TavilySearchResults(name="web_search", max_results=1)
tools = [tool]
# tool.invoke("What's a 'node' in LangGraph?")

from typing import Annotated

from langchain_anthropic import ChatAnthropic
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain.llms import Ollama  # 请确保安装了 langchain-ollama 扩展

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
def convert_message(message: Any) -> dict:
    """将 LangChain 消息对象转换为普通字典"""
    if isinstance(message, HumanMessage):
        return {"role": "user", "content": message.content}
    # 如果有其他消息类型，可以在这里添加转换逻辑
    return message

def chatbot(state: State):
    # 从 State 中提取用户消息
    messages = state["messages"]
    # 将消息转换为普通字典
    converted_messages = [convert_message(msg) for msg in messages]
    
    # 构建请求 payload
    payload = {
        # "model": "Qwen/Qwen2.5-7B-Instruct",
        "model": "Qwen/Qwen2.5-72B-Instruct",
        
        "messages": converted_messages,
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
        response.raise_for_status()  # 检查请求是否成功
        response_data = response.json()
        
        # 提取响应中的消息内容
        if "choices" in response_data and len(response_data["choices"]) > 0:
            assistant_message = response_data["choices"][0]["message"]
            
            # 检查响应中是否包含工具调用
            if "tool_calls" in assistant_message and len(assistant_message["tool_calls"]) > 0:
                print("Tool calls found:", assistant_message["tool_calls"])

                return {
                    "messages": [
                        {
                            "role": "assistant",
                            "content": assistant_message["content"],
                            "tool_calls": assistant_message["tool_calls"]
                        }
                    ]
                }
            else:
                print("No tool calls found in response.")
                return {"messages": [{"role": "assistant", "content": assistant_message["content"]}]}
        else:
            return {"messages": [{"role": "assistant", "content": "No response from SiliconFlow API"}]}
    
    except Exception as e:
        return {"messages": [{"role": "assistant", "content": f"Error: {str(e)}"}]}


graph_builder.add_node("chatbot", chatbot)

import json

from langchain_core.messages import ToolMessage


class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage.
        运行上一条 AIMessage 中请求的工具的节点。"""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in getattr(message, "tool_calls", []):
            tool_name = tool_call["name"]
            if tool_name not in self.tools_by_name:
                raise KeyError(f"Tool '{tool_name}' not found in registered tools.")
            tool_result = self.tools_by_name[tool_name].invoke(tool_call["args"])
            print("Tool result:", tool_result)  # 打印工具调用结果
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_name,
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}
    
tool_node = BasicToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)




def route_tools(state: State):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    print("State in route_tools:", state)  # 打印状态以调试
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    
    # 检查是否包含工具调用
    print("AI message:", ai_message)  # 打印最后一条消息以调试
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        print("Tool calls found in route_tools:", ai_message.tool_calls)  # 打印工具调用信息
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