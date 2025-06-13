import getpass
import os
from langgraph.types import Send
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import convert_to_messages
from typing import Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.types import Command

# 设置 API 密钥
def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"请输入您的 {var}: ")
_set_if_undefined("OPENAI_API_KEY")
_set_if_undefined("TAVILY_API_KEY")

# 初始化搜索工具
web_search = TavilySearch(max_results=3)

# 初始化模型（用于研究代理）
web_search_model = init_chat_model(
    model="Qwen/Qwen3-8B",
    model_provider="openai",
    temperature=0,
    openai_api_base="https://api.siliconflow.cn/v1",  # 这里需要替换成有效的 API 地址
    openai_api_key=os.environ.get("OPENAI_API_KEY")
)

# 创建研究代理
research_agent = create_react_agent(
    model=web_search_model,
    tools=[web_search],
    prompt=(
        "您是研究代理。\n\n"
        "指令：\n"
        "- 仅协助与研究相关的任务，不要做任何数学计算\n"
        "- 完成任务后直接向主管报告\n"
        "- 仅回复您的工作结果，不要包含任何其他文本。"
    ),
    name="research_agent",
)

# 定义打印消息的辅助函数
def pretty_print_message(message, indent=False):
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        return
    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)

def pretty_print_messages(update, last_message=False):
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        if len(ns) == 0:
            return
        graph_id = ns[-1].split(":")[0]
        print(f"来自子图 {graph_id} 的更新:")
        print("\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        update_label = f"来自节点 {node_name} 的更新:"
        if is_subgraph:
            update_label = "\t" + update_label
        print(update_label)
        print("\n")
        messages = convert_to_messages(node_update["messages"])
        if last_message:
            messages = messages[-1:]
        for m in messages:
            pretty_print_message(m, indent=is_subgraph)
        print("\n")

# 定义数学计算函数
def add(a: float, b: float):
    """加法"""
    return a + b

def multiply(a: float, b: float):
    """乘法"""
    return a * b

def divide(a: float, b: float):
    """除法"""
    return a / b

# 初始化数学模型和代理
math_model = init_chat_model(
    model="Qwen/Qwen3-8B",
    model_provider="openai",
    temperature=0,
    openai_api_base="https://api.siliconflow.cn/v1",  # 这里需要替换成有效的 API 地址
    openai_api_key=os.environ.get("OPENAI_API_KEY")
)
math_agent = create_react_agent(
    model=math_model,
    tools=[add, multiply, divide],
    prompt=(
        "您是数学代理。\n\n"
        "指令：\n"
        "- 仅协助与数学相关的任务\n"
        "- 完成任务后直接向主管报告\n"
        "- 仅回复您的工作结果，不要包含任何其他文本。"
    ),
    name="math_agent",
)

# 创建任务转移工具
def create_task_description_handoff_tool(*, agent_name: str, description: str | None = None):
    name = f"transfer_to_{agent_name}"
    description = description or f"向 {agent_name} 寻求帮助。"
    
    @tool(name, description=description)
    def handoff_tool(
        task_description: Annotated[
            str,
            "描述下一个代理应执行的任务，包括所有相关上下文。",
        ],
        state: Annotated[MessagesState, InjectedState]
    ) -> Command:
        task_description_message = {"role": "user", "content": task_description}
        agent_input = {**state, "messages": [task_description_message]}
        return Command(
            goto=[Send(agent_name, agent_input)],
            graph=Command.PARENT,
        )
    return handoff_tool

assign_to_research_agent_with_description = create_task_description_handoff_tool(
    agent_name="research_agent",
    description="将任务分配给研究代理，并提供详细的任务描述。"
)

assign_to_math_agent_with_description = create_task_description_handoff_tool(
    agent_name="math_agent",
    description="将任务分配给数学代理，并提供详细的任务描述。"
)

# 初始化监督代理模型
supervisor_model = init_chat_model(
    model="Qwen/Qwen3-8B",
    model_provider="openai",
    temperature=0.7,
    openai_api_base="https://api.siliconflow.cn/v1",  # 这里需要替换成有效的 API 地址
    openai_api_key=os.environ.get("OPENAI_API_KEY")
)

# 创建监督代理
supervisor_agent_with_description = create_react_agent(
    model=supervisor_model,
    tools=[
        assign_to_research_agent_with_description,
        assign_to_math_agent_with_description,
    ],
    prompt=(
        "您是主管，管理两个代理：\n"
        "- 研究代理。将研究相关任务分配给此代理\n"
        "- 数学代理。将数学相关任务分配给此代理\n"
        "一次只分配任务给一个代理，不要并行调用。\n"
        "您自己不要做任何实际工作，只负责任务分配。"
    ),
    name="supervisor",
)

# 构建状态图
supervisor_with_description = (
    StateGraph(MessagesState)
    .add_node(supervisor_agent_with_description, destinations=("research_agent", "math_agent"))
    .add_node(research_agent)
    .add_node(math_agent)
    .add_edge(START, "supervisor")
    .add_edge("research_agent", "supervisor")
    .add_edge("math_agent", "supervisor")
    .compile()
)

# 持续对话循环
print("智能助手已启动，您现在可以开始对话...")
print("输入 '退出' 或 'quit' 结束对话...")

while True:
    user_input = input("您: ")
    
    if user_input.lower() in ["退出", "quit"]:
        print("对话已结束。感谢您的使用！")
        break
    
    if not user_input.strip():
        print("请输入有效内容...")
        continue
    
    # 使用监督代理处理问题
    for chunk in supervisor_with_description.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        subgraphs=True,
    ):
        pretty_print_messages(chunk, last_message=True)