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


def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")
_set_if_undefined("OPENAI_API_KEY")
_set_if_undefined("TAVILY_API_KEY")

web_search = TavilySearch(max_results=3)
# 初始化模型
web_search_model = init_chat_model(
    model="Qwen/Qwen3-8B",
    model_provider="openai",
    temperature=0,
    openai_api_base="https://api.siliconflow.cn/v1",
    openai_api_key=os.environ.get("OPENAI_API_KEY")
)
research_agent = create_react_agent(
    model=web_search_model,
    tools=[web_search],
    prompt=(
        "You are a research agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with research-related tasks, DO NOT do any math\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="research_agent",
)


# 定义一个函数，用于打印单条消息
def pretty_print_message(message, indent=False):
    """
    将消息格式化为可读形式并打印出来。
    Args:
        message: 要打印的消息对象。
        indent: 是否缩进打印，默认为 False。
    """
    # 获取消息的 HTML 格式表示
    pretty_message = message.pretty_repr(html=True)
    # 如果不需要缩进，直接打印消息
    if not indent:
        print(pretty_message)
        return

    # 如果需要缩进，将消息的每一行前面加上制表符
    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)


# 定义一个函数，用于打印流代理输出的消息
def pretty_print_messages(update, last_message=False):
    """
    打印流代理输出的消息。
    Args:
        update: 流代理输出的更新内容。
        last_message: 是否只打印最后一条消息，默认为 False。
    """
    is_subgraph = False
    # 如果更新内容是元组，说明包含命名空间信息
    if isinstance(update, tuple):
        ns, update = update  # 解包元组，获取命名空间和更新内容
        # 如果命名空间为空，跳过打印
        if len(ns) == 0:
            return

        # 提取图 ID
        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")  # 打印子图更新信息
        print("\n")
        is_subgraph = True  # 标记为子图更新

    # 遍历更新内容中的每个节点及其更新
    for node_name, node_update in update.items():
        # 构造节点更新的标签
        update_label = f"Update from node {node_name}:"
        # 如果是子图更新，标签前加制表符缩进
        if is_subgraph:
            update_label = "\t" + update_label

        print(update_label)  # 打印节点更新标签
        print("\n")

        # 将消息转换为 LangChain 消息对象
        messages = convert_to_messages(node_update["messages"])
        # 如果只打印最后一条消息，取最后一条
        if last_message:
            messages = messages[-1:]

        # 遍历消息并打印
        for m in messages:
            pretty_print_message(m, indent=is_subgraph)  # 调用打印单条消息的函数
        print("\n")


# # 使用研究代理进行流式查询并打印结果
# for chunk in research_agent.stream(
#     {"messages": [{"role": "user", "content": "who is the mayor of NYC?"}]}
# ):
#     pretty_print_messages(chunk)  # 调用打印消息的函数

def add(a: float, b: float):
    """Add two numbers."""
    return a + b


def multiply(a: float, b: float):
    """Multiply two numbers."""
    return a * b


def divide(a: float, b: float):
    """Divide two numbers."""
    return a / b

math_model = init_chat_model(
    model="Qwen/Qwen3-8B",
    model_provider="openai",
    temperature=0,
    openai_api_base="https://api.siliconflow.cn/v1",
    openai_api_key=os.environ.get("OPENAI_API_KEY")
)
math_agent = create_react_agent(
    model=math_model,
    tools=[add, multiply, divide],
    prompt=(
        "You are a math agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with math-related tasks\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="math_agent",
)


# 任务描述传递工具生成函数
def create_task_description_handoff_tool(
    *, 
    agent_name: str,  # 目标代理名称
    description: str | None = None  # 工具描述（可选）
):
    """
    创建一个工具，用于将带有任务描述的任务转移给指定代理
    
    参数：
        agent_name: 目标代理名称
        description: 工具描述（默认自动生成）
    返回：
        一个能够将任务转移给指定代理的工具函数
    """
    name = f"transfer_to_{agent_name}"  # 生成工具名称
    description = description or f"Ask {agent_name} for help."  # 生成默认描述

    # 使用tool装饰器定义具体工具逻辑
    @tool(name, description=description)
    def handoff_tool(
        # 由监督代理（supervisor LLM）填充的任务描述
        task_description: Annotated[
            str,
            "Description of what the next agent should do, including all of the relevant context.",
        ],
        # 从状态中注入的当前消息状态（但不会被LLM填充）
        state: Annotated[MessagesState, InjectedState],
    ) -> Command:
        """
        带任务描述的任务转移工具实现
        
        参数：
            task_description: 下一个代理应执行的任务描述（由监督代理生成）
            state: 当前对话状态（消息历史）
        返回：
            Command对象，指示将任务转移给目标代理
        """
        # 创建包含任务描述的用户消息
        task_description_message = {"role": "user", "content": task_description}
        
        # 构造目标代理的输入状态
        agent_input = {**state, "messages": [task_description_message]}
        
        # 返回命令对象，指示状态转移
        return Command(
            # 指定状态转移目标为发送任务到目标代理
            goto=[Send(agent_name, agent_input)],
            # 指定状态转移发生在父图中
            graph=Command.PARENT,
        )

    # 返回生成的工具函数
    return handoff_tool

# 创建带有任务描述的任务转移工具实例
# 将任务转移给检索代理的工具（带任务描述）
assign_to_research_agent_with_description = create_task_description_handoff_tool(
    agent_name="research_agent",
    description="Assign task to a researcher agent with detailed task description."
)

# 将任务转移给数学代理的工具（带任务描述）
assign_to_math_agent_with_description = create_task_description_handoff_tool(
    agent_name="math_agent",
    description="Assign task to a math agent with detailed task description."
)

supervisor_model = init_chat_model(
    model="Qwen/Qwen3-8B",
    model_provider="openai",
    temperature=0.7,
    openai_api_base="https://api.siliconflow.cn/v1",
    openai_api_key=os.environ.get("OPENAI_API_KEY")
)
# 创建监督代理（supervisor agent）
supervisor_agent_with_description = create_react_agent(
    model=supervisor_model,  # 使用GPT-4.1模型作为监督代理
    tools=[
        assign_to_research_agent_with_description,  # 提供研究代理转移工具
        assign_to_math_agent_with_description,      # 提供数学代理转移工具
    ],
    # 监督代理的提示（prompt）定义其行为：
    # - 管理两个专业代理（研究代理和数学代理）
    # - 根据任务性质分配给合适的专业代理
    # - 每次只分配给一个代理，不并行调用
    # - 自身不做任何实际工作，只负责任务分配
    prompt=(
        "You are a supervisor managing two agents:\n"
        "- a research agent. Assign research-related tasks to this assistant\n"
        "- a math agent. Assign math-related tasks to this assistant\n"
        "Assign work to one agent at a time, do not call agents in parallel.\n"
        "Do not do any work yourself."
    ),
    name="supervisor",  # 监督代理名称
)

# 构建状态图，定义代理之间的交互流程
supervisor_with_description = (
    StateGraph(MessagesState)  # 使用消息状态作为状态类型
    # 添加监督代理节点，指定其可以转移到研究代理和数学代理
    .add_node(
        supervisor_agent_with_description, 
        destinations=("research_agent", "math_agent")
    )
    .add_node(research_agent)  # 添加研究代理节点
    .add_node(math_agent)      # 添加数学代理节点
    # 定义状态图的边，描述代理之间的调用关系：
    # - 从起点（START）开始于监督代理
    # - 研究代理完成后返回监督代理
    # - 数学代理完成后返回监督代理
    .add_edge(START, "supervisor")
    .add_edge("research_agent", "supervisor")
    .add_edge("math_agent", "supervisor")
    .compile()  # 编译状态图，使其可执行
)

for chunk in supervisor_with_description.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "find US and New York state GDP in 2024. what % of US GDP was New York state?",
            }
        ]
    },
    subgraphs=True,
):
    pretty_print_messages(chunk, last_message=True)