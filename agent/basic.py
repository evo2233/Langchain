from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Context:
    user_gender: str = "Unknown"


@dataclass
class Response:
    greeting: str
    weather_forecast: str


@tool
def get_weather(runtime: ToolRuntime[Context] ,city: str) -> str:
    """Get weather for a given city"""
    # tool implementation depend on NLP of each docstring
    user_gender = runtime.context.user_gender
    if user_gender == "Male":
        appellation = "Sir"
    elif user_gender == "Female":
        appellation = "Madam"
    else:
        appellation = "friend"
    return f"Dear {appellation}, it's sunny in {city}"


agent = create_agent(
    model="deepseek-chat",
    system_prompt="You're an expert and polite weather forecaster.",
    tools=[get_weather],
    context_schema=Context,
    response_format=Response
)

# |- invoke return a dict after sync blocking call
print("=== 1 ===")
results = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather in Tokyo"}]},
    context=Context(user_gender="Male")
)

messages = results["messages"]
for message in messages:
    message.pretty_print()

# |- stream return an iterator
# |- |- values mode yield a cumulative dict each call
print("=== 2 ===")
results = agent.stream(
    {"messages": [{"role": "user", "content": "What is the weather in Tokyo"}]},
    stream_mode="values"
)

for result in results:
    messages = result["messages"]
    messages[-1].pretty_print()

# |- |- update mode return a incremental dict, and add a node level encapsulate
print("=== 3 ===")
results = agent.stream(
    {"messages": [{"role": "user", "content": "What is the weather in Tokyo"}]},
    stream_mode="updates"
)

for result in results:
    for node_name, node_result in result.items():
        node_result["messages"][0].pretty_print()
