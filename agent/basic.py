from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()


# tool implementation depend on NLP of each docstring
def get_weather(city: str) -> str:
    """Get weather for a given city"""
    return f"It's sunny in {city}"


agent = create_agent(
    model="deepseek-chat",
    tools=[get_weather]
)

# |- invoke return a dict after sync blocking call
results = agent.invoke({"messages": [
    {"role": "user", "content": "What is the weather in Tokyo"}
]})

messages = results["messages"]
for message in messages:
    message.pretty_print()

# |- stream return an iterator
# |- |- values mode yield a cumulative dict each call
results = agent.stream(
    {"messages": [{"role": "user", "content": "What is the weather in Tokyo"}]},
    stream_mode="values"
)

for result in results:
    messages = result["messages"]
    messages[-1].pretty_print()

# |- |- update mode return a incremental dict, and add a node level encapsulate
results = agent.stream(
    {"messages": [{"role": "user", "content": "What is the weather in Tokyo"}]},
    stream_mode="updates"
)

for result in results:
    for node_name, node_result in result.items():
        node_result["messages"][0].pretty_print()
