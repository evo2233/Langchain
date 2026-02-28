from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv

load_dotenv()

agent = create_agent(model="deepseek-chat")
message_history = []


def handmade_saver(query):
    message_history.append(
        {"role": "user", "content": query}
    )
    print(f"================================ Human Message =================================")
    print(f"\n{query}")

    results = agent.stream(
        {"messages": message_history},
        stream_mode="updates"
    )

    for result in results:
        for node in result.values():
            node["messages"][0].pretty_print()
            message_history.extend(node["messages"])


checkpointer = InMemorySaver()
agent_m = create_agent(
    model="deepseek-chat",
    checkpointer=checkpointer
)
config = {"configurable": {"thread_id": "0"}}  # config the session ID


def official_saver(query):
    results = agent_m.invoke(
        {"messages": {"role": "user", "content": query}},
        config=config
    )

    messages = results["messages"]
    for message in messages:
        message.pretty_print()


if __name__ == '__main__':
    chat_list = [
        "Which city is Asakusa belongs to?",
        "What about Akihabara?"
    ]
    for q in chat_list:
        handmade_saver(q)
        official_saver(q)
