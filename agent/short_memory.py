from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()

agent = create_agent(model="deepseek-chat")
message_history = []  # only use in disposable_memory


def disposable_memory(query):
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


if __name__ == '__main__':
    chat_list = [
        "Which city is Asakusa belongs to?",
        "What about Akihabara?"
    ]
    for q in chat_list:
        disposable_memory(q)
