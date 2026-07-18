from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.tools import tool
load_dotenv()

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store = Chroma(
    collection_name="ani_list_202607",
    embedding_function=embeddings,
    persist_directory="../data/chroma_langchain_db"
)


@tool(response_format="content_and_artifact")
def retrive_context(query: str):
    """Retrive information to help answer a query"""
    retrive_docs = vector_store.similarity_search(query, k=2)
    content = '\n\n'.join(
        f"Source:{doc.metadata}\nContent:{doc.page_content}" for doc in retrive_docs
    )
    return content, retrive_docs


system_prompt = """
请你结合自己的理解，使用我们提供的信息检索工具，回答用户的问题。
"""
agent = create_agent(
    model="deepseek:deepseek-chat",
    tools=[retrive_context],
    system_prompt=system_prompt
)
results = agent.invoke(
    {"messages": [{"role": "user", "content": "统计一下 2026年7月新番表 中，出现次数最多的一个人的名字。"}]}
)

messages = results["messages"]
for message in messages:
    message.pretty_print()
