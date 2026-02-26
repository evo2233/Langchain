from typing import List
from langchain_core.documents import Document
from langchain_core.runnables import chain
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

embedding = OllamaEmbeddings(model="nomic-embed-text")

vector_0 = embedding.embed_query("Context Engineering history")

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embedding,
    persist_directory="../data/chroma_db"
)

results = vector_store.similarity_search("Context Engineering history")
for result in results:
    print(result)

results = vector_store.similarity_search_by_vector(vector_0)
for result in results:
    print(result)

results = vector_store.similarity_search_with_score("Context Engineering history")
for result, score in results:
    print(f"{result}\n得分：{score}")


def chroma_score():
    """
    Let's take a closer look at Chroma's scoring mechanism.
    """
    score_mechanisms = [
        "default",
        "cosine",  # 1 minus cosine of the angle between two vectors
        "l2",      # the L2 distance between two vectors
        "ip"       # the inner(dot) product of two vectors
    ]
    vector_stores = []
    # init db
    for measure in score_mechanisms:
        metadata = {"hnsw:space": score_mechanisms}
        if score_mechanisms is "default":
            metadata = None

        db = Chroma(
            collection_name=measure,
            embedding_function=embedding,
            persist_directory="../data/chroma_db",
            collection_metadata=metadata
        )
        vector_stores.append(db)
    # construct and store index
    docs = [
        Document(page_content="What's the weather today in Kyoto?"),
        Document(page_content="I like Akihabara")
    ]
    for db in vector_stores:
        db.add_documents(docs)
    # search
    query = "I like Anima"
    for i in range(len(score_mechanisms)):
        results = vector_stores[i].similarity_search_with_score(query)
        print(query)
        for result, score in results:
            print(result.page_content)
            print(f"{score_mechanisms[i]}: {score}")
    """
    from the result, we can know:
    the default method Chroma use is 'l2'
    all score mechanisms follow 'the smaller the closer'
    """


@chain
def retriever(query: str) -> List[Document]:
    """return the most relevant text(k: num of return)"""
    return vector_store.similarity_search(query, k=1)


results = retriever.invoke("Context Engineering history")
for result in results:
    print(result)
