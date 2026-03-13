from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import GitLoader, LarkSuiteDocLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 本文件旨在使用 LangChain 社区的 api 实现代码和飞书文档的向量化索引和入库存储
doc_loader = LarkSuiteDocLoader(
    domain="",
    access_token="",
    document_id=""
)
docs = doc_loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # num of char
    chunk_overlap=200,  # the overlap char length of two chunk
    add_start_index=True  # have 'start_index' key in metadata
)
all_splits = text_splitter.split_documents(docs)

# vectorisation
embedding = OllamaEmbeddings(model="nomic-embed-text")

vector_store = Chroma(
    collection_name="Lark_doc",
    embedding_function=embedding,
    persist_directory="../data/chroma_rag_db"
)

ids = vector_store.add_documents(documents=all_splits)
