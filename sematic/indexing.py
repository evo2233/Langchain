# 这个程序能将 PDF 读取到向量数据库，为后续的语义检索提供数据源
from langchain_community.document_loaders import PyPDFLoader

file_path = r"../data/raw_data/Context Engineering 2.0.pdf"

# split pdf into pages, return a List[documents]
# each item is an indexable object, and contains
# page_content: raw string of the page
# metadata: a dictionary
loader = PyPDFLoader(file_path)
docs = loader.load()

from langchain_text_splitters import RecursiveCharacterTextSplitter

# split page into chunks, return a List[documents] (same as page)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # num of char
    chunk_overlap=200,  # the overlap char length of two chunk
    add_start_index=True  # have 'start_index' key in metadata
)
all_splits = text_splitter.split_documents(docs)

# vectorisation: use embedded model to map the text to vector
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

embedding = OllamaEmbeddings(model="nomic-embed-text")
# we can use code like below to implement embedded model
# vector_0 = embedding.embed_query(all_splits[0].page_content)

# generally, we directly use vector base (like Chroma) to translate and store vector
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embedding,
    persist_directory="../data/chroma_db"
)

ids = vector_store.add_documents(documents=all_splits)
