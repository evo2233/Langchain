import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

url = "https://yuc.wiki/202607/"

html = requests.get(
    url,
    headers={
        "User-Agent": "Mozilla/5.0"
    },
    timeout=20
).text

soup = BeautifulSoup(html, "html.parser")

doc = Document(
    page_content=soup.get_text("\n"),
    metadata={
        "source": url
    }
)

print(len(doc.page_content))

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)
all_splits = text_splitter.split_documents([doc])
print(len(all_splits))
print(all_splits)

from langchain_ollama import OllamaEmbeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

from langchain_chroma import Chroma
vector_store = Chroma(
    collection_name="ani_list_202607",
    embedding_function=embeddings,
    persist_directory="../data/chroma_langchain_db"
)
ids = vector_store.add_documents(documents=all_splits)
