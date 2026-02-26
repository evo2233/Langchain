import chromadb


def list_collection(db_path):
    client = chromadb.PersistentClient(db_path)
    collections = client.list_collections()
    print(f"chromadb: {db_path} 有{len(collections)}个集合")

    for i, collection in enumerate(collections):
        print(f"collection{i}: {collection.name}, 有{collection.count()}条记录")


def delete_collection(db_path, collection_name):
    try:
        client = chromadb.PersistentClient(db_path)
        client.delete_collection(collection_name)
    except Exception as e:
        print(f"Error occurred when deleting {collection_name}, {e}")
