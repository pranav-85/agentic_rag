from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

def connect_to_local_milvus():
    connections.connect(alias="default", uri="milvus_lite.db")

def create_collection():
    id_field = FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64)
    vector_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024)
    text_field = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=20000)
    file_name_field = FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=256)

    schema = CollectionSchema(fields=[id_field, vector_field, text_field, file_name_field], description="Document collection")

    collection_name = "Chunked_Docs"
    if not utility.has_collection(collection_name):
        collection = Collection(name=collection_name, schema=schema)
        collection.create_index(
            field_name="embedding",
            index_params={"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 1024}}
        )
        collection.load()
    else:
        collection = Collection(collection_name)
        collection.load()

    return collection

# Usage example
if __name__ == "__main__":
    connect_to_local_milvus()
    collection = create_collection()
    print(f"Collection {collection.name} ready with {collection.num_entities} entities.")
