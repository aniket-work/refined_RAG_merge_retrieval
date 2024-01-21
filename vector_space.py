from langchain_community.vectorstores import Chroma

class VectorSpace:
    def __init__(self, texts, embedding_model, persist_directory):
        self.texts = texts
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory

    def create_chroma_vector_space(self):
        return Chroma.from_documents(
            self.texts,
            self.embedding_model,
            collection_metadata={"hnsw:space": "cosine"},
            persist_directory=self.persist_directory
        )

    def load_chroma_vector_space(self):
        return Chroma(persist_directory=self.persist_directory, embedding_function=self.embedding_model)
