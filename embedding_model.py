from langchain_community.embeddings import HuggingFaceBgeEmbeddings

class EmbeddingModel:
    def __init__(self, model_name, model_kwargs=None, encode_kwargs=None):
        self.model_name = model_name
        self.model_kwargs = model_kwargs or {}
        self.encode_kwargs = encode_kwargs or {}
        self.model = HuggingFaceBgeEmbeddings(
            model_name=self.model_name,
            model_kwargs=self.model_kwargs,
            encode_kwargs=self.encode_kwargs
        )

    def get_model(self):
        return self.model
