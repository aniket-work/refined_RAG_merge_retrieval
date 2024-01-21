
# Example usage:
# factory = EmbeddingModelFactory()
# model = factory.create_model()  # Defaults to HuggingFaceBgeEmbeddings
# model_custom = factory.create_model(model_name="HuggingFaceGPT2Embeddings", model_kwargs={"param1": "value1"})

from langchain_community.embeddings import (
    HuggingFaceBgeEmbeddings,
    OpenAIEmbeddings,
    LlamaCppEmbeddings,
    MosaicMLInstructorEmbeddings
)

class EmbeddingModelFactory:
    """
    A factory class for creating various embedding models from Hugging Face.

    Methods:
    - create_model(model_name, model_kwargs=None, encode_kwargs=None):
        Create an embedding model based on the specified model name.

    Attributes:
    - default_model (str): The default model to use when no specific model is requested.
    - supported_models (list): List of supported embedding model names.
    """

    def __init__(self, default_model="BAAI/bge-large-en"):
        """
        Initialize the EmbeddingModelFactory.

        Args:
        - default_model (str): The default model to use when no specific model is requested.
        """
        self.default_model = default_model
        self.supported_models = [
            "BAAI/bge-large-en",
            "OpenAIEmbeddings",
            "LlamaCppEmbeddings",
            "MosaicMLInstructorEmbeddings",
        ]


    def create_model(self, model_name=None, model_kwargs=None, encode_kwargs=None):
        """
        Create an embedding model based on the specified model name.

        Args:
        - model_name (str, optional): The name of the embedding model to create.
        - model_kwargs (dict, optional): Additional keyword arguments for initializing the model.
        - encode_kwargs (dict, optional): Additional keyword arguments for encoding text with the model.

        Returns:
        - object: The initialized embedding model.
        """
        model_name = model_name or self.default_model

        if model_name in self.supported_models:
            if model_name == "BAAI/bge-large-en":
                return HuggingFaceBgeEmbeddings(
                    model_name=model_name,
                    model_kwargs=model_kwargs or {},
                    encode_kwargs=encode_kwargs or {}
                )
            elif model_name == "OpenAIEmbeddings":
                return OpenAIEmbeddings(
                    model_name=model_name,
                    model_kwargs=model_kwargs or {},
                    encode_kwargs=encode_kwargs or {}
                )
            elif model_name == "LlamaCppEmbeddings":
                return LlamaCppEmbeddings(
                    model_name=model_name,
                    model_kwargs=model_kwargs or {},
                    encode_kwargs=encode_kwargs or {}
                )
            elif model_name == "MosaicMLInstructorEmbeddings":
                return MosaicMLInstructorEmbeddings(
                    model_name=model_name,
                    model_kwargs=model_kwargs or {},
                    encode_kwargs=encode_kwargs or {}
                )
        else:
            raise ValueError(f"Unsupported embedding model: {model_name}")


