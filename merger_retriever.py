import chromadb
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.vectorstores import Chroma
from langchain.retrievers.merger_retriever import MergerRetriever

class MergerRetrieverHandler:
    def __init__(self, retrievers):
        self.retrievers = retrievers
        self.merger = MergerRetriever(retrievers=self.retrievers)

    def get_relevant_documents(self, query):
        return self.merger.get_relevant_documents(query)

