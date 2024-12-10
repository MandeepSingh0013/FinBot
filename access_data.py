# from langchain_chroma import Chroma
# from langchain_openai import OpenAIEmbeddings
# from dotenv import load_dotenv
# import os

# # Load environment variables
# load_dotenv()

# class ChromaManager:
#     def __init__(self, collection_name, persist_directory="./vetorstore"):
#         """
#         Initializes the Chroma vector store manager.

#         Args:
#             collection_name (str): Name of the ChromaDB collection.
#             persist_directory (str): Path to persist the ChromaDB.
#         """
#         self.collection_name = collection_name
#         self.persist_directory = persist_directory
#         self.openai_api_key = os.getenv("OPENAI_KEY")
#         self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
#         self.vector_store = Chroma(
#             collection_name=self.collection_name,
#             embedding_function=self.embeddings,
#             persist_directory=self.persist_directory,
#         )

#     def query(self, text, top_k=5):
#         """
#         Queries the ChromaDB for similar documents.

#         Args:
#             text (str): The input query.
#             top_k (int): Number of top results to return.

#         Returns:
#             list: List of relevant documents.
#         """
#         query_embedding = self.embeddings.embed_query(text)
#         results = self.vector_store.similarity_search_by_vector(
#             embedding=query_embedding, k=top_k
#         )
#         return [
#             {"content": doc.page_content, "metadata": doc.metadata} for doc in results
#         ]






# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings
# from langchain_community.embeddings import OpenAIEmbeddings
load_dotenv()


def data_retriever(query):
    # Set OpenAI API Key
    OPENAI_API_KEY = os.getenv("OPENAI_KEY")

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vector_store = Chroma(
        collection_name="findata",
        embedding_function=embeddings,
        persist_directory="./vetorstore",  # Where to save data locally, remove if not necessary
    )

    results = vector_store.similarity_search(
        query,k=2#("What is the process for applying for a aadharcard?"), k=1
    )
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in results
    )
    return results, serialized
    # # print(results)
    # for doc in results:
    #     print(f"* {doc.page_content} [{doc.metadata}]")

    # print("====================")

    # print(results[0].page_content)
# results, serialized=data_retriever("What is the process for applying for a aadharcard?")
# print(results)
# print("\n")
# print(serialized)