import faiss
import numpy as np
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings
from langchain.docstore.in_memory import InMemoryDocstore

load_dotenv()

class LegalChatbot:
    def __init__(self, index_path="embeddings/faiss_index"):
        self.embeddings = MistralAIEmbeddings()
        self.index_path = index_path

        # ✅ Load FAISS index correctly
        self.index = faiss.read_index(f"{index_path}.index")

        # ✅ Initialize empty docstore (FAISS does not store raw docs)
        self.docstore = InMemoryDocstore()

        # ✅ Create a mapping from FAISS index to docstore (needed for retrieval)
        self.index_to_docstore_id = {}

        # ✅ Properly initialize FAISS with all required arguments
        self.vector_store = FAISS(
            index=self.index,
            embedding_function=self.embeddings,
            docstore=self.docstore,
            index_to_docstore_id=self.index_to_docstore_id
        )

    def retrieve_documents(self, query, k=5):
        """Retrieve top-k similar documents based on the query"""
        query_embedding = self.embeddings.embed_query(query)

        query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)  # Convert to NumPy array
        _, indices = self.index.search(query_embedding, k)

        # ✅ Fetch documents correctly
        retrieved_docs = []
        for i in indices[0]:
            if i != -1 and i in self.index_to_docstore_id:
                doc_id = self.index_to_docstore_id[i]
                doc = self.docstore.search(doc_id)
                if doc:
                    retrieved_docs.append(doc)

        return retrieved_docs
