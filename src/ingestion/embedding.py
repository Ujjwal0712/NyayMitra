from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings
from pdf_loader import load_pdf
from multiprocessing import Pool, cpu_count
import faiss
import pickle
import os
import gc  
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def chunk_worker(text):
    """Process each text document into semantic chunks."""
    try:
        text_splitter = SemanticChunker(MistralAIEmbeddings())
        return text_splitter.split_text(text)
    except Exception as e:
        print(f"Error in chunking: {e}")
        return []

def split_text_parallel(texts, num_workers=2, max_docs=5):
    """
    Split texts into chunks using multiprocessing.
    - Limits the number of concurrent workers to avoid system overload.
    - Processes a limited number of documents at a time to control memory usage.
    """
    num_workers = max(1, min(num_workers, cpu_count() - 1))  # Use fewer CPU cores

    all_chunks = []
    for i in range(0, len(texts), max_docs):
        batch_texts = texts[i : i + max_docs]  # Process in smaller batches
        with Pool(num_workers) as p:
            chunks = p.map(chunk_worker, batch_texts)
        
        all_chunks.extend([chunk for sublist in chunks for chunk in sublist])  # Flatten list
        
        # Free up memory after each batch
        gc.collect()

    return all_chunks

def store_embeddings(chunks, save_path="embeddings/faiss_index", batch_size=16):
    """
    Store embeddings in FAISS and save index.
    
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure directory exists
    embeddings = MistralAIEmbeddings()
    try:
        # ✅ Proper FAISS Initialization
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)

        # ✅ Save FAISS index properly
        faiss.write_index(vector_store.index, f"{save_path}.index")
        print(f"FAISS index successfully saved at {save_path}.index")
    
    except Exception as e:
        print(f"Error in storing embeddings: {e}")

if __name__ == "__main__":
    # Load PDF documents
    print("Loading PDF documents")
    documents = load_pdf()
    text_list = [doc.page_content for doc in documents]

    # Split text in parallel with limited processing
    print("Splitting text into semantic chunks")
    splits = split_text_parallel(text_list, num_workers=2, max_docs=3)
    print(f"Total chunks generated: {len(splits)}")

    # Store embeddings in a controlled manner
    print("Storing embeddings in FAISS index")
    store_embeddings(splits, batch_size=8)
