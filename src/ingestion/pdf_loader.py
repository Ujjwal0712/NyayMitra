from langchain_community.document_loaders import PyPDFDirectoryLoader
import os

def load_pdf(directory_path="D:\\legalchatbot\\dataset"):
    """
    Loads PDFs from a specified directory using PyPDFDirectoryLoader.

    Args:
        directory_path (str): Path to the directory containing PDFs.

    Returns:
        list: A list of LangChain Document objects.
    """

    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    try:
        loader = PyPDFDirectoryLoader(directory_path)
        pages = loader.load()
        return pages

    except Exception as e:
        print(f"Error loading PDFs: {e}")
        return []
