from langchain.chains import RetrievalQA
from langchain_mistralai import ChatMistralAI
from src.chatbot.retriever import LegalChatbot

class LegalChatbotResponse:
    def __init__(self):
        self.chatbot = LegalChatbot()
        self.llm = ChatMistralAI()

    def generate_answer(self, query):
        retrieved_docs = self.chatbot.retrieve_documents(query)

        content = "\n\n".join([doc.page_content for doc in retrieved_docs])

        prompt = f"User query: {query}\n\nRelevant legal documents: \n{content}\n\nProvide a concise legal answer with respect to the Indian Penal Code:"
        respones = self.llm.invoke(prompt)
        return respones
    

if __name__ == "__main__":
    bot = LegalChatbotResponse()
    user_query = input("Ask you question: ")
    answer = bot.generate_answer(user_query)
    print("\nNyayMitra: ", answer)