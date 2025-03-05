from src.chatbot.response import LegalChatbotResponse

def main():
    print("Welcome to NyayMitra - Your AI Legal Assistant")
    bot = LegalChatbotResponse()

    while True:
        query = input("\nAsk your legal question (or type 'exit' to quit): ")
        if query.lower() == "exit":
            print("Goodbye!")
            break

        answer = bot.generate_answer(query)
        print("\nChatbot:", answer)

if __name__ == "__main__":
    main()
