import os
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv

# ✅ Load API key from .env file or environment variables
load_dotenv()  
api_key = os.getenv("GROQ_API_KEY")

# ✅ Load documentation from URLs
doc_urls = [
    "https://segment.com/docs/",
    "https://docs.mparticle.com/",
    "https://docs.lytics.com/",
    "https://docs.zeotap.com/home/en-us/"
]

loader = UnstructuredURLLoader(doc_urls)
docs = loader.load()

# ✅ Setup LangChain QA Chain with Groq (without vector store)
chat_model = ChatGroq(model_name="mixtral-8x7b-32768", temperature=0, api_key=api_key)
qa_chain = load_qa_chain(llm=chat_model, chain_type="stuff")  # Using Stuff Chain

# ✅ Chat function
def chat(query: str):
    return qa_chain.run(input_documents=docs, question=query)

# ✅ Run the chatbot
if __name__ == "__main__":
    response = chat("How do I set up a new source in Segment?")
    print(response)
