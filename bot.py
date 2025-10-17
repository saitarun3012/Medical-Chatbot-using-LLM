import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Load environment variables (make sure you have a .env file with GROQ_API_KEY)
from dotenv import load_dotenv
load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    try:
        embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        return None

def set_custom_prompt():
    CUSTOM_PROMPT_TEMPLATE = """
    Use the pieces of information provided in the context to answer user's question.
    If you dont know the answer, just say that you don't know, don't try to make up an answer. 
    Don't provide anything out of the given context.

    Context: {context}
    Question: {question}

    Start the answer directly. No small talk please.
    """
    return PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE, 
        input_variables=["context", "question"]
    )

def load_groq_llm():
    """Load Groq LLM with error handling"""
    try:
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            st.error("GROQ_API_KEY not found in environment variables")
            return None
        
        llm = ChatGroq(
            model_name="llama-3.1-8b-instant",  # Free, fast Groq model
            temperature=0.1,  # Lower temperature for more factual responses
            max_tokens=512,
            groq_api_key=groq_api_key,
        )
        return llm
    except Exception as e:
        st.error(f"Error loading Groq LLM: {str(e)}")
        return None

def main():
    st.title("🤖 MediBot - Medical Assistant")
    st.write("Ask me anything about medical information from the provided documents!")

    # Initialize session state for messages
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    # Chat input
    prompt = st.chat_input("Ask your medical question here...")

    if prompt:
        # Display user message
        with st.chat_message('user'):
            st.markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        # Process the query
        with st.chat_message('assistant'):
            with st.spinner("Thinking..."):
                try:
                    # Get vector store
                    vectorstore = get_vectorstore()
                    if vectorstore is None:
                        st.error("Failed to load the knowledge base. Please check if the vector store exists.")
                        return

                    # Load LLM
                    llm = load_groq_llm()
                    if llm is None:
                        st.error("Failed to load the language model. Please check your API key.")
                        return

                    # Create QA chain
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=vectorstore.as_retriever(
                            search_type="similarity",
                            search_kwargs={'k': 4}  # Increased to get more context
                        ),
                        return_source_documents=True,
                        chain_type_kwargs={'prompt': set_custom_prompt()}
                    )

                    # Get response
                    response = qa_chain.invoke({'query': prompt})
                    
                    # Display results
                    result = response["result"]
                    source_documents = response["source_documents"]
                    
                    # Format source documents nicely
                    source_info = "\n\n**Sources:**\n"
                    for i, doc in enumerate(source_documents, 1):
                        page_info = f"Page {doc.metadata.get('page', 'N/A')}" if 'page' in doc.metadata else ""
                        source_info += f"{i}. {page_info} - {doc.page_content[:100]}...\n"
                    
                    full_response = f"{result}\n{source_info}"
                    
                    st.markdown(full_response)
                    st.session_state.messages.append({'role': 'assistant', 'content': full_response})

                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({'role': 'assistant', 'content': error_msg})

# Add some instructions and sidebar info
def add_sidebar_info():
    with st.sidebar:
        st.header("ℹ️ About MediBot")
        st.write("""
        MediBot is a medical assistant chatbot that can answer questions 
        based on the medical documents you've provided.
        
        **Features:**
        - Answers questions from uploaded medical PDFs
        - Provides source references
        - Fast responses using Groq's LLM API
        """)
        
        st.header("⚙️ Setup Instructions")
        st.write("""
        1. Add your Groq API key to a `.env` file:
           ```
           GROQ_API_KEY=your_groq_api_key_here
           ```
        2. Place medical PDFs in the `data/` folder
        3. Run the vector store creation script first
        """)

if __name__ == "__main__":
    add_sidebar_info()
    main()
