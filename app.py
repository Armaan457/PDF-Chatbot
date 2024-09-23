import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_community.embeddings  import OllamaEmbeddings

import os
from dotenv import load_dotenv
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="llama3-8b-8192", groq_api_key = groq_api_key)

session_id = "pdf_bot1"
config = {"configurable":{"session_id":session_id}}
if "store" not in st.session_state:
    st.session_state.store = {}

st.title("Conversational PDF Chatbot")
st.write("Ask questions in regard to a pdf's content (Currently trained on a pdf on Mangonels)")


embeddings = OllamaEmbeddings(model = "llama2")
db = Chroma(persist_directory="./pdf_chroma_db", embedding_function=embeddings)
retriever = db.as_retriever()

contextualise_q_system_prompt = (
    "Given a chat history and the latest user question"
    "which might reference a context in the chat history"
    "formulate a standalone question that can be understood without the chat history"
    "Do not answer the question, just reformulate it and return it as it it is"
)

contextualise_q_prompt = ChatPromptTemplate(
    [
        ("system", contextualise_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualise_q_prompt)

system_prompt = (
    "You are a helpful assistant for question answering tasks"
    "Use the follwing pieces of context to answer the question"
    "If you don't know the answer, just say that you don't know"
    "Use at max 3 lines and keep the answers precise"
    "\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

def get_session_history(session_id : str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

user_input  = st.text_input("What is your question?")
if user_input:
    session_hist = get_session_history(session_id)
    hist_dict = {}
    response = conversational_rag_chain.invoke(
        {"input" : user_input},
        config 
    )
    st.write("Assistant:", response["answer"])

    for i in range(len(session_hist.messages)-1):
        hist_dict[session_hist.messages[i].content] = session_hist.messages[i+1].content
            
    st.write("Chat_History:", hist_dict)