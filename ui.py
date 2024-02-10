import streamlit as st
from main import get_qa_chain, create_vector_db
st.title("Question Answering System 📚")
st.subheader("Ask any question and get the answer from the given context")

btn = st.button("Create Database") 
if btn:
    pass
question = st.text_input("Enter your question here")
if question:
    chain = get_qa_chain()
    response =chain(question)
    
    st.header("Answer: ")
    st.write(response["result"])
    
