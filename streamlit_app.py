import streamlit as st
from rag_qa import query_engine_function

st.title("RAG Document QA with LlamaIndex")
st.subheader("Hi! this is your Interactive QA System! built using LlamaIndex🐐 and OpenAI🔗")
st.write("📌 Check the spellings before prompting! ✨")

query = st.text_input("Enter your query:")
if query:
    try:
        response = query_engine_function(query)
        st.write("Answer:", response.response)
        st.write("Referred documents:",response.metadata) 
        # here we extract whole metadata from referred docs like file name,path,type,size,etc
        # if you only want to display test and referred docs id then use the commented code for query_engine_function in rag_qa.py
    except Exception as e:
        st.error(f"An error occurred: {e}")
