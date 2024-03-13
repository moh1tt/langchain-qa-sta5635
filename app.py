# QA Learning Chat Bot
import streamlit as st
import os
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# # Load environment variables
# from dotenv import load_dotenv

# load_dotenv()


def get_open_ai_chat_response(query, t):
    embeddings = CohereEmbeddings(model="embed-english-v3.0")
    vectorstore = Pinecone.from_existing_index(
        index_name=os.getenv('PINECONE_INDEX_NAME'), embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # RAG prompt
    template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # RAG
    model = ChatOpenAI(temperature=t, model="gpt-4-1106-preview")

    chain = (
        RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()})
        | prompt
        | model
        | StrOutputParser()
    )

    response = chain.invoke(query)

    return response


def main():
    st.set_page_config(
        page_title="STA5635 App")
    st.header(
        "Florida State University - Applied Machine Learning (STA5635) Help App")

    st.sidebar.title("About the App")
    st.sidebar.info(
        """The app is to help students in the STA5635 class at Florida State University with their course work. 
       The app uses custom data from all the lectures provided and performs similarity match with the help of vector db and provides 
       answers using OpenAI's api. To learn more about the app, please visit [my blog](https://moh1tt.vercel.app/blog/STA5635)""")
    input = st.text_input(
        "Ask a question", key="input")

    t = st.slider(
        "Temperature [the closer to 1 the more creative the answers get]", 0.0, 1.0, 0.5, 0.1)
    response = get_open_ai_chat_response(input, t)

    submit = st.button("Ask")

    if submit:
        st.subheader("The Response is:")
        st.write(response)


if __name__ == "__main__":
    main()
