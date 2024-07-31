import streamlit as st
import tiktoken
import os

import dotenv
dotenv.load_dotenv()

from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain_community.chat_models import ChatOllama

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain.retrievers import BM25Retriever

from bs4 import BeautifulSoup
import requests

from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate

# from streamlit_chat import message
from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory


def main():
    st.set_page_config(
        page_title="DirChat",
        page_icon=":books:"
    )

    st.title("_컴공 :red[AI 조교]_ :books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your file", type=['pdf'], accept_multiple_files=True)
        url = st.text_input("Enter URL to scrape")
        process = st.button("Process")

    if process:
        if not uploaded_files and not url:
            st.info("Please upload files or provide a URL to continue.")
            st.stop()
        files_text = get_text(uploaded_files)
        if url:
            web_text = get_web_content(url)
            files_text.append(web_text)
        
        text_chunks = get_text_chunks(files_text)
        vectorstore = get_vectorstore(text_chunks)

        st.session_state.conversation = get_conversation_chain(vectorstore)

        st.session_state.processComplete = True

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant",
                                         "content": "컴공 AI 조교입니다^^ 무엇을 도와드릴까요?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        if st.session_state.conversation is not None:
            with st.chat_message("assistant"):
                chain = st.session_state.conversation

                with st.spinner("Thinking..."):
                    result = chain({"question": query})
                    response = result['answer']
                    source_documents = result.get('source_documents', [])

                    st.markdown(response)
                    if source_documents:
                        with st.expander("참고 문서 확인"):
                            for doc in source_documents:
                                st.markdown(doc.metadata['source'], help=doc.page_content)

                # Add assistant message to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            st.error("Conversation chain is not set. Please process the files and/or URL first.")

        # 일반적인 대화 기능을 위한 응답 생성
        if st.session_state.conversation is None:
            llm = ChatOllama(model="EEVE-Korean-10.8B:latest", local=True)
            prompt = ChatPromptTemplate.from_template(
                "Your name is '컴공AI' 너는 항상 반말을 하는 챗봇이야. 다나까나 요 같은 높임말로 절대로 끝내지 마 항상 반말로 친근하게 대답해줘 "
                "영어로 질문을 받아도 무조건 한글로 답변해줘 한글이 아닌 답변일 때는 다시 생각해서 꼭 한글로 만들어줘 "
                "모든 답변 시작 끝에는 웃는 이모티콘을 추가해줘 그리고 설명은 최대 3줄까지 설명가능해 3줄이 넘을 것 같으면 요약해서 설명해줘"
            )

            conversation_chain = LLMChain(llm=llm, prompt=prompt)
            with st.spinner("Thinking..."):
                result = conversation_chain.run({"question": query})
                st.markdown(result)
                st.session_state.messages.append({"role": "assistant", "content": result})


def get_web_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    content = soup.get_text()
    return Document(page_content=content)


def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)


def get_text(docs):
    doc_list = []

    for doc in docs:
        file_name = doc.name  # doc 객체의 이름을 파일 이름으로 사용
        with open(file_name, "wb") as file:  # 파일을 doc.name으로 저장
            file.write(doc.getvalue())
            print(f"Uploaded {file_name}")
        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
            doc_list.extend(documents)
    return doc_list


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks


def get_vectorstore(text_chunks):
    model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb


def get_conversation_chain(vectorstore):
    llm = ChatOllama(model="EEVE-Korean-10.8B:latest", local=True)
    prompt = ChatPromptTemplate.from_template(
        "Your name is '컴공AI' 너는 항상 반말을 하는 챗봇이야. 다나까나 요 같은 높임말로 절대로 끝내지 마 항상 반말로 친근하게 대답해줘 "
        "영어로 질문을 받아도 무조건 한글로 답변해줘 한글이 아닌 답변일 때는 다시 생각해서 꼭 한글로 만들어줘 "
        "모든 답변 시작 끝에는 웃는 이모티콘을 추가해줘 그리고 설명은 최대 3줄까지 설명가능해 3줄이 넘을 것 같으면 요약해서 설명해줘"
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )

    return conversation_chain


if __name__ == '__main__':
    main()
