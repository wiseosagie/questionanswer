import os
import re
import pandas as pd
import streamlit as st
from dotenv import load_dotenv, find_dotenv
import openai
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain_community.document_loaders import (
    UnstructuredFileLoader,
    SeleniumURLLoader,
    WebBaseLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import validators
import tiktoken

# Load environment variables
load_dotenv(find_dotenv(), override=True)

def load_url(url):
    loader = SeleniumURLLoader(urls=[url])
    data = loader.load()
    return data

def chunk_data(data, chunk_size=256, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks

def create_embeddings(chunks):
    if chunks:
        embeddings = OpenAIEmbeddings()
        vector_store = Chroma.from_documents(chunks, embeddings)
        return vector_store
    else:
        st.error("The file was not uploaded successfully. Try a different file.")
        return None

def ask_and_get_answer(vector_store, q, k):
    llm = ChatOpenAI(model='gpt-4', temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)
    answer = chain.run(q)
    return answer

def verify_api_key(api_key):
    openai.api_key = api_key
    try:
        response = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",
            prompt="This is a test request to verify the OpenAI API key.",
            max_tokens=5
        )
    except openai.error.AuthenticationError:
        st.error("Your API key is invalid. Please re-enter the API key and refresh the page.")
        st.experimental_rerun()
    except Exception as e:
        st.error(f"An error occurred: {e}")

def verify_url(url):
    if not isinstance(url, str) or len(url) < 4:
        return False
    if not validators.url(url):
        return False
    return True

def calculate_embedding_cost(texts):
    enc = tiktoken.encoding_for_model('davinci-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    return total_tokens, total_tokens / 1000 * 0.004

def extract_contact_info(text):
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    phones = re.findall(r'\b\d{10}\b|\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', text)
    return emails, phones

def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']

# Ensure the directory exists
output_dir = '/mnt/data'
os.makedirs(output_dir, exist_ok=True)

# Streamlit UI
st.set_page_config(layout="wide")
st.markdown("""
    <style>
    [data-testid=stSidebar] {
        background-color: #87CEEB;
    }
    [data-testid=StyledLinkIconContainer] {
        font-weight: bold;
        color: green;
        font-size: 40px;
    }
    [data-testid=stWidgetLabel] {
        font-weight: bold;
        color: white;
        font-size: 3rem !important;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.header('Question-Answering Application', divider='green')

with st.sidebar:
    api_key = st.text_input('OpenAI API Key:', type='password')
    chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=512, on_change=clear_history)
    k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)
    st.divider()
    uploaded_file = st.file_uploader('Upload an Excel file with URLs:', type=['xlsx'])
    process_urls = st.button('Process URLs', on_click=clear_history)

    if process_urls:
        if not api_key:
            st.error("API key is missing. Please provide a valid API key.")
        else:
            os.environ['OPENAI_API_KEY'] = api_key
            verify_api_key(api_key)
            if uploaded_file:
                with st.spinner('Processing URLs...'):
                    df = pd.read_excel(uploaded_file)
                    if 'Website' not in df.columns:
                        st.error("The Excel sheet must contain a 'Website' column.")
                    else:
                        total_tokens = 0
                        emails_list = []
                        phones_list = []
                        for url in df['Website']:
                            if not verify_url(url):
                                emails_list.append('N/A')
                                phones_list.append('N/A')
                                continue

                            data = load_url(url)
                            content = ' '.join([doc.page_content for doc in data])
                            emails, phones = extract_contact_info(content)
                            emails_list.append(', '.join(emails) if emails else 'N/A')
                            phones_list.append(', '.join(phones) if phones else 'N/A')
                            
                            # Calculate tokens for embedding cost
                            chunks = chunk_data(data, chunk_size=chunk_size)
                            tokens, _ = calculate_embedding_cost(chunks)
                            total_tokens += tokens

                        df['Extracted Emails'] = emails_list
                        df['Extracted Phones'] = phones_list

                        total_cost = total_tokens / 1000 * 0.004
                        num_emails = sum([1 for email in emails_list if email != 'N/A'])
                        num_phones = sum([1 for phone in phones_list if phone != 'N/A'])

                        output_path = os.path.join(output_dir, 'processed_urls_with_contacts.xlsx')
                        df.to_excel(output_path, index=False)
                        st.success(f"Processed data saved to {output_path}")
                        st.download_button(
                            label="Download Processed File",
                            data=open(output_path, 'rb').read(),
                            file_name='processed_urls_with_contacts.xlsx'
                        )

                        st.write(f"Total cost of embeddings: ${total_cost:.4f}")
                        st.write(f"Number of emails collected: {num_emails}")
                        st.write(f"Number of phone numbers collected: {num_phones}")
            else:
                st.error("No file uploaded. Please upload an Excel file.")

q = st.text_input('Ask a question about your document:', placeholder='Ask a question')
if q and 'vs' in st.session_state:
    vector_store = st.session_state.vs
    answer = ask_and_get_answer(vector_store, q, k)
    st.text_area('LLM Answer', value=answer)

    if 'history' not in st.session_state:
        st.session_state.history = ''
    value = f'Q: {q}\nA: {answer}'
    st.session_state.history = f'{value}\n{"-" * 100}\n{st.session_state.history}'
    st.text_area(label='Chat History', value=st.session_state.history, height=400)
