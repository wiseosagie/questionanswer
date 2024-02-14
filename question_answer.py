from langchain_core.vectorstores import VectorStoreRetriever
import openai
import streamlit as st 
from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import chroma
from langchain.vectorstores.chroma import Chroma
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders import SeleniumURLLoader
import os 


def load_url(url):
    print(url)
    from langchain_community.document_loaders import UnstructuredURLLoader
    
    loader = UnstructuredURLLoader(urls=url)
    data = loader.load()
    print(data)
    return data

def load_document(file):
    import os
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'loading {file}')
        loader = PyPDFLoader(file)
    # elif extension == '.docx':
    #     #from langchain.document_loaders import DocxtxtLoader
    #     from langchain_community.document_loaders import Docx2txtLoader
    #     print(f'loading {file}')
    #     loader = Docx2txtLoader(file)
    elif extension == '.csv':
        from langchain_community.document_loaders.csv_loader import CSVLoader
        print(f'loading {file}')
        loader = CSVLoader(file)
    elif extension == '.txt':
        from langchain_community.document_loaders import TextLoader
        print(f'loading {file}')
        loader = TextLoader(file)
    else:
        from langchain_community.document_loaders import UnstructuredFileLoader
        loader = UnstructuredFileLoader(file)
        # print('Document format is not supported!')
        # return None
    data = loader.load()
    print(data[0].page_content[:4000])
    return data



def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks


def create_embeddings(chunks):
    if chunks == []:
        st.error("The file was not uploaded successfully, the file may contain images and not text. Try a different file.")
        exit()
        
    else:
        embeddings = OpenAIEmbeddings()
        vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store

# Get answer to questions
def ask_and_get_answer(vector_store, q, k):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':k})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)
    # chain = RetrievalQA.from_llm(llm=openai(), retriever=retriever)

    answer = chain.run(q)
    return answer

# Calculate Cost
def print_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    # print(f'Total tokens: {total_tokens}')
    # print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.004:.6f}')
    return total_tokens, total_tokens / 1000 * 0.004

#
def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('davinci-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    # print(f'Total tokens: {total_tokens}')
    # print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.004:.6f}')
    return total_tokens, (total_tokens / 1000 * 0.004)

def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

    # Custom CSS to set the background video of the main body

    st.set_page_config(layout="wide")

    video_html = """
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

            #myVideo {
            position: fixed;
            right: 0;
            bottom: 0;
            min-width: 100%; 
            min-height: 100%;
            }

            .content {
            position: fixed;
            bottom: 0;
            background: rgba(0, 0, 0, 0.5);
            color: #f1f1f1;
            width: 100%;
            padding: 20px;
            }
            </style>
            
            	
            
            """


    st.markdown(video_html, unsafe_allow_html=True)
    
# <source src="C:\Users\wiseo\Downloads\tech_bg.mp4">
    # <source type="video/mp4" src="tech_bg.mp4">

     
# <video autoplay muted loop id="myVideo">
#                 <source src="https://videos.pond5.com/binary-digital-tech-data-code-footage-009784511_main_xxl.mp4">
#             Your browser does not support HTML5 video.
#             </video>



    # Custom CSS to set the background image of the main body
    
#     st.markdown(
#         """
#         <style>
#             [data-testid=stAppViewContainer] {
#                 background: url("https://images.unsplash.com/photo-1498747946579-bde604cb8f44?q=80&w=1932&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
#                 background-size: cover;
#                 }
#             [data-testid="stHeader"]{
#                 background-color:rgba(0,0,0,0);
#             }

#         </style>
#         """,
#             unsafe_allow_html=True,
# )
    
    
    # st.markdown("""
    #     <style>
    #         [data-testid=stSidebar] {
    #             background-color: #87CEEB;
    #         }
                    
    #         [data-testid=StyledLinkIconContainer] {
    #         font-weight: bold;
    #         color: green;
    #         font-size: 40px;    
    #     }
    #              [data-testid=stWidgetLabel] {
    #         font-weight: bold;
    #         color: white;
    #         font-size: 3rem !important;
    #         font-weight: bold;
    #     }
                
    #     </style>
    #     """, unsafe_allow_html=True)
    
    # Custom CSS to set the label color of the text input

   
    





    st.header('Question-Answering Application', divider='green')
    with st.sidebar:

        # api_key = st.text_input('OpenAI API Key:', type='password')
        # if api_key:
        #     os.environ['OPENAI_API_KEY'] = api_key
            # st.success("Api key entered")
        # else:
        #     st.error("API key is missing. Please provide a valid API key.")
        chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=512, on_change=clear_history)
        k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)
        st.divider()
        url_un = st.text_input('Enter URL:')
        check_url_data = st.button('Check URL Data', on_click=clear_history)
        st.divider()
        uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt', 'csv'])
        add_data = st.button('Add Data', on_click=clear_history)
       
        
        #Get the text data from URL
        # url = ["https://www.wichita.edu/services/its/userservices/documents/ITS_Computer_Purchase_Policy_02-22-19.pdf"]
        url=[url_un]
        if check_url_data:
            with st.spinner('Reading, chunking and embedding file ....'):
                from langchain_community.document_loaders import UnstructuredURLLoader
                loader = UnstructuredURLLoader(urls=url)
                data = loader.load()
                chunks = chunk_data(data, chunk_size=chunk_size)
                # st.write(f' Chunk Size : {chunk_size}, Chunks: {len(chunks)}')
                st.write(f"<p style='color: red; font-size: 20px; font-weight: bold;'>Chunk Size: {chunk_size}, Chunks: {len(chunks)}</p>", unsafe_allow_html=True)
                tokens, embedding_cost = calculate_embedding_cost(chunks)
                # st.write(f'Embedding cost: ${embedding_cost: .4f}')
                st.write(f"<p style='color: green; font-size: 20px; font-weight: bold;'>Embedding cost: ${embedding_cost: .4f}</p>", unsafe_allow_html=True)
                vector_store = create_embeddings(chunks)
                st.session_state.vs = vector_store
                st.success("The file has been uploaded, chunked, and embedded successfully")


        if uploaded_file and add_data:
            with st.spinner('Reading, chunking and embedding file ....'):
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)
                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size=chunk_size)
                # st.write(f' Chunk Size : {chunk_size}, Chunks: {len(chunks)}')
                st.write(f"<p style='color: red; font-size: 20px; font-weight: bold;'>Chunk Size: {chunk_size}, Chunks: {len(chunks)}</p>", unsafe_allow_html=True)
                tokens, embedding_cost = calculate_embedding_cost(chunks)
                # st.write(f'Embedding cost: ${embedding_cost: .4f}')
                st.write(f"<p style='color: green; font-size: 20px; font-weight: bold;'>Embedding cost: ${embedding_cost: .4f}</p>", unsafe_allow_html=True)
                vector_store = create_embeddings(chunks)
                st.session_state.vs = vector_store
                st.success("The file has been uploaded, chunked, and embedded successfully")
    
    q = st.text_input('Lets help you answer questions about your document:', placeholder='Ask a question')
    if q:
        if 'vs' in st.session_state:
            vector_store = st.session_state.vs
            # st.write(f'k: {k}')
            answer = ask_and_get_answer(vector_store, q, k)
            st.text_area('LLM Answer', value=answer)
    
            st.divider()
            if 'history' not in st.session_state:
                st.session_state.history = ''
            value = f'Q: {q} \nA: {answer}'
            st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
            h = st.session_state.history
            st.text_area(label='Chat History', value=h, key='history', height=400)
















    