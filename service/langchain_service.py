from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone as LangChainPinecone
from dotenv import load_dotenv
import os

# .env 파일을 로딩
load_dotenv()

# Pinecone setting
index = os.getenv("PINECONE_INDEX")
environment = os.getenv("PINECONE_ENVIRONMENT")

# genai setting
api_key = os.getenv('GOOGLE_API_KEY')


# 문서를 vector store로 저장하는 함수
def create_vectorstore(documents):
    # HuggingFaceEmbeddings 모델 사용
    embeddings_model = HuggingFaceEmbeddings(
        model_name='jhgan/ko-sbert-nli',
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True},
    )

    # Pinecone에 저장
    vectorstore = LangChainPinecone.from_documents(documents, embeddings_model, index_name=index)
    return vectorstore


def create_embeddings(documents):
    # 임베딩 모델 생성
    embeddings_model = GoogleGenerativeAIEmbeddings(model='models/embedding-001', api_key=api_key)

    # Document 객체로 변환
    documents_as_objects = [Document(page_content=doc) for doc in documents]

    # Pinecone 인덱스 객체 생성
    vectorstore = LangChainPinecone.from_documents(
        documents_as_objects,  # Document 객체 리스트
        embeddings_model,  # 임베딩 모델
        index_name=index  # Pinecone 인덱스 이름
    )

    return vectorstore
