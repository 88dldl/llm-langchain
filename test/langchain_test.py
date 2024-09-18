import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone
from dotenv import load_dotenv


# .env 파일을 로딩
load_dotenv()

# Pinecone 설정
index_name = os.getenv("PINECONE_INDEX")
api_key = os.environ.get("PINECONE_API_KEY")

# Pinecone 클라이언트 설정
pc = Pinecone(api_key=api_key)
index = pc.Index(index_name)


def test_hugging_vectorstore(vectorstore, query_text):
    # 검색 쿼리
    embeddings_model = HuggingFaceEmbeddings(
        model_name='jhgan/ko-sbert-nli',
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True},
    )
    query_vector = embeddings_model.embed_query(query_text)

    # Pinecone에서 유사도 검색
    results = index.query(vector=query_vector, top_k=5)  # top_k는 반환할 결과 수

    # 결과 출력
    for result in results['matches']:
        print(result)



def test_genai_vectorstore(index, query_text):
    # 임베딩 모델 생성
    embeddings_model = GoogleGenerativeAIEmbeddings(model='models/embedding-001', api_key=api_key)

    # 쿼리 벡터 생성
    query_vector = embeddings_model.embed_query(query_text)

    # Pinecone에서 유사도 검색
    results = index.query(vector=query_vector, top_k=5)  # top_k는 반환할 결과 수

    # 결과 출력
    for result in results['matches']:
        print(result)
