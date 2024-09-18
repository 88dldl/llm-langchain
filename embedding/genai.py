import os
from dotenv import load_dotenv
import pinecone
from pinecone import Pinecone

from db.database_service import fetch_and_process_data
from service.langchain_service import create_embeddings
from test.langchain_test import test_genai_vectorstore

# .env 파일을 로딩
load_dotenv()

pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
)

# Pinecone 인덱스 객체 생성
index_name = os.getenv("PINECONE_INDEX")
index = pc.Index(index_name)

if __name__ == "__main__":
    # 데이터 불러오기 및 처리
    documents = fetch_and_process_data()

    # Document 객체에서 문자열만 추출
    texts = [doc.page_content for doc in documents]

    # 벡터스토어 생성 및 파이콘에 저장
    vectorstore = create_embeddings(texts)

    # 쿼리 테스트
    query_text = '스프링부트 공부하는 info 추천해줘'
    results = test_genai_vectorstore(index, query_text)

