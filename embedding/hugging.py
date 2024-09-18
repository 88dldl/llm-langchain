from service.langchain_service import create_vectorstore
from db.database_service import fetch_and_process_data
from test.langchain_test import test_hugging_vectorstore

if __name__ == "__main__":
    # 데이터 불러오기 및 처리
    documents = fetch_and_process_data()

    # 벡터스토어 생성 및 파이콘에 저장
    vectorstore = create_vectorstore(documents)

    # 쿼리 테스트
    query_text = '스프링부트 공부하는 info 추천해줘'
    test_hugging_vectorstore(vectorstore,query_text)