from langchain_community.embeddings import logger
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from db.database import get_info

def fetch_and_process_data():
    # db 정보 불러오기
    df = get_info()
    info_list = df['info'].tolist()
    logger.info("Fetched %d entries from database", len(info_list))

    # 문자열을 Document 객체로 변환
    documents = [Document(page_content=info) for info in info_list]

    # 문서를 청크로 나누기
    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    documents = text_splitter.split_documents(documents)
    logger.info("Split %d documents", len(documents))

    return documents