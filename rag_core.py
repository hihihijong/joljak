import streamlit as st  # 웹 인터페이스 제어 및 에러 메시지 출력
import openai  # OpenAI API 사용 (GPT 모델 연동)
import numpy as np  # 벡터 연산 및 배열 처리
from sklearn.metrics.pairwise import cosine_similarity  # 벡터 간 유사도 계산 (검색 엔진 핵심)
import io  # 파일 바이트 처리를 위한 라이브러리
from PyPDF2 import PdfReader  # PDF 파일 텍스트 추출 라이브러리
import nltk  # 자연어 처리 (문장 분리 등) 라이브러리

# --- NLTK 데이터 다운로드 ---
# 텍스트를 문장 단위로 정확하게 자르기 위해 'punkt' 토크나이저 데이터가 필요합니다.
# 실행 환경에 데이터가 없으면 자동으로 다운로드합니다.
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# --- OpenAI 클라이언트 초기화 ---
# 환경 변수에서 API 키를 로드하여 클라이언트를 생성합니다.
# 키가 없거나 오류 발생 시 앱을 중단하고 경고를 띄웁니다.
try:
    client = openai.OpenAI()
except openai.AuthenticationError:
    st.info("시작하려면 'OPENAI_API_KEY' 환경 변수를 설정해야 합니다.")
    st.stop()
except Exception as e:
    st.error(f"OpenAI 클라이언트 초기화 중 오류: {e}")
    st.stop()

# --- RAG 기본 프롬프트 ---
# AI에게 페르소나와 답변 규칙을 부여합니다.
# 검색된 문서(Context)와 사용자 질문(Question)이 이곳에 주입됩니다.
DEFAULT_RAG_PROMPT = """ 당신은 업로드된 [문서]의 내용을 바탕으로 답변하는 AI 챗봇입니다.
[문서]의 내용만으로 [질문]에 답해야 합니다.

[문서]
{retrieved_context}
---

[질문]
{user_prompt}

---
[기본 규칙]
1. [문서] 내용을 바탕으로 [질문]에 대해 최대한 자세히 설명해.
2. 답변할 때 "[문서]에 따르면", "제공된 정보에 의하면" 같은 표현은 절대 사용하지 마. 마치 네가 원래 알고 있는 지식인 것처럼 자연스럽게 답변해.
3. [문서]에 질문과 **정확히 같은 단어가 없더라도**, 문맥상 의미가 통하는 내용(동의어, 유의어, 상위/하위 개념)이 있다면 찾아서 답변해.
   - 예시: 사용자가 '기숙사'를 물었는데 문서에 '생활관'이 있다면, '생활관' 정보를 답변해.
   - 예시: 사용자가 '학식'을 물었는데 문서에 '식당 메뉴'가 있다면, 그 내용을 답변해.
4. 질문의 단어가 오타이거나 줄임말(약어)이어도, 문서의 내용과 매칭하여 유연하게 해석해.
5. 만약 [문서]와 전혀 관련이 없는 내용이라면 그때만 "관련 정보를 찾을 수 없습니다"라고 답해.

[언어 규칙]
- 질문이 **한국어**인 경우: **한국어**로 답변해.
- 질문이 **영어**인 경우: [문서]의 내용(한국어)을 **영어**로 번역하여 답변해.

[이미지 출력 규칙]
- 답변과 관련된 이미지를 보여줘야 한다면, 답변 끝에 `[IMAGE: 파일명.확장자]` 형식을 추가해.
- 예시: "학교 전경입니다. [IMAGE: school_view.jpg]"
- (주의: 이미지 파일은 실제로 존재하는 파일명이어야 함)

[추가 지시사항]
{user_instructions} """

# --- 파일 처리 함수들 ---

def extract_text_from_pdf(file_bytes):
    """
    PDF 파일의 바이너리 데이터를 받아 텍스트만 추출합니다.
    PyPDF2를 사용하여 각 페이지의 텍스트를 순회하며 합칩니다.
    """
    try:
        pdf_file = io.BytesIO(file_bytes)
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"PDF 텍스트 추출 중 오류 발생: {e}")
        return None

def extract_text_from_txt(file_bytes):
    """
    TXT 파일의 텍스트를 추출합니다.
    한글 인코딩 호환성을 위해 utf-8 시도 후 실패 시 euc-kr로 재시도합니다.
    """
    try:
        return file_bytes.decode('utf-8')
    except UnicodeDecodeError:
        try:
            return file_bytes.decode('euc-kr')
        except Exception as e:
            st.error(f"TXT 텍스트 추출 중 오류 발생: {e}")
            return None

def split_text_into_chunks(text, chunk_size=300, overlap_sentences=2):
    """
    [핵심 로직: 청킹]
    긴 텍스트를 의미 단위(문장)를 유지하며 작은 조각(chunk)으로 나눕니다.
    
    - chunk_size: 한 조각의 최대 글자 수 목표치
    - overlap_sentences: 문맥 단절을 막기 위해 앞뒤 청크끼리 겹치는 문장 수
    """
    if text is None or text.strip() == "":
        return []
    
    # 1. 문장 단위로 분리 (NLTK 사용)
    # NLTK(Natural Language Toolkit): 자연어 처리를 위한 파이썬 패키지
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        return []

    chunks = []
    current_chunk_sentences = []
    current_length = 0

    for i, sentence in enumerate(sentences):
        sentence_length = len(sentence)
        
        # 현재 청크에 문장을 더해도 크기 제한을 넘지 않으면 추가
        if current_length + sentence_length + (len(current_chunk_sentences) > 0) <= chunk_size:
            current_chunk_sentences.append(sentence)
            current_length += sentence_length + 1
        
        # 크기 제한을 넘으면 현재 청크 저장 후 새 청크 시작
        else:
            if current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))
            
            # (예외) 문장 하나가 chunk_size보다 크면 그냥 자름
            if sentence_length > chunk_size:
                chunks.append(sentence[:chunk_size])
                current_chunk_sentences = []
                current_length = 0
            else:
                # [Overlap 로직] 이전 청크의 끝부분 문장을 가져와 새 청크의 시작으로 삼음
                start_index = max(0, len(current_chunk_sentences) - overlap_sentences)
                current_chunk_sentences = current_chunk_sentences[start_index:]
                
                current_chunk_sentences.append(sentence)
                current_length = len(" ".join(current_chunk_sentences)) + (len(current_chunk_sentences) -1)

    # 남은 문장 처리
    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))

    return chunks

# --- RAG 핵심 함수들 (임베딩 & 검색) ---

@st.cache_data
def get_kb_embeddings(documents):
    """
    문서 청크 리스트를 받아 OpenAI Embeddings API를 통해 벡터(숫자 배열)로 변환합니다.
    Streamlit 캐싱(@st.cache_data)을 사용하여 비용과 시간을 절약합니다.
    """
    st.write(f"{len(documents)}개 청크 임베딩을 계산 중입니다...")
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=documents
        )
        return [embedding_object.embedding for embedding_object in response.data]
    except Exception as e:
        st.error(f"KB 임베딩 중 오류 발생: {e}")
        return None

def get_query_embedding(query):
    """
    사용자의 질문(Query)을 벡터로 변환합니다. (검색을 위해 문서 벡터와 비교용)
    """
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        embedding_vector = response.data[0].embedding
        return np.array(embedding_vector).reshape(1, -1)
    except Exception as e:
        st.error(f"쿼리 임베딩 중 오류 발생: {e}")
        return None

def retrieve_documents(query_embedding, kb_embeddings_list, kb_chunks, k=2):
    """
    [검색 로직] 질문 벡터와 문서 벡터들 간의 코사인 유사도를 계산하여
    가장 관련성 높은 상위 k개의 문서 청크를 반환합니다.
    """
    if query_embedding is None or not kb_embeddings_list:
        return "문서 검색에 실패했습니다."
    
    doc_embeddings = np.array(kb_embeddings_list)
    
    # 코사인 유사도 계산 (값이 클수록 유사함)
    # 코사인 유사도: 텍스트의 길이에 영향을 덜 받으면서 의미적 유사도를 빠르고 정확하게 판단, 텍스트 검색 분야에서 표준적으로 사용되는 방식
    similarities = cosine_similarity(query_embedding, doc_embeddings)
    
    # 유사도 순으로 정렬하여 상위 k개 인덱스 추출
    top_k_indices = np.argsort(similarities[0])[-k:][::-1]
    
    # 인덱스에 해당하는 텍스트 청크 추출
    retrieved_docs_content = [kb_chunks[i] for i in top_k_indices]
    
    return "\n\n".join(retrieved_docs_content)

def get_openai_response(prompt): 
    """
    [생성 로직 - 스트리밍]
    완성된 프롬프트를 GPT-4o에 전송하고, 답변을 타자기처럼 실시간으로 반환(yield)합니다.
    Generator 기능을 사용하여 app.py의 st.write_stream과 연동됩니다.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o", 
            messages=[{"role": "user", "content": prompt}],
            stream=True  # 스트리밍 활성화
        )
        
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content # 조각 데이터를 즉시 반환
                
    except Exception as e:
        yield f"OpenAI API 호출 중 오류 발생: {e}"