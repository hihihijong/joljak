import streamlit as st
import os
import re
import rag_core as core  # 작성한 rag_core 모듈 임포트

# --- 페이지 설정 ---
st.set_page_config(page_title="신입생 가이드", page_icon="🏫", layout="wide")
st.title("🏫 용인대학교 신입생을 위한 가이드 챗봇")

# --- 유틸리티 함수 ---
def display_message_with_images(role, content):
    """
    메시지 내용 중 [IMAGE: ...] 태그를 감지하여 텍스트 아래에 이미지를 출력하는 함수
    """
    with st.chat_message(role):
        # 정규표현식으로 이미지 태그 추출
        image_matches = re.findall(r"\[IMAGE:\s*(.*?)\]", content)
        # 텍스트 출력 시에는 태그 제거 (깔끔하게 보이기 위함)
        clean_text = re.sub(r"\[IMAGE:\s*.*?\]", "", content)
        st.markdown(clean_text)
        
        # 이미지가 있다면 파일 존재 확인 후 출력
        if image_matches:
            for image_name in image_matches:
                image_path = image_name.strip()
                if os.path.exists(image_path):
                    st.image(image_path, caption=image_path)
                else:
                    st.warning(f"⚠️ 이미지 파일을 찾을 수 없습니다: {image_path}")

# --- 사이드바: 설정 및 모드 선택 ---
st.sidebar.header("📌 주제 선택")

# [주제 선택] 라디오 버튼으로 '학교 전체' vs '학과 전공' 모드 전환
topic_mode = st.sidebar.radio(
    "어떤 것에 대해 궁금한가요?",
    ["🏫 용인대학교 (전체)", "📖 AI융합학부 (전공)"],
    index=0
)

st.sidebar.markdown("---")

# 사용자 추가 지시사항 (프롬프트 인젝션)
if 'user_instructions' not in st.session_state:
    st.session_state.user_instructions = ""

st.sidebar.subheader("추가 지시사항")
st.sidebar.text_area(
    "AI에게 추가로 요청할 사항:",
    key="user_instructions",
    height=100,
    placeholder="예) 답변을 짧게 3줄로 요약해줘."
)

# --- 파일 목록 설정 ---
# 주제별로 참조할 파일명을 다르게 설정
FILES_UNIV = ["yonginuniv.txt"]   
FILES_AI   = ["aihakbu.txt", "yogeon.txt"]   

# 선택된 모드에 따라 파일 목록, 캐시 키, 시스템 페르소나 설정
if topic_mode == "🏫 용인대학교 (전체)":
    current_files = FILES_UNIV
    cache_key = "univ" # 세션 상태 분리를 위한 키
    system_persona = "당신은 '용인대학교'의 학교 생활 전반을 안내하는 멘토입니다."
else:
    current_files = FILES_AI
    cache_key = "ai"
    system_persona = "당신은 'AI융합학부'의 교육과정과 전공 정보를 안내하는 조교입니다."

# --- 세션 상태(저장소) 초기화 ---
# 모드별(univ/ai)로 청크와 임베딩 데이터를 따로 저장하여 혼선을 방지함
# 듀얼 KB(Knowledge Base)시스템을 구축
if f'kb_chunks_{cache_key}' not in st.session_state:
    st.session_state[f'kb_chunks_{cache_key}'] = []
if f'kb_embeddings_{cache_key}' not in st.session_state:
    st.session_state[f'kb_embeddings_{cache_key}'] = []

# --- 데이터 로딩 및 임베딩 (최초 1회 실행) ---
# 해당 주제의 임베딩 데이터가 없을 경우 파일 읽기 시작
if current_files and (not st.session_state[f'kb_embeddings_{cache_key}']):
    all_texts = []
    
    with st.spinner(f"'{topic_mode}' 관련 지식을 배우는 중입니다..."):
        for filename in current_files:
            try:
                with open(filename, "rb") as f:
                    file_bytes = f.read()
                
                text = None
                ext = os.path.splitext(filename)[1].lower()
                
                # 파일 확장자에 따른 텍스트 추출
                if ext == ".pdf":
                    text = core.extract_text_from_pdf(file_bytes)
                elif ext == ".txt":
                    text = core.extract_text_from_txt(file_bytes)
                
                if text:
                    all_texts.append(text)
                else:
                    st.sidebar.warning(f"'{filename}' 텍스트 추출 실패")
            except FileNotFoundError:
                st.sidebar.error(f"⚠️ 파일 없음: {filename}")
            except Exception as e:
                st.sidebar.error(f"오류 ({filename}): {e}")

    # 텍스트가 준비되면 청크 분할 및 임베딩 생성
    if all_texts:
        combined_text = "\n\n".join(all_texts)
        
        # [정확도 튜닝] chunk_size=300, overlap=2로 작고 촘촘하게 분할
        chunks = core.split_text_into_chunks(combined_text, chunk_size=300, overlap_sentences=2)
        st.session_state[f'kb_chunks_{cache_key}'] = chunks
        
        if chunks:
            embeddings = core.get_kb_embeddings(chunks)
            st.session_state[f'kb_embeddings_{cache_key}'] = embeddings
            st.sidebar.success(f"학습 완료! ({len(chunks)}개 지식)")
        else:
            st.sidebar.error("청크 생성 실패")

# --- 현재 모드의 데이터 로드 ---
current_kb_chunks = st.session_state.get(f'kb_chunks_{cache_key}', [])
current_kb_embeddings = st.session_state.get(f'kb_embeddings_{cache_key}', [])
rag_ready = len(current_kb_embeddings) > 0

# 상단 상태 알림창
if rag_ready:
    st.info(f"현재 **[{topic_mode}]** 모드로 대화 중입니다.")
else:
    st.warning("⚠️ 해당 주제의 학습 파일이 없거나 로딩되지 않았습니다.")

# --- 채팅 인터페이스 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 대화 기록 표시
for message in st.session_state.messages:
    display_message_with_images(message["role"], message["content"])

# [사용자 입력 처리]
if user_prompt := st.chat_input("궁금한 내용을 물어보세요..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    display_message_with_images("user", user_prompt)
    
    response = ""
    
    if rag_ready:
        with st.spinner(f"[{topic_mode}] 관련 문서 검색 중..."):
            
            # [쿼리 확장] 사용자의 질문이 짧을 경우(단어 검색 등), 검색 품질 향상을 위해 쿼리 보강
            # 쿼리 확장: 사용자의 원래 질문을 AI가 이해하기 쉬운 구체적인 형태나 유사한 문장으로 재구성하는 기술
            search_query = user_prompt
            if len(search_query) < 10:
                search_query = f"{user_prompt} {user_prompt}의 뜻, 정의, 역할, 관련 내용에 대해 상세히 설명해줘."
            
            # 1. 질문 임베딩 생성
            query_embedding = core.get_query_embedding(search_query)
            
            # 2. 관련 문서 검색 (k=12로 풍부하게 검색)
            # KNN(K-Nearest Neighbors) 알고리즘
            retrieved_context = core.retrieve_documents(
                query_embedding, 
                current_kb_embeddings, 
                current_kb_chunks,     
                k=12    #k=12가 의미 하는 바 : 검색된 결과 중 상위 12등까지만 가져와라
            )
            
            # 3. 프롬프트 조립 (Context + Question + Persona)
            prompt_template = core.DEFAULT_RAG_PROMPT
            final_instructions = f"{system_persona}\n{st.session_state.user_instructions}"
            
            augmented_prompt = prompt_template.format(
                retrieved_context=retrieved_context,
                user_prompt=user_prompt,
                user_instructions=final_instructions
            )
            
            # 4. 스트리밍 방식으로 AI 답변 생성
            stream = core.get_openai_response(augmented_prompt)
            
            # st.write_stream으로 타자기 효과 구현
            with st.chat_message("assistant"):
                response = st.write_stream(stream)
            
            # 5. 답변 완료 후 이미지 태그 처리
            image_matches = re.findall(r"\[IMAGE:\s*(.*?)\]", response)
            if image_matches:
                for image_name in image_matches:
                    image_path = image_name.strip()
                    if os.path.exists(image_path):
                        st.image(image_path, caption=image_path)

    else:
        # RAG 준비가 안 되었을 때의 예외 처리
        response = "해당 주제의 데이터가 준비되지 않았습니다."
        st.warning(response)
        display_message_with_images("assistant", response)

    # 대화 기록 저장
    st.session_state.messages.append({"role": "assistant", "content": response})