import streamlit as st
import uuid
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage

# 프로젝트 모듈 임포트
from config import Config, setup_langsmith, validate_config
from agent.workflow import create_legal_agent
from utils.conversation import ConversationManager
from utils.logger import log_conversation, log_error, logger
from utils.pdf_processor import get_pdf_processor

# 페이지 설정
st.set_page_config(
    page_title="⚖️ 법률 AI 어시스턴트",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def initialize_app():
    """앱 초기화 (캐시되어 한 번만 실행)"""
    try:
        # 설정 검증
        validate_config()
        
        # LangSmith 설정
        setup_langsmith()
        
        # Agent 초기화
        agent = create_legal_agent()
        
        # Vector DB 강제 초기화 (캐시 시점에 문서 로드 보장)
        from agent.tools import get_vector_db
        vector_db = get_vector_db()
        
        # 문서가 제대로 로드되었는지 확인
        if hasattr(vector_db, 'documents') and vector_db.documents:
            logger.info(f"✅ Streamlit 캐시: {len(vector_db.documents)}개 문서 로드 완료")
        else:
            logger.warning("⚠️ Streamlit 캐시: 문서 로드 실패")
        
        # 대화 관리자 초기화
        conversation_manager = ConversationManager()
        
        logger.info("앱 초기화 완료")
        return agent, conversation_manager
        
    except Exception as e:
        st.error(f"앱 초기화 실패: {e}")
        log_error(e, "app_initialization")
        st.stop()

def initialize_session_state():
    """세션 상태 초기화"""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "conversation_count" not in st.session_state:
        st.session_state.conversation_count = 0
    
    # PDF 관련 세션 상태 추가
    if "uploaded_pdf_text" not in st.session_state:
        st.session_state.uploaded_pdf_text = None
    
    if "pdf_filename" not in st.session_state:
        st.session_state.pdf_filename = None
    
    if "has_pdf" not in st.session_state:
        st.session_state.has_pdf = False

def handle_pdf_upload():
    """PDF 업로드 및 처리"""
    st.subheader("📎 계약서 첨부")
    
    # 파일 업로드 위젯
    uploaded_file = st.file_uploader(
        "계약서 PDF 파일을 업로드하세요",
        type=['pdf'],
        help="업로드된 계약서는 OCR 처리되어 질문 답변에 활용됩니다.",
        key="pdf_uploader"
    )
    
    # 현재 첨부된 파일 정보 표시
    if st.session_state.has_pdf and st.session_state.pdf_filename:
        st.success(f"✅ 첨부된 파일: {st.session_state.pdf_filename}")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.session_state.uploaded_pdf_text:
                text_preview = st.session_state.uploaded_pdf_text[:200] + "..." if len(st.session_state.uploaded_pdf_text) > 200 else st.session_state.uploaded_pdf_text
                st.info(f"📄 내용 미리보기: {text_preview}")
        
        with col2:
            if st.button("🗑️ 파일 제거", type="secondary"):
                st.session_state.uploaded_pdf_text = None
                st.session_state.pdf_filename = None
                st.session_state.has_pdf = False
                st.rerun()
    
    # 새 파일이 업로드된 경우 처리
    if uploaded_file is not None:
        if not st.session_state.has_pdf or st.session_state.pdf_filename != uploaded_file.name:
            
            with st.spinner("📄 PDF 처리 중... (최초 실행시 모델 다운로드로 시간이 소요될 수 있습니다)"):
                try:
                    # PDF 처리기 가져오기
                    pdf_processor = get_pdf_processor()
                    
                    # 파일 정보 표시
                    file_info = pdf_processor.get_file_info(uploaded_file)
                    st.info(f"📁 {file_info['name']} ({file_info['size_mb']} MB)")
                    
                    # OCR 처리
                    success, result, filename = pdf_processor.process_uploaded_file(uploaded_file)
                    
                    if success:
                        # 성공한 경우 세션 상태 업데이트
                        st.session_state.uploaded_pdf_text = result
                        st.session_state.pdf_filename = filename
                        st.session_state.has_pdf = True
                        
                        st.success(f"✅ 처리 완료! {len(result):,} 글자 추출됨")
                        st.balloons()
                        st.rerun()
                        
                    else:
                        # 실패한 경우 간단한 오류 메시지
                        st.error(f"❌ 처리 실패: {result}")
                        
                        # 일반적인 해결 방법 제시
                        with st.expander("💡 해결 방법", expanded=False):
                            st.markdown("""
                            **가능한 원인 및 해결방법:**
                            - 인터넷 연결 확인 후 재시도
                            - PDF 파일이 너무 크거나 이미지 품질이 낮은 경우
                            - 페이지 새로고침 후 다시 업로드
                            """)
                        
                        # 세션 상태 초기화
                        st.session_state.uploaded_pdf_text = None
                        st.session_state.pdf_filename = None
                        st.session_state.has_pdf = False
                        
                except Exception as e:
                    st.error(f"❌ 오류 발생: 파일 처리 중 문제가 발생했습니다.")
                    
                    # 개발자용 로그 (사용자에게는 숨김)
                    with st.expander("🔧 상세 오류 (개발자용)", expanded=False):
                        st.code(str(e))
                    
                    logger.error(f"PDF 업로드 처리 오류: {e}", exc_info=True)

def display_chat_message(message, is_user=True):
    """채팅 메시지 표시"""
    role = "user" if is_user else "assistant"
    avatar = "👤" if is_user else "⚖️"
    
    with st.chat_message(role, avatar=avatar):
        if is_user:
            st.markdown(message)
        else:
            # AI 답변에 대한 추가 정보 표시
            if isinstance(message, dict):
                st.markdown(message.get("content", ""))
                
                # 신뢰도 및 분석 정보 표시 (사이드바 정보용)
                if message.get("show_details", False):
                    with st.expander("🔍 분석 정보", expanded=False):
                        st.write(f"**법률 분야**: {message.get('legal_domain', 'N/A')}")
                        st.write(f"**질문 유형**: {message.get('query_type', 'N/A')}")
                        st.write(f"**신뢰도**: {message.get('confidence', 'N/A'):.2f}")
                        if message.get('keywords'):
                            st.write(f"**핵심 키워드**: {', '.join(message.get('keywords', [])[:5])}")
            else:
                st.markdown(message)

def create_sidebar():
    """사이드바 생성"""
    with st.sidebar:
        st.header("📊 세션 정보")
        st.write(f"**세션 ID**: {st.session_state.session_id}")
        st.write(f"**대화 수**: {st.session_state.conversation_count}")
        st.write(f"**시작 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        # PDF 첨부 상태 표시
        if st.session_state.has_pdf:
            st.write(f"**첨부 파일**: ✅ {st.session_state.pdf_filename}")
        else:
            st.write("**첨부 파일**: ❌ 없음")
        
        st.divider()
        
        # 대화 기록 관리
        st.header("💾 대화 관리")
        
        if st.button("🗑️ 대화 기록 삭제", type="secondary"):
            st.session_state.messages = []
            st.session_state.conversation_count = 0
            # PDF 정보는 유지
            st.rerun()
        
        if st.button("📤 대화 내보내기", type="secondary"):
            if st.session_state.messages:
                try:
                    conversation_manager = st.session_state.conversation_manager
                    export_path = conversation_manager.export_conversations(
                        st.session_state.session_id
                    )
                    if export_path:
                        st.success(f"대화 기록이 저장되었습니다: {export_path}")
                    else:
                        st.warning("내보낼 대화 기록이 없습니다.")
                except Exception as e:
                    st.error(f"내보내기 실패: {e}")
            else:
                st.warning("대화 기록이 없습니다.")
        
        st.divider()
        
        # 설정 정보
        st.header("⚙️ 설정")
        st.write(f"**모델**: {Config.OPENAI_MODEL}")
        st.write(f"**Temperature**: {Config.OPENAI_TEMPERATURE}")
        st.write(f"**LangSmith**: {'✅' if Config.LANGCHAIN_TRACING_V2 else '❌'}")
        
        # 사용 가이드
        st.divider()
        st.header("📋 사용 가이드")
        st.markdown("""
        **기본 질문 예시:**
        - 계약 위반 시 손해배상은?
        - 임대차 보증금 반환 문제
        - 회사 설립 절차가 궁금해요
        - 이혼 시 재산분할은?
        
        **PDF 첨부 시:**
        - "이 계약서에서 위약금 조항을 분석해주세요"
        - "첨부한 계약서의 문제점은?"
        - "일반적인 계약서와 비교해주세요"
        
        **주의사항:**
        - 구체적 법률 자문은 전문가 상담 필요
        - 제공되는 정보는 참고용입니다
        """)

def main():
    """메인 애플리케이션"""
    
    # 앱 초기화
    agent, conversation_manager = initialize_app()
    
    # 세션 상태 초기화
    initialize_session_state()
    
    # conversation_manager를 세션 상태에 저장
    st.session_state.conversation_manager = conversation_manager
    
    # 제목
    st.title("⚖️ 법률 AI 어시스턴트")
    st.markdown("---")
    
    # 사이드바
    create_sidebar()
    
    # 메인 컨테이너
    main_container = st.container()
    
    with main_container:
        # PDF 업로드 영역
        handle_pdf_upload()
        st.markdown("---")
        
        # 환영 메시지 (첫 방문시)
        if not st.session_state.messages:
            welcome_message = """
            ### 👋 안녕하세요! 법률 AI 어시스턴트입니다.
            
            저는 한국 법률에 대한 질문에 답변드리는 AI입니다. 
            궁금한 법률 문제가 있으시면 언제든 질문해주세요!
            """
            
            if st.session_state.has_pdf:
                welcome_message += f"""
            
            📎 **{st.session_state.pdf_filename}** 파일이 첨부되었습니다.
            첨부하신 계약서 내용을 참고하여 더 구체적인 답변을 드릴 수 있습니다.
            """
            else:
                welcome_message += """
            
            💡 **팁**: 계약서 PDF를 첨부하시면 해당 문서를 분석하여 더 정확한 답변을 드릴 수 있습니다.
            """
            
            st.markdown(welcome_message)
            st.markdown("---")
        
        # 이전 대화 표시
        for message in st.session_state.messages:
            if isinstance(message, HumanMessage):
                display_chat_message(message.content, is_user=True)
            elif isinstance(message, AIMessage):
                display_chat_message(message.content, is_user=False)
        
        # 사용자 입력
        placeholder_text = "법률 관련 질문을 입력하세요..."
        if st.session_state.has_pdf:
            placeholder_text = f"'{st.session_state.pdf_filename}' 관련 질문을 입력하세요..."
        
        if prompt := st.chat_input(placeholder_text):
            # 사용자 메시지 추가 및 표시
            user_message = HumanMessage(content=prompt)
            st.session_state.messages.append(user_message)
            display_chat_message(prompt, is_user=True)
            
            # AI 응답 생성
            with st.chat_message("assistant", avatar="⚖️"):
                with st.spinner("🤔 답변을 생성하고 있습니다..."):
                    try:
                        # Agent 실행 - PDF 정보 포함
                        result = agent.invoke({
                            "messages": st.session_state.messages,
                            "session_id": st.session_state.session_id,
                            "uploaded_pdf_text": st.session_state.uploaded_pdf_text,
                            "pdf_filename": st.session_state.pdf_filename,
                            "has_pdf": st.session_state.has_pdf
                        })
                        
                        # 응답 처리
                        ai_response = result.get("final_answer", "죄송합니다. 답변을 생성할 수 없습니다.")
                        
                        # AI 메시지 추가
                        ai_message = AIMessage(content=ai_response)
                        st.session_state.messages.append(ai_message)
                        
                        # 응답 표시
                        st.markdown(ai_response)
                        
                        # 분석 정보 표시 (expandable)
                        if result.get("legal_domain") and result.get("query_type"):
                            with st.expander("🔍 AI 분석 정보", expanded=False):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("법률 분야", result.get("legal_domain", "N/A"))
                                    st.metric("질문 유형", result.get("query_type", "N/A"))
                                    if st.session_state.has_pdf:
                                        st.metric("첨부 파일", "✅ 분석됨")
                                with col2:
                                    confidence = result.get("confidence", 0)
                                    st.metric("신뢰도", f"{confidence:.2f}" if confidence else "N/A")
                                    processing_time = result.get("processing_time", 0)
                                    st.metric("처리 시간", f"{processing_time:.2f}초" if processing_time else "N/A")
                                
                                if result.get("keywords"):
                                    st.write("**핵심 키워드**:", ", ".join(result.get("keywords", [])[:5]))
                                
                                # 검색 결과 정보
                                search_results = result.get("search_results", {})
                                if search_results.get("search_successful"):
                                    st.write(f"**검색된 문서**: {search_results.get('total_results', 0)}개")
                        
                        # 대화 기록 저장
                        try:
                            conversation_manager.save_conversation(
                                session_id=st.session_state.session_id,
                                user_message=prompt,
                                ai_response=ai_response,
                                metadata={
                                    "legal_domain": result.get("legal_domain"),
                                    "query_type": result.get("query_type"),
                                    "confidence": result.get("confidence"),
                                    "processing_time": result.get("processing_time"),
                                    "has_pdf": st.session_state.has_pdf,
                                    "pdf_filename": st.session_state.pdf_filename
                                }
                            )
                            
                            # 로그 기록
                            log_conversation(prompt, ai_response, st.session_state.session_id)
                            
                        except Exception as e:
                            log_error(e, "conversation_save")
                        
                        # 대화 수 증가
                        st.session_state.conversation_count += 1
                        
                    except Exception as e:
                        error_message = "죄송합니다. 시스템 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
                        st.error(error_message)
                        log_error(e, "agent_execution")
                        
                        # 에러 메시지도 대화에 추가
                        ai_message = AIMessage(content=error_message)
                        st.session_state.messages.append(ai_message)
            
            # 페이지 새로고침으로 입력창 초기화
            st.rerun()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("⚠️ 애플리케이션 시작 실패")
        st.exception(e)
        log_error(e, "app_startup") 