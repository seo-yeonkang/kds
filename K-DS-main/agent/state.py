from typing import TypedDict, List, Dict, Optional, Any
from langchain_core.messages import BaseMessage

class LegalAgentState(TypedDict):
    """법률 Agent의 상태 정의"""
    
    # 기본 메시지 및 질문
    messages: List[BaseMessage]
    question: str
    
    # 질문 분석 결과 (나중에 RAG용)
    legal_domain: Optional[str]      # 법률 분야 (민법, 상법, 형법 등)
    query_type: Optional[str]        # 질문 유형 (사실확인, 법령해석, 판례분석 등)
    keywords: Optional[List[str]]    # 핵심 키워드
    
    # PDF 업로드 관련 (새로 추가)
    uploaded_pdf_text: Optional[str]  # OCR로 추출된 텍스트
    pdf_filename: Optional[str]       # 업로드된 파일명
    has_pdf: Optional[bool]           # PDF 첨부 여부
    
    # 검색 결과 (나중에 RAG 추가시 사용)
    qa_results: Optional[List[Dict]]      # Q&A 검색 결과
    case_results: Optional[List[Dict]]    # 판례 검색 결과
    legal_refs: Optional[List[Dict]]      # 법조문 참조 결과
    search_results: Optional[Dict]        # 하이브리드 검색 결과
    
    # Agent 처리 결과
    reasoning: Optional[str]         # 추론 과정
    final_answer: str               # 최종 답변
    confidence: Optional[float]      # 신뢰도 점수
    sources: Optional[List[Dict]]   # 참조 출처
    
    # 메타데이터
    session_id: Optional[str]       # 세션 ID
    agent_outcome: Optional[str]    # Agent 실행 결과 ('completed', 'error' 등)
    error_message: Optional[str]    # 에러 메시지 (있다면)
    processing_time: Optional[float] # 처리 시간

def create_initial_state(
    messages: List[BaseMessage], 
    session_id: str = None
) -> LegalAgentState:
    """초기 상태 생성"""
    return LegalAgentState(
        messages=messages,
        question="",
        legal_domain=None,
        query_type=None,
        keywords=None,
        uploaded_pdf_text=None,
        pdf_filename=None,
        has_pdf=False,
        qa_results=None,
        case_results=None,
        legal_refs=None,
        search_results=None,
        reasoning=None,
        final_answer="",
        confidence=None,
        sources=None,
        session_id=session_id,
        agent_outcome=None,
        error_message=None,
        processing_time=None
    ) 