import time
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from agent.state import LegalAgentState
from agent.tools import analyze_legal_domain, analyze_query_type, extract_keywords, search_legal_documents
from config import Config
from utils.logger import log_agent_action, log_error

def analyze_legal_query(state: LegalAgentState) -> Dict[str, Any]:
    """
    사용자 질문을 분석하는 노드
    - 질문 추출
    - 법률 분야 분류  
    - 질문 유형 분석
    - 키워드 추출
    """
    
    start_time = time.time()
    
    try:
        # 사용자 메시지에서 질문 추출
        user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
        if not user_messages:
            raise ValueError("사용자 질문을 찾을 수 없습니다.")
        
        question = user_messages[-1].content
        log_agent_action("analyze_legal_query", {"question_preview": question[:100]})
        
        # 도구들을 사용해 질문 분석
        domain_result = analyze_legal_domain.invoke({"question": question})
        type_result = analyze_query_type.invoke({"question": question})  
        keywords_result = extract_keywords.invoke({"question": question})
        
        processing_time = time.time() - start_time
        
        return {
            "question": question,
            "legal_domain": domain_result["domain"],
            "query_type": type_result["primary_type"],
            "keywords": keywords_result["legal_terms"] + keywords_result["general_keywords"],
            "processing_time": processing_time,
            "agent_outcome": "analysis_completed"
        }
        
    except Exception as e:
        log_error(e, "analyze_legal_query")
        processing_time = time.time() - start_time
        
        return {
            "question": "분석 실패",
            "legal_domain": "알 수 없음",
            "query_type": "일반상담", 
            "keywords": [],
            "processing_time": processing_time,
            "agent_outcome": "analysis_error",
            "error_message": str(e)
        }

def generate_legal_response(state: LegalAgentState) -> Dict[str, Any]:
    """
    법률 응답을 생성하는 노드
    PDF 컨텍스트와 RAG 검색 결과를 통합하여 근거 기반 응답 생성
    """
    
    start_time = time.time()
    
    try:
        # 1. 질문 기반 문서 검색
        question = state.get("question", "")
        search_results = search_legal_documents.invoke({"query": question})
        
        # 2. PDF 컨텍스트 확인
        has_pdf = state.get("has_pdf", False)
        pdf_text = state.get("uploaded_pdf_text", "")
        pdf_filename = state.get("pdf_filename", "")
        
        log_agent_action("generate_legal_response", {
            "domain": state.get('legal_domain'),
            "query_type": state.get('query_type'),
            "keywords_count": len(state.get('keywords', [])),
            "search_successful": search_results.get("search_successful", False),
            "found_documents": search_results.get("total_results", 0),
            "has_pdf": has_pdf,
            "pdf_text_length": len(pdf_text) if pdf_text else 0
        })
        
        # 3. LLM 초기화
        llm = ChatOpenAI(
            model=Config.OPENAI_MODEL,
            temperature=Config.OPENAI_TEMPERATURE,
            openai_api_key=Config.OPENAI_API_KEY
        )
        
        # 4. PDF 컨텍스트 구성
        pdf_context = ""
        if has_pdf and pdf_text:
            # 긴 텍스트는 요약하거나 앞부분만 사용
            if len(pdf_text) > 3000:
                pdf_display = pdf_text[:3000] + "\n\n[... 내용이 길어 일부만 표시 ...]"
            else:
                pdf_display = pdf_text
                
            pdf_context = f"""
=== 업로드된 계약서 내용 ({pdf_filename}) ===
{pdf_display}

"""
        
        # 5. 검색 결과를 활용한 시스템 프롬프트 구성
        search_context = ""
        sources = []
        
        if search_results.get("search_successful", False) and search_results.get("results"):
            search_context = "\n=== 관련 문서 검색 결과 ===\n"
            for i, result in enumerate(search_results["results"][:3], 1):  # 상위 3개만 사용
                search_context += f"\n[문서 {i}]\n"
                search_context += f"내용: {result['text'][:500]}...\n"
                search_context += f"출처: {result['source']}\n"
                search_context += f"관련성: {result['relevance_score']:.2f}\n"
                
                sources.append({
                    "source": result['source'],
                    "relevance_score": result['relevance_score'],
                    "article_number": result.get('article_number', ''),
                    "content_labels": result.get('content_labels', [])
                })
        
        # 6. 통합 프롬프트 구성
        instruction_context = ""
        if has_pdf and pdf_text and search_results.get("search_successful", False):
            instruction_context = """
위에 제공된 업로드된 계약서 내용과 관련 문서 검색 결과를 모두 참고하여 답변해주세요.
- 업로드된 계약서의 구체적인 내용을 우선적으로 분석하고
- 일반적인 기준과 비교하여 설명해주세요
- 검색된 문서의 내용을 인용할 때는 출처를 명시해주세요
"""
        elif has_pdf and pdf_text:
            instruction_context = """
위에 제공된 업로드된 계약서 내용을 참고하여 답변해주세요.
- 계약서의 구체적인 내용을 분석하고 설명해주세요
"""
        elif search_results.get("search_successful", False):
            instruction_context = """
위 검색 결과를 참고하여 정확하고 근거 있는 답변을 제공해주세요.
- 검색된 문서의 내용을 인용할 때는 출처를 명시해주세요
"""
        
        enhanced_prompt = f"""{Config.LEGAL_SYSTEM_PROMPT}

추가 분석 정보:
- 법률 분야: {state.get('legal_domain', '일반법률')}
- 질문 유형: {state.get('query_type', '일반상담')}
- 핵심 키워드: {', '.join(state.get('keywords', [])[:5])}

{pdf_context}{search_context}

{instruction_context}"""

        # 7. 메시지 구성
        messages = [
            SystemMessage(content=enhanced_prompt),
            *state["messages"]
        ]
        
        # 8. LLM 호출
        response = llm.invoke(messages)
        
        processing_time = time.time() - start_time
        
        # 9. 신뢰도 계산 (PDF 및 검색 결과 유무에 따라 조정)
        confidence = 0.8  # 기본 신뢰도
        if has_pdf and pdf_text:
            confidence = min(0.95, confidence + 0.15)  # PDF 있으면 신뢰도 대폭 증가
        if search_results.get("search_successful", False):
            confidence = min(0.95, confidence + 0.05)  # 검색 성공시 신뢰도 소폭 증가
        
        # 10. 추론 과정 설명 생성
        reasoning_parts = []
        reasoning_parts.append(f"법률 분야 '{state.get('legal_domain')}'에 대한 '{state.get('query_type')}' 유형의 질문으로 분석")
        
        if has_pdf:
            reasoning_parts.append(f"업로드된 계약서({pdf_filename}) 내용 분석")
        
        reasoning_parts.append(f"{search_results.get('total_results', 0)}개의 관련 문서 검색")
        reasoning_parts.append("통합 분석하여 응답 생성")
        
        reasoning = " → ".join(reasoning_parts) + "했습니다."
        
        return {
            "final_answer": response.content,
            "reasoning": reasoning,
            "confidence": confidence,
            "sources": sources,
            "search_results": search_results,
            "processing_time": processing_time,
            "agent_outcome": "completed"
        }
        
    except Exception as e:
        log_error(e, "generate_legal_response")
        processing_time = time.time() - start_time
        
        return {
            "final_answer": "죄송합니다. 일시적인 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
            "reasoning": "시스템 오류 발생",
            "confidence": 0.0,
            "sources": [],
            "search_results": {"search_successful": False, "results": []},
            "processing_time": processing_time,
            "agent_outcome": "error", 
            "error_message": str(e)
        }

def validate_legal_answer(state: LegalAgentState) -> Dict[str, Any]:
    """
    생성된 답변을 검증하는 노드
    현재는 기본적인 검증만 수행
    향후 더 정교한 검증 로직 추가 예정
    """
    
    start_time = time.time()
    
    try:
        answer = state.get("final_answer", "")
        
        log_agent_action("validate_legal_answer", {"answer_length": len(answer)})
        
        # 기본적인 검증
        validation_results = {
            "has_content": len(answer.strip()) > 0,
            "appropriate_length": 50 <= len(answer) <= 3000,
            "contains_korean": any(ord(char) >= 0xAC00 and ord(char) <= 0xD7A3 for char in answer),
            "no_error_keywords": not any(keyword in answer.lower() for keyword in ["error", "exception", "traceback"])
        }
        
        # 검증 점수 계산
        passed_checks = sum(validation_results.values())
        total_checks = len(validation_results)
        validation_score = passed_checks / total_checks
        
        processing_time = time.time() - start_time
        
        # 검증 실패시 기본 답변으로 대체
        if validation_score < 0.7:
            return {
                "final_answer": "죄송합니다. 현재 정확한 답변을 제공하기 어려운 상황입니다. 구체적인 법률 자문이 필요하시다면 전문 변호사와 상담해보시기 바랍니다.",
                "confidence": 0.3,
                "processing_time": processing_time,
                "agent_outcome": "validation_failed"
            }
        
        return {
            "confidence": min(state.get("confidence", 0.8) * validation_score, 1.0),
            "processing_time": processing_time,
            "agent_outcome": "validation_passed"
        }
        
    except Exception as e:
        log_error(e, "validate_legal_answer")
        processing_time = time.time() - start_time
        
        return {
            "final_answer": "시스템 오류가 발생했습니다. 관리자에게 문의해주세요.",
            "confidence": 0.0,
            "processing_time": processing_time,
            "agent_outcome": "validation_error",
            "error_message": str(e)
        }

# 향후 RAG 노드들 추가 예정
def search_qa_database(state: LegalAgentState) -> Dict[str, Any]:
    """
    Q&A 데이터베이스 검색 노드 (향후 구현)
    """
    # TODO: RAG 구현시 추가
    return {"qa_results": []}

def search_case_database(state: LegalAgentState) -> Dict[str, Any]:
    """
    판례 데이터베이스 검색 노드 (향후 구현) 
    """
    # TODO: RAG 구현시 추가
    return {"case_results": []}

def search_legal_references(state: LegalAgentState) -> Dict[str, Any]:
    """
    법조문 참조 검색 노드 (향후 구현)
    """
    # TODO: RAG 구현시 추가
    return {"legal_refs": []} 