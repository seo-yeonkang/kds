from langgraph.graph import StateGraph, END
from typing import Dict, Any, List, Union, Optional
from langchain_core.messages import BaseMessage

from agent.state import LegalAgentState, create_initial_state
from agent.nodes import (
    analyze_legal_query,
    generate_legal_response, 
    validate_legal_answer,
    # 향후 RAG 노드들
    # search_qa_database,
    # search_case_database,
    # search_legal_references
)
from utils.logger import log_agent_action, logger

class LegalAgent:
    """법률 전문 Agent 클래스"""
    
    def __init__(self):
        self.workflow = self._create_workflow()
        self.app = self.workflow.compile()
        logger.info("법률 Agent 초기화 완료")
    
    def _create_workflow(self) -> StateGraph:
        """Agent 워크플로우 생성"""
        
        # StateGraph 생성
        workflow = StateGraph(LegalAgentState)
        
        # 노드 추가
        workflow.add_node("analyze_query", analyze_legal_query)
        workflow.add_node("generate_response", generate_legal_response)
        workflow.add_node("validate_answer", validate_legal_answer)
        
        # 향후 RAG 노드들 추가 예정
        # workflow.add_node("search_qa", search_qa_database)
        # workflow.add_node("search_cases", search_case_database)
        # workflow.add_node("search_legal_refs", search_legal_references)
        
        # 워크플로우 정의
        workflow.set_entry_point("analyze_query")
        
        # 현재 단순한 순차 플로우
        # 향후 조건부 라우팅 추가 예정
        workflow.add_edge("analyze_query", "generate_response")
        workflow.add_edge("generate_response", "validate_answer")
        workflow.add_edge("validate_answer", END)
        
        # 향후 RAG 플로우 (주석 처리)
        # workflow.add_conditional_edges(
        #     "analyze_query",
        #     self._route_after_analysis,
        #     {
        #         "search_qa": "search_qa",
        #         "search_cases": "search_cases", 
        #         "generate_only": "generate_response"
        #     }
        # )
        
        logger.info("Agent 워크플로우 구성 완료")
        return workflow
    
    def _route_after_analysis(self, state: LegalAgentState) -> str:
        """
        질문 분석 후 다음 단계 결정 (향후 구현)
        현재는 사용하지 않음
        """
        # TODO: RAG 구현시 활성화
        # query_type = state.get("query_type", "일반상담")
        # legal_domain = state.get("legal_domain", "일반법률")
        
        # if query_type == "판례분석":
        #     return "search_cases"
        # elif query_type in ["법령해석", "사실확인"]:
        #     return "search_qa"
        # else:
        #     return "generate_only"
        
        return "generate_response"
    
    def invoke(self, input_data: Union[Dict[str, Any], List[BaseMessage]], session_id: str = None) -> Dict[str, Any]:
        """Agent 실행"""
        
        try:
            # 입력 데이터 파싱
            if isinstance(input_data, dict):
                messages = input_data.get("messages", [])
                session_id = input_data.get("session_id", session_id)
                uploaded_pdf_text = input_data.get("uploaded_pdf_text")
                pdf_filename = input_data.get("pdf_filename")
                has_pdf = input_data.get("has_pdf", False)
                
                log_agent_action("agent_invoke", {
                    "session_id": session_id,
                    "messages_count": len(messages),
                    "has_pdf": has_pdf,
                    "pdf_filename": pdf_filename
                })
            else:
                # 기존 방식 (하위 호환성)
                messages = input_data
                uploaded_pdf_text = None
                pdf_filename = None
                has_pdf = False
                
                log_agent_action("agent_invoke", {
                    "session_id": session_id,
                    "messages_count": len(messages),
                    "has_pdf": False
                })
            
            # 초기 상태 생성
            initial_state = create_initial_state(messages, session_id)
            
            # PDF 정보 추가
            if has_pdf:
                initial_state.update({
                    "uploaded_pdf_text": uploaded_pdf_text,
                    "pdf_filename": pdf_filename,
                    "has_pdf": has_pdf
                })
            
            # Agent 실행
            result = self.app.invoke(initial_state)
            
            log_agent_action("agent_completed", {
                "session_id": session_id,
                "outcome": result.get("agent_outcome"),
                "processing_time": result.get("processing_time"),
                "confidence": result.get("confidence"),
                "has_pdf_analysis": has_pdf
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Agent 실행 실패: {e}", exc_info=True)
            return {
                "final_answer": "시스템 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
                "agent_outcome": "error",
                "error_message": str(e)
            }
    
    async def ainvoke(self, input_data: Union[Dict[str, Any], List[BaseMessage]], session_id: str = None) -> Dict[str, Any]:
        """Agent 비동기 실행"""
        
        try:
            # 입력 데이터 파싱
            if isinstance(input_data, dict):
                messages = input_data.get("messages", [])
                session_id = input_data.get("session_id", session_id)
                uploaded_pdf_text = input_data.get("uploaded_pdf_text")
                pdf_filename = input_data.get("pdf_filename")
                has_pdf = input_data.get("has_pdf", False)
                
                log_agent_action("agent_ainvoke", {
                    "session_id": session_id,
                    "messages_count": len(messages),
                    "has_pdf": has_pdf,
                    "pdf_filename": pdf_filename
                })
            else:
                # 기존 방식 (하위 호환성)
                messages = input_data
                uploaded_pdf_text = None
                pdf_filename = None
                has_pdf = False
                
                log_agent_action("agent_ainvoke", {
                    "session_id": session_id,
                    "messages_count": len(messages),
                    "has_pdf": False
                })
            
            # 초기 상태 생성
            initial_state = create_initial_state(messages, session_id)
            
            # PDF 정보 추가
            if has_pdf:
                initial_state.update({
                    "uploaded_pdf_text": uploaded_pdf_text,
                    "pdf_filename": pdf_filename,
                    "has_pdf": has_pdf
                })
            
            # Agent 비동기 실행
            result = await self.app.ainvoke(initial_state)
            
            log_agent_action("agent_completed_async", {
                "session_id": session_id,
                "outcome": result.get("agent_outcome"),
                "processing_time": result.get("processing_time"),
                "confidence": result.get("confidence"),
                "has_pdf_analysis": has_pdf
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Agent 비동기 실행 실패: {e}", exc_info=True)
            return {
                "final_answer": "시스템 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
                "agent_outcome": "error",
                "error_message": str(e)
            }
    
    def stream(self, messages, session_id: str = None):
        """Agent 스트리밍 실행 (향후 구현)"""
        # TODO: 스트리밍 응답이 필요할 때 구현
        pass

def create_legal_agent() -> LegalAgent:
    """법률 Agent 인스턴스 생성"""
    return LegalAgent()

# 향후 추가될 라우팅 함수들
def should_search_qa(state: LegalAgentState) -> bool:
    """Q&A 검색이 필요한지 판단"""
    # TODO: RAG 구현시 활성화
    return False

def should_search_cases(state: LegalAgentState) -> bool:
    """판례 검색이 필요한지 판단"""  
    # TODO: RAG 구현시 활성화
    return False

def should_search_legal_refs(state: LegalAgentState) -> bool:
    """법조문 검색이 필요한지 판단"""
    # TODO: RAG 구현시 활성화
    return False 