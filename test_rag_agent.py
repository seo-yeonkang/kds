#!/usr/bin/env python3
"""
RAG 통합 Agent 테스트 스크립트
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agent.workflow import create_legal_agent
from langchain_core.messages import HumanMessage
from config import setup_langsmith

def test_rag_agent():
    """RAG 통합 Agent 테스트"""
    
    print("🤖 RAG 통합 Agent 테스트 시작...")
    
    try:
        # LangSmith 설정
        setup_langsmith()
        
        # Agent 생성
        print("📝 Agent 초기화 중...")
        agent = create_legal_agent()
        
        # 테스트 질문들
        test_questions = [
            "근로계약서에서 임금 지급 방법은 어떻게 되나요?",
            "근로시간은 어떻게 정해지나요?",
            "휴가 규정에 대해 알려주세요."
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n{'='*50}")
            print(f"🔍 테스트 {i}: {question}")
            print('='*50)
            
            # 메시지 생성
            messages = [HumanMessage(content=question)]
            
            # Agent 실행 - 올바른 매개변수 형식 사용
            input_data = {
                "messages": messages,
                "session_id": f"test_session_{i}",
                "has_pdf": False
            }
            
            result = agent.invoke(input_data)
            
            # 결과 출력
            print(f"📊 처리 결과:")
            print(f"  - 최종 답변: {result.get('final_answer', 'N/A')[:200]}...")
            print(f"  - 추론 과정: {result.get('reasoning', 'N/A')}")
            print(f"  - 신뢰도: {result.get('confidence', 'N/A')}")
            print(f"  - 검색 성공: {result.get('search_results', {}).get('search_successful', 'N/A')}")
            print(f"  - 찾은 문서 수: {result.get('search_results', {}).get('total_results', 'N/A')}")
            
            # 소스 정보 출력
            sources = result.get('sources', [])
            if sources:
                print(f"  - 참고 문서:")
                for j, source in enumerate(sources[:2], 1):
                    print(f"    [{j}] {source.get('source', 'N/A')} (관련성: {source.get('relevance_score', 'N/A'):.2f})")
            
        print(f"\n🎉 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_rag_agent() 