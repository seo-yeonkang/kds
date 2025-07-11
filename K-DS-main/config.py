import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class Config:
    """애플리케이션 설정 관리"""
    
    # OpenAI 설정
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
    OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))
    
    # LangSmith 설정 (선택사항)
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "law-agent-chatbot")
    
    # 로깅 설정
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # 데이터베이스 설정
    CONVERSATION_DB = os.getenv("CONVERSATION_DB", "conversations.db")
    
    # 법률 시스템 프롬프트
    LEGAL_SYSTEM_PROMPT = """당신은 한국 법률 전문 AI 어시스턴트입니다.

전문 분야:
- 민사법, 상법, 형법, 행정법 등 한국 법률 전반
- 판례 해석 및 법리 설명  
- 일반인도 이해하기 쉬운 법률 설명

응답 원칙:
1. 정확하고 신뢰할 수 있는 법률 정보 제공
2. 복잡한 법률 용어는 쉽게 풀어서 설명
3. 구체적인 법률 조문이나 판례 언급 시 정확성 강조
4. 개인적인 법률 자문은 전문 변호사 상담 권유
5. 불확실한 내용은 "확실하지 않다"고 명시

답변 구조:
📌 핵심 답변
📚 관련 법률 설명 (있다면)  
⚠️ 주의사항 또는 전문가 상담 필요성

현재는 기본적인 법률 질문 응답이 가능하며, 향후 판례 검색 및 법령 조회 기능이 추가될 예정입니다."""

def setup_langsmith():
    """LangSmith 설정"""
    if Config.LANGCHAIN_TRACING_V2:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        if Config.LANGCHAIN_API_KEY:
            os.environ["LANGCHAIN_API_KEY"] = Config.LANGCHAIN_API_KEY
        os.environ["LANGCHAIN_PROJECT"] = Config.LANGCHAIN_PROJECT
        print("✅ LangSmith 모니터링이 활성화되었습니다.")
    else:
        print("ℹ️ LangSmith 모니터링이 비활성화되어 있습니다.")

def validate_config():
    """필수 설정 검증"""
    errors = []
    
    if not Config.OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY가 설정되지 않았습니다.")
    
    if errors:
        raise ValueError(f"설정 오류:\n" + "\n".join(f"- {error}" for error in errors))
    
    print("✅ 설정 검증 완료") 