# ⚖️ 법률 AI 어시스턴트

LangChain과 LangGraph를 사용한 법률 전문 Agent 챗봇입니다.

## 🎯 프로젝트 개요

이 프로젝트는 한국 법률 관련 질문에 답변하는 AI 어시스턴트입니다. Agent 기반 아키텍처를 사용하여 질문을 분석하고, 적절한 법률 분야를 식별하여 전문적인 답변을 제공합니다.

### 🌟 주요 기능

- **Agent 기반 아키텍처**: LangGraph를 사용한 확장 가능한 워크플로우
- **법률 전문화 번역**: M2M100 기반 파인튜닝을 통한 법률 전문 번역역
- **질문 유형 분석**: 사실확인, 법령해석, 판례분석 등 유형별 처리
- **대화 기록 관리**: SQLite 기반 대화 기록 저장 및 관리
- **실시간 모니터링**: LangSmith 연동으로 Agent 동작 추적
- **사용자 친화적 UI**: Streamlit 기반 직관적인 웹 인터페이스

### 🔧 기술 스택

- **Agent Framework**: LangChain + LangGraph
- **LLM**: OpenAI GPT-4
- **Frontend**: Streamlit
- **Database**: SQLite
- **Monitoring**: LangSmith
- **Language**: Python 3.8+

## 📁 프로젝트 구조

```
law_agent_chatbot/
├── agent/                 # Agent 관련 모듈
│   ├── __init__.py
│   ├── state.py          # Agent 상태 정의
│   ├── nodes.py          # Agent 노드 함수들
│   ├── tools.py          # Agent 도구들
│   └── workflow.py       # LangGraph 워크플로우
├── utils/                # 유틸리티 모듈
│   ├── __init__.py
│   ├── conversation.py   # 대화 기록 관리
│   └── logger.py         # 로깅 시스템
├── config.py             # 설정 관리
├── app.py               # Streamlit 앱
├── requirements.txt     # 의존성 패키지
├── env.example         # 환경 변수 예시
└── README.md           # 프로젝트 설명
```

## 🚀 설치 및 실행

### 1. 환경 설정

```bash
# 저장소 클론
git clone <repository-url>
cd law_agent_chatbot

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정

```bash
# .env 파일 생성
cp env.example .env

# .env 파일 편집
OPENAI_API_KEY=your-openai-api-key-here
LANGCHAIN_API_KEY=your-langsmith-api-key-here  # 선택사항
```

### 3. 앱 실행

```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501`에 접속하여 챗봇을 사용할 수 있습니다.

## 💡 사용법

### 기본 사용법

1. 웹 브라우저에서 앱에 접속
2. 하단의 입력창에 법률 관련 질문 입력
3. AI가 질문을 분석하고 답변 제공
4. 사이드바에서 분석 정보 및 대화 관리 기능 활용

### 질문 예시

- 근로 관련 질문 (한국어 + 중국어)
- 사장님이 여권을 달라고 하십니다. 괜찮은가요?
- 야간근로수당은 어떻게 되나요?


## 📊 모니터링

### LangSmith 연동

LangSmith를 사용하여 Agent의 동작을 실시간으로 모니터링할 수 있습니다:

- Agent 실행 추적
- 각 노드별 성능 측정
- 오류 발생 시 상세 로그
- 사용자 질문 패턴 분석

### 로깅 시스템

- 대화 기록 자동 저장
- 오류 로그 기록
- 성능 메트릭 수집
- 일별 로그 파일 생성

## 🛡️ 주의사항

- 이 시스템은 참고용 정보를 제공하며, 구체적인 법률 자문을 대체하지 않습니다
- 중요한 법률 문제의 경우 반드시 전문 변호사와 상담하시기 바랍니다
- 제공되는 정보의 정확성을 보장하지 않으므로 신중히 판단하여 사용하세요

## 🤝 기여하기

프로젝트 개선을 위한 기여를 환영합니다!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 있습니다.

## 🆘 지원

문제가 발생하거나 질문이 있으시면 Issues를 통해 문의해주세요.

---

⚖️ **법률 AI 어시스턴트** - 당신의 법률 궁금증을 해결해드립니다! 
