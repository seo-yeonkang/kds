import re
import numpy as np
from typing import List, Dict, Optional
from langchain.tools import tool
from datetime import datetime

from utils.logger import log_agent_action, logger
from rag.vector_db import ContractVectorDB
from rag.embeddings import ContractEmbeddings
from rag.data_loader import ContractDataLoader
from utils.pdf_processor import get_pdf_processor

# 전역 인스턴스 (한 번만 초기화)
_vector_db = None
_embedding_model = None

def get_vector_db():
    global _vector_db
    if _vector_db is None:
        _vector_db = ContractVectorDB()
        _vector_db.initialize_db()  # ChromaDB 초기화
        
        # 저장된 BM25 인덱스 로드 시도
        if _vector_db.hybrid_engine.load_index():
            logger.info("✅ 저장된 BM25 인덱스 로드 완료")
        else:
            logger.warning("⚠️ BM25 인덱스를 찾을 수 없습니다. build_bm25_index.py를 실행하세요.")
            # 인덱스가 없으면 일단 빈 상태로 진행 (벡터 검색만 사용)
            
    return _vector_db

def get_embedding_model():
    """임베딩 모델 싱글톤 인스턴스 반환"""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = ContractEmbeddings()
    return _embedding_model

# 현재는 기본적인 도구들만 구현
# 나중에 RAG 도구들 (search_qa_tool, search_case_tool 등) 추가 예정

@tool
def analyze_legal_domain(question: str) -> Dict[str, str]:
    """
    질문을 분석하여 법률 분야를 분류합니다.
    현재는 간단한 키워드 기반 분류를 사용하며, 향후 ML 모델로 업그레이드 예정입니다.
    """
    
    log_agent_action("analyze_legal_domain", {"question_length": len(question)})
    
    # 법률 분야별 키워드 매핑
    domain_keywords = {
        "민사법": ["계약", "손해배상", "부동산", "임대차", "매매", "채권", "채무", "불법행위", "소유권"],
        "상법": ["회사", "주식", "이사", "감사", "상거래", "어음", "수표", "파산", "기업"],
        "형법": ["범죄", "처벌", "형벌", "고발", "고소", "수사", "재판", "판결", "실형", "벌금"],
        "행정법": ["행정", "공무원", "행정처분", "행정소송", "허가", "인가", "취소", "정부"],
        "노동법": ["근로", "임금", "해고", "퇴직", "산업재해", "노동조합", "직장", "고용"],
        "가족법": ["결혼", "이혼", "상속", "양육", "위자료", "재산분할", "친권", "부양"],
        "조세법": ["세금", "납세", "과세", "소득세", "법인세", "부가가치세", "세무서"],
    }
    
    # 키워드 매칭으로 분야 결정
    domain_scores = {}
    for domain, keywords in domain_keywords.items():
        score = sum(1 for keyword in keywords if keyword in question)
        if score > 0:
            domain_scores[domain] = score
    
    # 가장 높은 점수의 분야 선택
    if domain_scores:
        predicted_domain = max(domain_scores, key=domain_scores.get)
        confidence = domain_scores[predicted_domain] / len(question.split())
    else:
        predicted_domain = "일반법률"
        confidence = 0.1
    
    return {
        "domain": predicted_domain,
        "confidence": str(confidence),
        "matched_keywords": str(domain_scores)
    }

@tool  
def analyze_query_type(question: str) -> Dict[str, str]:
    """
    질문의 유형을 분석합니다.
    """
    
    log_agent_action("analyze_query_type", {"question": question[:50] + "..."})
    
    # 질문 유형별 패턴
    type_patterns = {
        "사실확인": [r"이 맞나요?", r"사실인가요?", r"정말인가요?", r"맞습니까?"],
        "법령해석": [r"법에서는", r"조문", r"법률", r"규정", r"어떻게 정하고"],
        "판례분석": [r"판례", r"법원", r"판결", r"사례", r"실제로"],
        "절차문의": [r"어떻게 해야", r"절차", r"방법", r"신청", r"제출"],
        "권리구제": [r"소송", r"고소", r"고발", r"구제", r"손해배상"],
        "일반상담": [r"조언", r"상담", r"도움", r"궁금", r"알고싶"]
    }
    
    detected_types = []
    for query_type, patterns in type_patterns.items():
        for pattern in patterns:
            if re.search(pattern, question):
                detected_types.append(query_type)
                break
    
    # 기본값 설정
    if not detected_types:
        detected_types = ["일반상담"]
    
    return {
        "primary_type": detected_types[0],
        "all_types": str(detected_types),
        "analysis_time": datetime.now().isoformat()
    }

@tool
def extract_keywords(question: str) -> Dict[str, List[str]]:
    """
    질문에서 법률 관련 핵심 키워드를 추출합니다.
    """
    
    log_agent_action("extract_keywords", {"question_length": len(question)})
    
    # 법률 전문용어 사전 (일부)
    legal_terms = [
        "계약", "계약서", "계약금", "위약금", "손해배상", "불법행위", "채권", "채무",
        "소유권", "임대차", "매매", "부동산", "전세", "월세", "보증금",
        "이혼", "상속", "위자료", "양육비", "재산분할", "친권",
        "근로계약", "임금", "해고", "퇴직금", "산업재해",
        "형사처벌", "고소", "고발", "수사", "기소", "판결",
        "행정처분", "취소", "무효", "허가", "인가",
        "주식회사", "유한회사", "이사", "감사", "파산", "회생"
    ]
    
    # 질문에서 법률 용어 찾기
    found_terms = [term for term in legal_terms if term in question]
    
    # 일반 키워드 추출 (간단한 방식)
    common_words = ["을", "를", "이", "가", "에", "의", "은", "는", "으로", "로", "에서", "와", "과"]
    words = [word for word in question.split() if len(word) > 1 and word not in common_words]
    
    return {
        "legal_terms": found_terms[:5],  # 최대 5개
        "general_keywords": words[:10],   # 최대 10개
        "extraction_method": "rule_based"
    }

@tool
def process_uploaded_pdf(uploaded_file, current_state: Dict = None) -> Dict[str, any]:
    """
    업로드된 PDF 파일을 OCR 처리하여 텍스트를 추출합니다.
    추출된 텍스트는 Agent 상태에 저장됩니다.
    """
    
    log_agent_action("process_uploaded_pdf", {"filename": getattr(uploaded_file, 'name', 'unknown')})
    
    try:
        # PDF 처리기 가져오기
        pdf_processor = get_pdf_processor()
        
        # 파일 정보 가져오기
        file_info = pdf_processor.get_file_info(uploaded_file)
        
        log_agent_action("pdf_file_info", {
            "filename": file_info['name'],
            "size_mb": file_info['size_mb'],
            "type": file_info['type']
        })
        
        # OCR 처리
        success, result, filename = pdf_processor.process_uploaded_file(uploaded_file)
        
        if success:
            # 성공한 경우
            text_length = len(result)
            preview = result[:200] + "..." if len(result) > 200 else result
            
            log_agent_action("pdf_ocr_success", {
                "filename": filename,
                "text_length": text_length,
                "preview": preview
            })
            
            return {
                "success": True,
                "extracted_text": result,
                "filename": filename,
                "text_length": text_length,
                "preview": preview,
                "message": f"PDF 처리 완료: {filename} ({text_length:,} 글자)"
            }
        else:
            # 실패한 경우
            log_agent_action("pdf_ocr_error", {"error": result})
            
            return {
                "success": False,
                "extracted_text": None,
                "filename": None,
                "text_length": 0,
                "preview": "",
                "message": f"PDF 처리 실패: {result}"
            }
            
    except Exception as e:
        error_msg = f"PDF 처리 중 예상치 못한 오류: {str(e)}"
        log_agent_action("process_uploaded_pdf_error", {"error": error_msg})
        
        return {
            "success": False,
            "extracted_text": None,
            "filename": None,
            "text_length": 0,
            "preview": "",
            "message": error_msg
        }

@tool
def search_legal_documents(query: str) -> Dict[str, any]:
    """
    법률 문서에서 관련 내용을 검색합니다.
    하이브리드 검색(벡터 + BM25)을 사용하여 가장 관련성 높은 문서를 찾습니다.
    """
    
    log_agent_action("search_legal_documents", {"query": query[:50] + "..."})
    
    try:
        # 싱글톤 인스턴스 사용 (모델 로딩 시간 단축)
        vector_db = get_vector_db()
        embedding_model = get_embedding_model()
        
        # 쿼리 임베딩 생성
        query_embedding = embedding_model.create_single_embedding(query)
        
        # 하이브리드 검색 실행
        search_results = vector_db.hybrid_search(
            query=query,
            query_embedding=query_embedding,
            k=5,
            vector_weight=0.7,
            keyword_weight=0.3
        )
        
        # 검색 결과 포맷팅
        formatted_results = []
        for result in search_results:
            formatted_results.append({
                "text": result.get("text", ""),
                "relevance_score": result.get("relevance_score", 0),
                "source": result.get("metadata", {}).get("source_file", ""),
                "article_number": result.get("metadata", {}).get("article_number", ""),
                "content_labels": result.get("metadata", {}).get("content_labels", [])
            })
        
        return {
            "results": formatted_results,
            "total_results": len(formatted_results),
            "search_successful": True,
            "search_summary": f"{len(formatted_results)}개의 관련 문서를 찾았습니다."
        }
        
    except Exception as e:
        log_agent_action("search_legal_documents_error", {"error": str(e)})
        return {
            "results": [],
            "total_results": 0,
            "search_successful": False,
            "search_summary": f"검색 중 오류가 발생했습니다: {str(e)}"
        }

# 현재 사용 가능한 도구들
AVAILABLE_TOOLS = [
    analyze_legal_domain,
    analyze_query_type, 
    extract_keywords,
    search_legal_documents,
    process_uploaded_pdf
] 