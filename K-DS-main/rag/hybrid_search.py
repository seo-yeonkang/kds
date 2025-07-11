import re
import math
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from collections import Counter, defaultdict
from utils.logger import logger

class BM25SearchEngine:
    """BM25 기반 키워드 검색 엔진"""
    
    def __init__(self, k1: float = 1.2, b: float = 0.75, index_path: str = "data/bm25_index.pkl"):
        self.k1 = k1  # 용어 빈도 포화 매개변수
        self.b = b    # 문서 길이 정규화 매개변수
        self.index_path = Path(index_path)
        self.documents = []
        self.tokenized_docs = []
        self.doc_freqs = []
        self.idf_values = {}
        self.avg_doc_length = 0
    
    def save_index(self):
        """BM25 인덱스를 파일로 저장"""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        index_data = {
            'documents': self.documents,
            'tokenized_docs': self.tokenized_docs,
            'doc_freqs': self.doc_freqs,
            'idf_values': self.idf_values,
            'avg_doc_length': self.avg_doc_length,
            'k1': self.k1,
            'b': self.b
        }
        
        with open(self.index_path, 'wb') as f:
            pickle.dump(index_data, f)
        
        logger.info(f"BM25 인덱스 저장 완료: {self.index_path} ({len(self.documents)}개 문서)")
    
    def load_index(self) -> bool:
        """저장된 BM25 인덱스를 로드"""
        if not self.index_path.exists():
            logger.info(f"BM25 인덱스 파일이 없습니다: {self.index_path}")
            return False
        
        try:
            with open(self.index_path, 'rb') as f:
                index_data = pickle.load(f)
            
            self.documents = index_data['documents']
            self.tokenized_docs = index_data['tokenized_docs']
            self.doc_freqs = index_data['doc_freqs']
            self.idf_values = index_data['idf_values']
            self.avg_doc_length = index_data['avg_doc_length']
            self.k1 = index_data.get('k1', self.k1)
            self.b = index_data.get('b', self.b)
            
            logger.info(f"BM25 인덱스 로드 완료: {len(self.documents)}개 문서")
            return True
            
        except Exception as e:
            logger.error(f"BM25 인덱스 로드 실패: {e}")
            return False
    
    def tokenize(self, text: str) -> List[str]:
        """한국어 텍스트 토큰화"""
        # 한국어 단어 분리 (간단한 형태소 분석)
        # 공백, 구두점으로 분리하고 2글자 이상만 사용
        tokens = re.findall(r'[가-힣]{2,}|[a-zA-Z]{2,}|\d+', text.lower())
        return tokens
    
    def build_index(self, documents: List[Dict]):
        """BM25 인덱스 구축"""
        self.documents = documents
        self.tokenized_docs = []
        self.doc_freqs = []
        
        # 1. 문서 토큰화 및 용어 빈도 계산
        for doc in documents:
            text = doc.get('text', '')
            tokens = self.tokenize(text)
            self.tokenized_docs.append(tokens)
            
            # 용어 빈도 계산
            term_freq = Counter(tokens)
            self.doc_freqs.append(term_freq)
        
        # 2. 평균 문서 길이 계산
        doc_lengths = [len(tokens) for tokens in self.tokenized_docs]
        self.avg_doc_length = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0
        
        # 3. IDF 계산
        self._calculate_idf()
        
        # 4. 인덱스 자동 저장
        self.save_index()
    
    def _calculate_idf(self):
        """역문서 빈도 (IDF) 계산"""
        # 각 용어가 등장하는 문서 수 계산
        doc_count = len(self.documents)
        term_doc_count = defaultdict(int)
        
        for doc_freq in self.doc_freqs:
            for term in doc_freq.keys():
                term_doc_count[term] += 1
        
        # IDF 계산: log((N - df + 0.5) / (df + 0.5))
        for term, df in term_doc_count.items():
            self.idf_values[term] = math.log((doc_count - df + 0.5) / (df + 0.5))
    
    def calculate_bm25_score(self, query_tokens: List[str], doc_index: int) -> float:
        """BM25 점수 계산"""
        if doc_index >= len(self.doc_freqs):
            return 0.0
        
        doc_freq = self.doc_freqs[doc_index]
        doc_length = len(self.tokenized_docs[doc_index])
        
        score = 0.0
        for term in query_tokens:
            if term in doc_freq:
                # 용어 빈도
                tf = doc_freq[term]
                # IDF 값
                idf = self.idf_values.get(term, 0)
                
                # BM25 점수 계산
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
                
                score += idf * (numerator / denominator)
        
        return score
    
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """BM25 검색 실행"""
        if not self.documents:
            return []
        
        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []
        
        # 모든 문서에 대해 BM25 점수 계산
        scored_docs = []
        for i, doc in enumerate(self.documents):
            score = self.calculate_bm25_score(query_tokens, i)
            
            if score > 0:
                scored_docs.append({
                    'index': i,
                    'document': doc,
                    'bm25_score': score,
                    'matched_tokens': [t for t in query_tokens if t in self.doc_freqs[i]]
                })
        
        # 점수 순으로 정렬
        scored_docs.sort(key=lambda x: x['bm25_score'], reverse=True)
        
        return scored_docs[:top_k]


class HybridSearchEngine:
    """하이브리드 검색 엔진: 벡터 검색 + BM25 키워드 검색 조합"""
    
    def __init__(self, index_path: str = "data/bm25_index.pkl"):
        self.keyword_mapping = self._create_keyword_mapping()
        self.bm25_engine = BM25SearchEngine(index_path=index_path)
    
    def load_index(self) -> bool:
        """저장된 BM25 인덱스를 로드"""
        return self.bm25_engine.load_index()
    
    def _create_keyword_mapping(self) -> Dict[str, List[str]]:
        """키워드 매핑 테이블 생성"""
        return {
            "근로시간": ["근무시간", "근로시간", "업무시간", "작업시간"],
            "휴가": ["휴가", "휴일", "연차", "휴무"],
            "임금": ["임금", "보수", "급여", "월급", "연봉", "급료"],
            "퇴직금": ["퇴직금", "퇴직급여", "퇴직수당"],
            "계약": ["계약", "근로계약", "고용계약"],
            "해지": ["해지", "해고", "퇴직", "종료"],
            "연장근로": ["연장근로", "초과근무", "야근", "오버타임"],
            "휴게시간": ["휴게시간", "휴식시간", "break"],
            "수당": ["수당", "장려금", "보너스", "상여금"],
            "보험": ["보험", "사회보험", "4대보험"],
            "교육": ["교육", "연수", "훈련"],
            "출장": ["출장", "파견", "외근"],
            "징계": ["징계", "제재", "처벌"],
            "승진": ["승진", "인사", "진급"]
        }
    
    def expand_query(self, query: str) -> str:
        """쿼리 확장: 키워드 매핑을 통한 동의어 추가"""
        expanded_terms = []
        
        # 원본 쿼리 추가
        expanded_terms.append(query)
        
        # 키워드 매핑 적용
        for main_keyword, related_keywords in self.keyword_mapping.items():
            for keyword in related_keywords:
                if keyword in query.lower():
                    # 관련 키워드들을 모두 추가
                    expanded_terms.extend(related_keywords)
                    break
        
        return " ".join(set(expanded_terms))
    
    def build_index(self, documents: List[Dict]):
        """하이브리드 검색을 위한 인덱스 구축"""
        self.bm25_engine.build_index(documents)
    
    def bm25_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """BM25 기반 키워드 검색"""
        # 쿼리 확장
        expanded_query = self.expand_query(query)
        
        # BM25 검색 실행
        results = self.bm25_engine.search(expanded_query, top_k)
        
        # 결과 포맷팅
        formatted_results = []
        for result in results:
            formatted_results.append({
                'index': result['index'],
                'document': result['document'],
                'keyword_score': result['bm25_score'],
                'matched_keywords': result['matched_tokens']
            })
        
        return formatted_results
    
    def hybrid_search(self, 
                     query: str, 
                     vector_results: List[Dict], 
                     documents: List[Dict],
                     vector_weight: float = 0.7,
                     keyword_weight: float = 0.3) -> List[Dict]:
        """하이브리드 검색: 벡터 + BM25 키워드 결과 조합"""
        
        # BM25 검색 실행
        bm25_results = self.bm25_search(query, top_k=len(documents))
        
        # 결과 조합을 위한 점수 계산
        hybrid_scores = {}
        max_bm25_score = max([r['keyword_score'] for r in bm25_results], default=1.0)
        
        # 벡터 검색 결과 처리
        for i, result in enumerate(vector_results):
            doc_index = result.get('index', i)
            vector_score = result.get('relevance_score', 0)
            
            hybrid_scores[doc_index] = {
                'vector_score': vector_score,
                'keyword_score': 0,
                'document': result,
                'matched_keywords': []
            }
        
        # BM25 검색 결과 처리
        for result in bm25_results:
            doc_index = result['index']
            # BM25 점수 정규화 (0~1 범위)
            normalized_bm25 = result['keyword_score'] / max_bm25_score if max_bm25_score > 0 else 0
            
            if doc_index in hybrid_scores:
                hybrid_scores[doc_index]['keyword_score'] = normalized_bm25
                hybrid_scores[doc_index]['matched_keywords'] = result['matched_keywords']
            else:
                hybrid_scores[doc_index] = {
                    'vector_score': 0,
                    'keyword_score': normalized_bm25,
                    'document': result['document'],
                    'matched_keywords': result['matched_keywords']
                }
        
        # 최종 점수 계산 및 정렬
        final_results = []
        for doc_index, scores in hybrid_scores.items():
            final_score = (scores['vector_score'] * vector_weight + 
                          scores['keyword_score'] * keyword_weight)
            
            if final_score > 0:
                result = {
                    'text': scores['document'].get('text', ''),
                    'metadata': scores['document'].get('metadata', {}),
                    'relevance_score': final_score,
                    'vector_score': scores['vector_score'],
                    'keyword_score': scores['keyword_score'],
                    'matched_keywords': scores['matched_keywords']
                }
                final_results.append(result)
        
        # 최종 점수로 정렬
        final_results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return final_results
    
    def get_search_explanation(self, query: str, results: List[Dict]) -> str:
        """검색 결과 설명 생성"""
        if not results:
            return "검색 결과가 없습니다."
        
        expanded_query = self.expand_query(query)
        
        explanation = f"'{query}' 하이브리드 검색 결과 (BM25 기반):\n"
        explanation += f"- 확장된 쿼리: {expanded_query}\n"
        explanation += f"- 총 {len(results)}개 문서 발견\n"
        
        for i, result in enumerate(results[:3], 1):
            explanation += f"\n{i}. 종합 점수: {result['relevance_score']:.3f}"
            explanation += f" (벡터: {result['vector_score']:.3f}, BM25: {result['keyword_score']:.3f})"
            if result['matched_keywords']:
                explanation += f" | 매칭 토큰: {', '.join(result['matched_keywords'])}"
        
        return explanation 