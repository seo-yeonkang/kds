import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import numpy as np
from pathlib import Path
from utils.logger import logger
from .hybrid_search import HybridSearchEngine

class ContractVectorDB:
    def __init__(self, persist_directory: str = "data/vector_db", collection_name: str = "contracts"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        
        # 초기화는 지연 로딩
        self.client = None
        self.collection = None
        self.hybrid_engine = None
        self.documents = []  # 검색을 위한 문서 저장
        
        logger.info(f"Vector DB initialized: {persist_directory}/{collection_name}")
    
    def initialize_db(self):
        """ChromaDB 초기화 및 문서 로드"""
        if self.client is None:
            self.client = chromadb.PersistentClient(path=str(self.persist_directory))
            self.collection = self.client.get_or_create_collection(self.collection_name)
            self.hybrid_engine = HybridSearchEngine()
            
            # 기존 문서 수 확인
            doc_count = self.collection.count()
            logger.info(f"ChromaDB initialized: {doc_count} documents")
            
            # BM25 인덱스 로드 및 self.documents 설정
            if self.hybrid_engine.load_index():
                logger.info("✅ BM25 인덱스 로드 완료")
                # BM25 엔진에서 문서들을 가져와 self.documents 설정
                if hasattr(self.hybrid_engine.bm25_engine, 'documents') and self.hybrid_engine.bm25_engine.documents:
                    self.documents = self.hybrid_engine.bm25_engine.documents
                    logger.info(f"✅ 문서 로드 완료: {len(self.documents)}개")
                else:
                    logger.warning("⚠️ BM25 인덱스에 문서가 없습니다")
                    self.documents = []
            else:
                logger.warning("⚠️ BM25 인덱스를 찾을 수 없습니다")
                self.documents = []
    
    def add_documents(self, documents: List[Dict], embeddings: List[np.ndarray]):
        self.initialize_db()
        logger.info(f"Adding {len(documents)} documents")
        
        # 원본 문서 저장 (하이브리드 검색용)
        self.documents = documents
        
        # BM25 인덱스는 별도로 구축된 것을 사용 (build_bm25_index.py)
        # 저장된 인덱스 로드 시도
        if not self.hybrid_engine.load_index():
            logger.warning("⚠️ BM25 인덱스를 찾을 수 없습니다. 검색 성능이 제한될 수 있습니다.")
            logger.info("💡 'python build_bm25_index.py'를 실행하여 인덱스를 생성하세요.")
        
        try:
            # ChromaDB 배치 크기 제한 (5000개씩 나누어서 처리)
            batch_size = 5000
            total_docs = len(documents)
            
            for i in range(0, total_docs, batch_size):
                end_idx = min(i + batch_size, total_docs)
                batch_docs = documents[i:end_idx]
                batch_embeddings = embeddings[i:end_idx]
                
                print(f"배치 {i//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size}: {end_idx - i}개 문서 저장 중...")
                
                ids = [f"doc_{j}" for j in range(i, end_idx)]
                texts = [doc.get("text", "") for doc in batch_docs]
                
                metadatas = []
                for doc in batch_docs:
                    metadata = {
                        "document_title": str(doc.get("document_title", "")),
                        "document_category": str(doc.get("document_category", "")),
                        "article_number": int(doc.get("article_number", 0)) if doc.get("article_number") else 0,
                        "content_labels": ",".join(doc.get("content_labels", [])) if doc.get("content_labels") else "",
                    }
                    metadatas.append(metadata)
                
                self.collection.add(
                    ids=ids,
                    documents=texts,
                    embeddings=[emb.tolist() for emb in batch_embeddings],
                    metadatas=metadatas
                )
                
                print(f"✅ 배치 완료: {end_idx}/{total_docs} 문서")
            
            logger.info(f"Successfully added {len(documents)} documents")
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise
    
    def search(self, query_embedding: np.ndarray, k: int = 5, where: Optional[Dict] = None):
        """기본 벡터 검색 (ChromaDB 호환성 개선)"""
        self.initialize_db()
        
        try:
            # ChromaDB 호환성을 위해 ids 제외
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
            
            formatted_results = []
            for i in range(len(results["documents"][0])):
                # 순서 기반 인덱스 사용 (ChromaDB 제한으로 인한 임시 해결책)
                result = {
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "relevance_score": 1 - results["distances"][0][i],
                    "index": i,  # 검색 결과 순서 기반
                    "search_rank": i  # 벡터 검색 순위
                }
                if result["metadata"].get("content_labels"):
                    result["metadata"]["content_labels"] = result["metadata"]["content_labels"].split(",")
                formatted_results.append(result)
            
            return formatted_results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def bm25_search(self, query: str, k: int = 5):
        """BM25 전용 검색"""
        if not self.documents:
            logger.warning("No documents available for BM25 search")
            return []
        
        results = self.hybrid_engine.bm25_search(query, top_k=k)
        
        formatted_results = []
        for result in results:
            doc_index = result['index']
            formatted_results.append({
                "text": result['document'].get('text', ''),
                "metadata": result['document'].get('metadata', {}),
                "relevance_score": result['keyword_score'],
                "bm25_score": result['keyword_score'],
                "matched_tokens": result['matched_keywords'],
                "index": doc_index
            })
        
        return formatted_results
    
    def hybrid_search(self, query: str, query_embedding: np.ndarray, k: int = 5, 
                     vector_weight: float = 0.7, keyword_weight: float = 0.3):
        """하이브리드 검색: 벡터 + BM25 키워드 검색 조합"""
        if not self.documents:
            logger.warning("No documents available for search")
            return []
        
        # 1. 벡터 검색 실행
        vector_results = self.search(query_embedding, k=min(k*3, 20))
        
        # 2. BM25 검색 실행
        bm25_results = self.bm25_search(query, k=min(k*3, 20))
        
        # 3. 텍스트 기반 매칭으로 결과 조합
        hybrid_scores = {}
        max_bm25_score = max([r['bm25_score'] for r in bm25_results], default=1.0)
        
        # BM25 결과를 기준으로 텍스트 매칭
        for bm25_result in bm25_results:
            doc_index = bm25_result['index']
            bm25_text = bm25_result['text']
            normalized_bm25 = bm25_result['bm25_score'] / max_bm25_score if max_bm25_score > 0 else 0
            
            # 벡터 결과에서 같은 텍스트 찾기
            vector_score = 0
            matched_vector_result = None
            
            for vector_result in vector_results:
                # 텍스트가 같거나 유사한 경우 매칭
                if vector_result['text'] == bm25_text or vector_result['text'][:100] == bm25_text[:100]:
                    vector_score = vector_result['relevance_score']
                    matched_vector_result = vector_result
                    break
            
            hybrid_scores[doc_index] = {
                'vector_score': vector_score,
                'keyword_score': normalized_bm25,
                'document': bm25_result,
                'matched_keywords': bm25_result['matched_tokens'],
                'text_matched': matched_vector_result is not None
            }
        
        # 벡터 전용 결과도 추가 (BM25에서 누락된 것들)
        for vector_result in vector_results:
            vector_text = vector_result['text']
            
            # 이미 매칭된 텍스트인지 확인
            already_matched = False
            for scores in hybrid_scores.values():
                if scores['document']['text'] == vector_text or scores['document']['text'][:100] == vector_text[:100]:
                    already_matched = True
                    break
            
            if not already_matched:
                # 새로운 인덱스 생성 (벡터 전용)
                new_index = len(hybrid_scores) + 1000  # 중복 방지를 위한 오프셋
                hybrid_scores[new_index] = {
                    'vector_score': vector_result['relevance_score'],
                    'keyword_score': 0,
                    'document': vector_result,
                    'matched_keywords': [],
                    'text_matched': False
                }
        
        # 최종 점수 계산 및 정렬
        final_results = []
        for doc_index, scores in hybrid_scores.items():
            final_score = (scores['vector_score'] * vector_weight + 
                          scores['keyword_score'] * keyword_weight)
            
            if final_score > 0:
                result = {
                    'text': scores['document']['text'],
                    'metadata': scores['document'].get('metadata', {}),
                    'relevance_score': final_score,
                    'vector_score': scores['vector_score'],
                    'keyword_score': scores['keyword_score'],
                    'matched_keywords': scores['matched_keywords'],
                    'index': doc_index,
                    'text_matched': scores['text_matched']
                }
                final_results.append(result)
        
        # 최종 점수로 정렬
        final_results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return final_results[:k]
    
    def search_with_explanation(self, query: str, query_embedding: np.ndarray, k: int = 5):
        """검색 결과와 설명을 함께 반환"""
        results = self.hybrid_search(query, query_embedding, k)
        explanation = self.hybrid_engine.get_search_explanation(query, results)
        
        return results, explanation
    
    def compare_search_methods(self, query: str, query_embedding: np.ndarray, k: int = 5):
        """하이브리드 검색만 실행"""
        results = {
            'hybrid': self.hybrid_search(query, query_embedding, k)
        }
        
        return results
    
    def get_collection_info(self) -> Dict:
        self.initialize_db()
        
        try:
            return {
                "collection_name": self.collection_name,
                "total_documents": self.collection.count(),
                "persist_directory": str(self.persist_directory),
                "hybrid_search_enabled": True,
                "bm25_enabled": True,
                "stored_documents": len(self.documents),
                "avg_doc_length": getattr(self.hybrid_engine.bm25_engine, 'avg_doc_length', 0)
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {
                "collection_name": self.collection_name,
                "total_documents": 0,
                "persist_directory": str(self.persist_directory),
                "hybrid_search_enabled": False,
                "bm25_enabled": False,
                "stored_documents": 0,
                "avg_doc_length": 0
            } 