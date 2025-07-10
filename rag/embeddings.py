import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import torch
from utils.logger import logger

class ContractEmbeddings:
    """계약서 전용 임베딩 생성기 (BGE-M3 사용)"""
    
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        """
        BGE-M3 모델 초기화
        - 다국어 지원 (한국어 포함)
        - Dense, Sparse, ColBERT 방식 모두 지원
        """
        self.model_name = model_name
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"임베딩 모델 초기화: {model_name} (device: {self.device})")
    
    def load_model(self):
        """모델 로딩 (지연 로딩)"""
        if self.model is None:
            try:
                logger.info(f"BGE-M3 모델 로딩 중...")
                self.model = SentenceTransformer(self.model_name, device=self.device)
                logger.info(f"모델 로딩 완료: {self.model_name}")
            except Exception as e:
                logger.error(f"모델 로딩 실패: {e}")
                raise
    
    def create_embeddings(self, documents: List[Dict]) -> List[np.ndarray]:
        """문서들의 임베딩 생성"""
        self.load_model()
        
        # 텍스트만 추출
        texts = [doc.get('text', '') for doc in documents]
        logger.info(f"임베딩 생성 시작: {len(texts)} 문서")
        
        try:
            # BGE-M3로 임베딩 생성
            embeddings = self.model.encode(
                texts,
                batch_size=32,  # 메모리 효율성
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True  # 코사인 유사도 최적화
            )
            
            logger.info(f"임베딩 생성 완료: shape {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"임베딩 생성 실패: {e}")
            raise
    
    def create_single_embedding(self, text: str) -> np.ndarray:
        """단일 텍스트 임베딩 생성 (쿼리용)"""
        self.load_model()
        
        try:
            embedding = self.model.encode(
                [text],
                convert_to_numpy=True,
                normalize_embeddings=True
            )[0]
            return embedding
            
        except Exception as e:
            logger.error(f"단일 임베딩 생성 실패: {e}")
            raise
    
    def get_model_info(self) -> Dict:
        """모델 정보 반환"""
        self.load_model()
        
        return {
            'model_name': self.model_name,
            'max_seq_length': getattr(self.model, 'max_seq_length', 'unknown'),
            'embedding_dimension': self.model.get_sentence_embedding_dimension(),
            'device': self.device,
            'supports_multilingual': True,
            'supports_hybrid_search': True  # BGE-M3 특징
        }
    
    def preprocess_text(self, text: str) -> str:
        """텍스트 전처리 (필요시)"""
        # BGE-M3는 별도 전처리가 거의 필요 없음
        # 필요시 여기에 추가
        return text.strip()
    
    def batch_encode(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """배치 단위 인코딩 (메모리 효율성)"""
        self.load_model()
        
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            all_embeddings.extend(batch_embeddings)
            
            logger.debug(f"배치 처리 완료: {i + len(batch)}/{len(texts)}")
        
        return all_embeddings 