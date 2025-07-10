import json
import os
from pathlib import Path
from typing import List, Dict
from utils.logger import logger

class ContractDataLoader:
    """계약서 데이터 로더"""
    
    def __init__(self, data_dir: str = "data/rag_labeling"):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"데이터 디렉토리를 찾을 수 없습니다: {data_dir}")
    
    def load_all_contracts(self) -> List[Dict]:
        """모든 계약서 데이터 로딩"""
        documents = []
        json_files = list(self.data_dir.glob("*.json"))
        
        if not json_files:
            raise FileNotFoundError(f"JSON 파일을 찾을 수 없습니다: {self.data_dir}")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                
                # 각 chunk에 파일 정보 추가
                for chunk in chunks:
                    chunk['source_file'] = json_file.name
                    documents.append(chunk)
                
            except Exception as e:
                logger.error(f"파일 로딩 실패: {json_file.name} - {e}")
                continue
        
        return documents
    
    def load_sample_data(self, sample_size: int = 100) -> List[Dict]:
        """샘플 데이터 로딩 (테스트용)"""
        all_documents = self.load_all_contracts()
        sample_docs = all_documents[:sample_size]
        return sample_docs
    
    def get_data_statistics(self) -> Dict:
        """데이터 통계 정보"""
        documents = self.load_all_contracts()
        
        stats = {
            'total_documents': len(documents),
            'document_types': {},
            'content_labels': set(),
            'article_numbers': set(),
            'avg_chunk_length': 0
        }
        
        total_length = 0
        for doc in documents:
            # 문서 유형 통계
            doc_type = doc.get('document_category', 'unknown')
            stats['document_types'][doc_type] = stats['document_types'].get(doc_type, 0) + 1
            
            # 컨텐츠 라벨 수집
            if doc.get('content_labels'):
                stats['content_labels'].update(doc['content_labels'])
            
            # 조항 번호 수집
            if doc.get('article_number'):
                stats['article_numbers'].add(doc['article_number'])
            
            # 평균 길이 계산
            total_length += doc.get('chunk_length', 0)
        
        stats['avg_chunk_length'] = total_length / len(documents) if documents else 0
        stats['content_labels'] = list(stats['content_labels'])
        stats['article_numbers'] = sorted(list(stats['article_numbers']))
        
        return stats
    
    def filter_documents(self, documents: List[Dict], **filters) -> List[Dict]:
        """문서 필터링"""
        filtered = documents
        
        if 'document_category' in filters:
            filtered = [doc for doc in filtered if doc.get('document_category') == filters['document_category']]
        
        if 'article_number' in filters:
            filtered = [doc for doc in filtered if doc.get('article_number') == filters['article_number']]
        
        if 'content_labels' in filters:
            target_labels = filters['content_labels']
            if isinstance(target_labels, str):
                target_labels = [target_labels]
            filtered = [doc for doc in filtered 
                       if any(label in doc.get('content_labels', []) for label in target_labels)]
        
        return filtered
    
    def load_all_documents(self) -> List[Dict]:
        """모든 문서 데이터 로딩 (별칭)"""
        return self.load_all_contracts()
    
    def get_document_types(self) -> List[str]:
        """문서 유형 목록 반환"""
        stats = self.get_data_statistics()
        return list(stats['document_types'].keys())
    
    def get_avg_text_length(self) -> float:
        """평균 텍스트 길이 반환"""
        stats = self.get_data_statistics()
        return stats['avg_chunk_length']
    
    def get_unique_articles(self) -> List[int]:
        """고유 조항 번호 목록 반환"""
        stats = self.get_data_statistics()
        return stats['article_numbers']
    
    def load_labor_contracts_only(self) -> List[Dict]:
        """근로계약서만 로딩 (document_title 기준)"""
        all_documents = self.load_all_documents()
        labor_contracts = [doc for doc in all_documents 
                      if doc.get('document_title', '').startswith('근로계약서')]
        return labor_contracts 