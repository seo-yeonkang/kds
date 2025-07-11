#!/usr/bin/env python3
"""
BM25 인덱스 구축 스크립트

이 스크립트를 한 번 실행하면 BM25 인덱스가 data/bm25_index.pkl 파일로 저장됩니다.
이후 Streamlit 앱에서는 이 파일을 빠르게 로드하여 사용할 수 있습니다.

사용법:
    python build_bm25_index.py
"""

import sys
import time
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python path에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rag.data_loader import ContractDataLoader
from rag.hybrid_search import HybridSearchEngine
from utils.logger import logger

def main():
    """BM25 인덱스 구축 메인 함수"""
    
    print("🚀 BM25 인덱스 구축을 시작합니다...")
    start_time = time.time()
    
    try:
        # 1. 데이터 로더 초기화
        print("📂 계약서 데이터를 로드하는 중...")
        data_loader = ContractDataLoader()
        all_docs = data_loader.load_all_documents()
        print(f"✅ {len(all_docs):,}개 문서 로드 완료")
        
        # 2. 하이브리드 검색 엔진 초기화
        print("🔧 BM25 인덱스를 구축하는 중...")
        hybrid_engine = HybridSearchEngine()
        
        # 3. BM25 인덱스 구축 (자동으로 저장됨)
        hybrid_engine.build_index(all_docs)
        
        # 4. 완료 확인
        elapsed_time = time.time() - start_time
        print(f"""
🎉 BM25 인덱스 구축 완료!

📊 구축 정보:
   - 처리된 문서: {len(all_docs):,}개
   - 소요 시간: {elapsed_time:.2f}초
   - 저장 위치: data/bm25_index.pkl

💡 이제 Streamlit 앱을 실행하면 첫 질문부터 빠르게 응답합니다!
   
🚀 앱 실행: streamlit run app.py
        """)
        
        # 5. 인덱스 테스트
        print("🧪 인덱스 테스트 중...")
        test_results = hybrid_engine.bm25_search("계약 위반", top_k=3)
        if test_results:
            print(f"✅ 테스트 성공: '{test_results[0]['document']['text'][:50]}...' 등 {len(test_results)}개 결과")
        else:
            print("⚠️ 테스트에서 결과가 없습니다.")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        logger.error(f"BM25 인덱스 구축 실패: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 