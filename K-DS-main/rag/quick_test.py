import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.data_loader import ContractDataLoader
from rag.embeddings import ContractEmbeddings
from rag.vector_db import ContractVectorDB

def smart_rag_test():
    """하이브리드 검색 테스트 (벡터 + BM25)"""
    print("🚀 하이브리드 검색 시스템 테스트")
    print("="*50)
    
    vector_db = ContractVectorDB()
    
    # 기존 데이터 확인
    collection_info = vector_db.get_collection_info()
    
    if collection_info['total_documents'] > 0:
        print(f"✅ 기존 데이터: {collection_info['total_documents']} 문서")
        
        # 원본 문서 로딩 (BM25용)
        data_loader = ContractDataLoader()
        all_docs = data_loader.load_all_documents()
        
        # BM25 인덱스 구축
        vector_db.documents = all_docs
        vector_db.hybrid_engine.build_index(all_docs)
        
        print(f"✅ 하이브리드 인덱스 구축 완료: {len(all_docs)} 문서")
        
        # 검색용 임베딩 모델 (지연 로딩)
        embeddings = ContractEmbeddings()
        
    else:
        print("❌ 기존 데이터 없음")
        return
    
    # 검색 테스트 
    print("\n" + "="*50)
    print("🔍 하이브리드 검색 테스트 (벡터 + BM25)")
    print("="*50)
    
    test_questions = [
        "근로시간은 어떻게 되나요?",
        "연차휴가는 며칠인가요?",
        "퇴직금은 어떻게 계산하나요?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n🎯 질문 {i}: {question}")
        print("-"*30)
        
        # 질문 임베딩 생성
        query_emb = embeddings.create_single_embedding(question)
        
        # 하이브리드 검색
        comparison = vector_db.compare_search_methods(question, query_emb, k=3)
        results = comparison['hybrid']
        
        if not results:
            print("   ❌ 검색 결과 없음")
            continue
            
        for j, result in enumerate(results, 1):
            score = result['relevance_score']
            text_preview = result['text'][:80] + "..." if len(result['text']) > 80 else result['text']
            keywords = result.get('matched_keywords', [])
            vector_score = result.get('vector_score', 0)
            bm25_score = result.get('keyword_score', 0)
            
            print(f"   {j}. 점수: {score:.3f} (벡터: {vector_score:.3f} + BM25: {bm25_score:.3f})")
            print(f"      키워드: {keywords}")
            print(f"      내용: {text_preview}")
            print()
    
    print("🎉 하이브리드 검색 테스트 완료!")

if __name__ == "__main__":
    smart_rag_test() 