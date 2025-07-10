import sys
from pathlib import Path
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.data_loader import ContractDataLoader
from rag.embeddings import ContractEmbeddings
from rag.vector_db import ContractVectorDB

def full_embedding_process():
    """전체 데이터셋 임베딩 생성 및 벡터DB 업데이트"""
    print("🚀 전체 데이터셋 임베딩 생성 시작")
    print("="*60)
    
    # 1. 데이터 로딩
    print("📂 전체 데이터 로딩 중...")
    start_time = time.time()
    
    data_loader = ContractDataLoader()
    # all_documents = data_loader.load_all_documents()
    all_documents = data_loader.load_labor_contracts_only()
    
    load_time = time.time() - start_time
    print(f"✅ 데이터 로딩 완료: {len(all_documents)} 문서 ({load_time:.1f}초)")
    
    # 2. 임베딩 생성
    print(f"\n🧠 임베딩 생성 중... ")
    
    
    embedding_start = time.time()
    embeddings_model = ContractEmbeddings()
    embeddings = embeddings_model.create_embeddings(all_documents)
    
    embedding_time = time.time() - embedding_start
    print(f"✅ 임베딩 생성 완료: {embeddings.shape} ({embedding_time:.1f}초)")
    
    # 3. 기존 벡터DB 삭제 및 재생성
    print(f"\n💾 벡터DB 업데이트 중...")
    print("⚠️  기존 벡터DB를 삭제하고 새로 생성합니다...")
    
    # 기존 벡터DB 디렉토리 삭제
    import shutil
    vector_db_path = Path("data/vector_db")
    if vector_db_path.exists():
        shutil.rmtree(vector_db_path)
        print("🗑️  기존 벡터DB 삭제 완료")
    
    # 새 벡터DB 생성
    vector_db_start = time.time()
    vector_db = ContractVectorDB()
    vector_db.add_documents(all_documents, embeddings)
    
    vector_db_time = time.time() - vector_db_start
    print(f"✅ 벡터DB 생성 완료 ({vector_db_time:.1f}초)")
    
    # 4. 최종 확인
    print(f"\n📊 최종 결과:")
    info = vector_db.get_collection_info()
    print(f"   - 총 문서 수: {info['total_documents']}")
    print(f"   - 벡터DB 저장 위치: {info['persist_directory']}")
    print(f"   - 하이브리드 검색: {'✅' if info['hybrid_search_enabled'] else '❌'}")
    
    # 5. 시간 요약
    total_time = time.time() - start_time
    print(f"\n⏱️  소요 시간 요약:")
    print(f"   - 데이터 로딩: {load_time:.1f}초")
    print(f"   - 임베딩 생성: {embedding_time:.1f}초")
    print(f"   - 벡터DB 생성: {vector_db_time:.1f}초")
    print(f"   - 총 소요 시간: {total_time:.1f}초 ({total_time/60:.1f}분)")
    
    print(f"\n🎉 전체 임베딩 생성 완료!")
    
    
    return vector_db

if __name__ == "__main__":
    try:
        vector_db = full_embedding_process()
        
        # 간단한 테스트
        print(f"\n 빠른 검색 테스트:")
        from rag.embeddings import ContractEmbeddings
        
        test_embeddings = ContractEmbeddings()
        test_query = "근로시간은 어떻게 되나요?"
        query_emb = test_embeddings.create_single_embedding(test_query)
        
        results = vector_db.hybrid_search(test_query, query_emb, k=1)
        if results:
            result = results[0]
            print(f"   질문: {test_query}")
            print(f"   점수: {result['relevance_score']:.3f} (벡터: {result['vector_score']:.3f} + BM25: {result['keyword_score']:.3f})")
            print(f"   내용: {result['text'][:60]}...")
        
        else:
            print("❌ 검색 결과 없음")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  사용자에 의해 중단됨")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc() 