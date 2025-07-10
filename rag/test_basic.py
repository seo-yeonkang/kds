import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    print("=== RAG System Test ===")
    
    # Test 1: Import modules
    print("\n1. Testing imports...")
    try:
        from rag.data_loader import ContractDataLoader
        print("   ✓ ContractDataLoader imported")
    except Exception as e:
        print(f"   ✗ ContractDataLoader failed: {e}")
        return
    
    try:
        from rag.embeddings import ContractEmbeddings
        print("   ✓ ContractEmbeddings imported")
    except Exception as e:
        print(f"   ✗ ContractEmbeddings failed: {e}")
        return
    
    try:
        from rag.vector_db import ContractVectorDB
        print("   ✓ ContractVectorDB imported")
    except Exception as e:
        print(f"   ✗ ContractVectorDB failed: {e}")
        return
    
    # Test 2: Data loading
    print("\n2. Testing data loading...")
    try:
        loader = ContractDataLoader()
        docs = loader.load_sample_data(sample_size=10)
        print(f"   ✓ Loaded {len(docs)} documents")
        
        if docs:
            print(f"   ✓ First doc keys: {list(docs[0].keys())}")
    except Exception as e:
        print(f"   ✗ Data loading failed: {e}")
        return
    
    # Test 3: Embeddings
    print("\n3. Testing embeddings...")
    try:
        embedder = ContractEmbeddings()
        info = embedder.get_model_info()
        print(f"   ✓ Model info: {info['model_name']}")
        
        # Test with small sample
        embeddings = embedder.create_embeddings(docs[:5])
        print(f"   ✓ Created embeddings: {embeddings.shape}")
    except Exception as e:
        print(f"   ✗ Embeddings failed: {e}")
        return
    
    # Test 4: Vector DB
    print("\n4. Testing vector database...")
    try:
        vector_db = ContractVectorDB()
        vector_db.add_documents(docs[:5], embeddings)
        print("   ✓ Added documents to vector DB")
        
        # Test search
        query_emb = embedder.create_single_embedding("working hours")
        results = vector_db.search(query_emb, k=3)
        print(f"   ✓ Search returned {len(results)} results")
        
        for i, result in enumerate(results):
            score = result['relevance_score']
            text = result['text'][:50] + "..."
            print(f"      {i+1}. Score: {score:.3f} - {text}")
    except Exception as e:
        print(f"   ✗ Vector DB failed: {e}")
        return
    
    print("\n=== All tests passed! ===")

if __name__ == "__main__":
    main() 