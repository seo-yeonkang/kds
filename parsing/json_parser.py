import json
import os
import re
from pathlib import Path

def parse_contract_json(input_path, output_dir, max_chunk_size=1000):
    """
    계약서 JSON 파일을 RAG용 chunk로 파싱
    
    Args:
        input_path: 입력 JSON 파일 경로
        output_dir: 출력 디렉토리 경로
        max_chunk_size: 최대 chunk 크기 (기본값: 1000자)
    
    Returns:
        list: 파싱된 chunks 리스트
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 메타데이터 추출
    file_info = data['document']['metadata']['file_info']
    information = {
        'document_title': file_info['document_name'],
        'document_type': data['document']['type'],
        'document_date': file_info['document_creation_date'],
        'document_category': file_info['document_category']['sub_category']
    }
    information.update(file_info['additional_info'])
    
    # 조항별 그룹화 (is_article_title과 article 필드 활용)
    articles = {}
    misc_sections = []
    
    for section in data['document']['sections']:
        text = section['content']['description']
        format_info = section['format']
        format_type = format_info['format_type']
        is_article_title = format_info.get('is_article_title', False)
        article_num = format_info.get('article')
        
        if format_type == '본문' and is_article_title and article_num is not None:
            # 조항 제목 시작
            articles[article_num] = {
                'title': text,
                'content': [],
                'type': 'article',
                'content_labels': section['content']['content_labels']
            }
        elif format_type == '본문' and not is_article_title and article_num is not None:
            # is_article_title이 false이지만 숫자로 시작하는 조항일 수 있음
            if re.match(r'^\d+\.', text.strip()) and article_num not in articles:
                # 새로운 조항 제목으로 처리
                articles[article_num] = {
                    'title': text,
                    'content': [],
                    'type': 'article',
                    'content_labels': section['content']['content_labels']
                }
            elif article_num in articles:
                # 기존 조항에 내용 추가
                articles[article_num]['content'].append(text)
            else:
                # 기타 섹션으로 처리
                misc_sections.append({
                    'text': text,
                    'type': format_type,
                    'content_labels': section['content']['content_labels']
                })
        else:
            # 서문, 기명날인 등 기타 섹션
            misc_sections.append({
                'text': text,
                'type': format_type,
                'content_labels': section['content']['content_labels']
            })
    
    # 동적 청킹 함수
    def create_dynamic_chunks(title, content_list, max_size):
        """
        세부 항목들을 길이에 따라 동적으로 그룹화
        """
        chunks = []
        current_chunk_items = []
        current_chunk_size = len(title) + 1  # 제목 + 줄바꿈
        
        for item in content_list:
            item_size = len(item)
            
            # 현재 청크에 추가했을 때 크기 확인
            if current_chunk_size + item_size + 1 <= max_size:  # +1은 줄바꿈
                current_chunk_items.append(item)
                current_chunk_size += item_size + 1
            else:
                # 현재 청크 완료
                if current_chunk_items:
                    chunk_text = title + '\n' + '\n'.join(current_chunk_items)
                    chunks.append({
                        'text': chunk_text,
                        'items': current_chunk_items.copy(),
                        'size': len(chunk_text)
                    })
                
                # 새로운 청크 시작
                current_chunk_items = [item]
                current_chunk_size = len(title) + len(" (계속)") + item_size + 2  # +2는 줄바꿈들
        
        # 마지막 청크 처리
        if current_chunk_items:
            chunk_text = title + (" (계속)" if chunks else "") + '\n' + '\n'.join(current_chunk_items)
            chunks.append({
                'text': chunk_text,
                'items': current_chunk_items.copy(),
                'size': len(chunk_text)
            })
        
        return chunks
    
    # 스마트 청킹 적용
    final_chunks = []
    
    # 먼저 기타 섹션들 추가 (서문, 제목 등)
    for misc in misc_sections:
        final_chunks.append({
            'text': misc['text'],
            'chunk_length': len(misc['text']),
            'type': misc['type'],
            'content_labels': misc['content_labels'],
            **information
        })
    
    # 조항별 처리 (번호 순서대로)
    for article_num in sorted(articles.keys()):
        article_data = articles[article_num]
        title = article_data['title']
        content_list = article_data['content']
        
        if not content_list:
            # 제목만 있는 경우
            final_chunks.append({
                'text': title,
                'article_number': article_num,
                'article_title': title,
                'chunk_length': len(title),
                'type': 'article',
                'content_labels': article_data['content_labels'],
                **information
            })
        else:
            # 동적 청킹 적용
            article_chunks = create_dynamic_chunks(title, content_list, max_chunk_size)
            
            for i, chunk_data in enumerate(article_chunks):
                chunk_info = {
                    'text': chunk_data['text'],
                    'article_number': article_num,
                    'article_title': title,
                    'chunk_part': i + 1,
                    'total_parts': len(article_chunks),
                    'chunk_length': chunk_data['size'],
                    'items_count': len(chunk_data['items']),
                    'type': 'article',
                    'content_labels': article_data['content_labels'],
                    **information
                }
                final_chunks.append(chunk_info)
    
    # 저장
    document_name = information['document_title']
    output_path = os.path.join(output_dir, f"{document_name}_parsed.json")
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_chunks, f, ensure_ascii=False, indent=2)
    
    return final_chunks

def batch_parse_contracts(input_dir, output_dir, max_chunk_size=1000):
    """
    디렉토리 내 모든 JSON 파일을 배치 처리
    
    Args:
        input_dir: 입력 디렉토리 경로
        output_dir: 출력 디렉토리 경로
        max_chunk_size: 최대 chunk 크기 (기본값: 1000자)
    
    Returns:
        dict: 처리 결과 통계
    """
    input_path = Path(input_dir)
    json_files = list(input_path.glob("*.json"))
    
    if not json_files:
        print(f"❌ {input_dir}에서 JSON 파일을 찾을 수 없습니다.")
        return {}
    
    print(f"📁 {len(json_files)}개의 JSON 파일을 처리합니다...")
    
    results = {
        'total_files': len(json_files),
        'processed_files': 0,
        'failed_files': 0,
        'total_chunks': 0,
        'failed_list': []
    }
    
    for json_file in json_files:
        try:
            print(f"🔄 처리중: {json_file.name}")
            chunks = parse_contract_json(str(json_file), output_dir, max_chunk_size)
            results['processed_files'] += 1
            results['total_chunks'] += len(chunks)
            print(f"✅ {json_file.name}: {len(chunks)}개 chunks 생성")
            
        except Exception as e:
            print(f"❌ {json_file.name} 처리 실패: {str(e)}")
            results['failed_files'] += 1
            results['failed_list'].append({
                'file': json_file.name,
                'error': str(e)
            })
    
    print(f"\n📊 처리 완료 결과:")
    print(f"   - 총 파일 수: {results['total_files']}")
    print(f"   - 성공: {results['processed_files']}")
    print(f"   - 실패: {results['failed_files']}")
    print(f"   - 총 chunks: {results['total_chunks']}")
    
    if results['failed_list']:
        print(f"\n❌ 실패한 파일들:")
        for failed in results['failed_list']:
            print(f"   - {failed['file']}: {failed['error']}")
    
    return results

if __name__ == "__main__":
    # 배치 처리 실행
    input_directory = "../data/training_labeling"
    output_directory = "../data/rag_labeling"
    
    print("🚀 계약서 JSON 파일 배치 처리 시작")
    print(f"📂 입력 디렉토리: {input_directory}")
    print(f"📂 출력 디렉토리: {output_directory}")
    print(f"📏 최대 청크 크기: 1000자")
    print("-" * 50)
    
    results = batch_parse_contracts(input_directory, output_directory, max_chunk_size=1000)
    
    print("\n🎉 처리 완료!") 