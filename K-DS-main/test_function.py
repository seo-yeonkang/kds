import json
import os

def making_json_improved(path, save_dir, max_chunk_size=1000):
    with open(path, 'r', encoding='utf-8') as f:
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
        elif format_type == '본문' and not is_article_title and article_num is not None and article_num in articles:
            # 기존 조항에 내용 추가
            articles[article_num]['content'].append(text)
        else:
            # 서문, 기명날인 등 기타 섹션
            misc_sections.append({
                'text': text,
                'type': format_type,
                'content_labels': section['content']['content_labels']
            })
    
    # 동적 청킹 함수
    def create_dynamic_chunks(title, content_list, article_num, max_size):
        """
        세부 항목들을 길이에 따라 동적으로 그룹화
        """
        chunks = []
        current_chunk_items = []
        current_chunk_size = len(title) + 1  # 제목 + 줄바꿈
        
        print(f"\n조항 {article_num} 세부 항목 분석:")
        print(f"  제목: {title} ({len(title)}자)")
        
        for i, item in enumerate(content_list):
            item_size = len(item)
            print(f"  내용{i+1}: {item[:50]}... ({item_size}자)")
            
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
                    print(f"    → 청크 {len(chunks)} 생성: {len(chunk_text)}자 (항목 {len(current_chunk_items)}개)")
                
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
            print(f"    → 청크 {len(chunks)} 생성: {len(chunk_text)}자 (항목 {len(current_chunk_items)}개)")
        
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
            article_chunks = create_dynamic_chunks(title, content_list, article_num, max_chunk_size)
            
            print(f"조항 {article_num}: 총 {len(article_chunks)}개 청크 생성")
            
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
    save_path = os.path.join(save_dir, f"{document_name}_proper.json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(final_chunks, f, ensure_ascii=False, indent=2)
    
    return final_chunks

# 테스트 실행
if __name__ == "__main__":
    print("=== 정확한 조항 구분 테스트 ===")
    result = making_json_improved('05/training_labeling/근로계약서_0005.json', '.', max_chunk_size=1000)
    
    print(f'\n총 {len(result)}개의 chunk가 생성되었습니다.')
    
    print('\n=== 기타 섹션들 ===')
    for chunk in result:
        if chunk['type'] != 'article':
            print(f"- {chunk['type']}: {chunk['text'][:50]}... ({chunk['chunk_length']}자)")
    
    print('\n=== 조항별 chunks ===')
    for chunk in result:
        if chunk['type'] == 'article':
            part_info = f" (part {chunk['chunk_part']}/{chunk['total_parts']})" if 'chunk_part' in chunk else ""
            print(f"\n--- 조항 {chunk['article_number']}{part_info} ---")
            print(f"제목: {chunk.get('article_title', 'N/A')}")
            print(f"길이: {chunk['chunk_length']}자")
            if 'items_count' in chunk:
                print(f"포함 항목 수: {chunk['items_count']}개")
            print(f"라벨: {chunk.get('content_labels', [])}")
            print(f"내용: {chunk['text'][:150]}...")
    
    print("\n" + "="*50)
    print("=== 500자 기준 비교 테스트 ===")
    result_500 = making_json_improved('05/training_labeling/근로계약서_0004.json', '.', max_chunk_size=500)
    
    article_chunks_1000 = [c for c in result if c['type'] == 'article']
    article_chunks_500 = [c for c in result_500 if c['type'] == 'article']
    
    print(f"1000자 기준: {len(article_chunks_1000)}개 조항 청크")
    print(f"500자 기준: {len(article_chunks_500)}개 조항 청크") 