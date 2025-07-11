import sys
import os

# 프로젝트 루트 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # utils의 상위 디렉토리
sys.path.insert(0, project_root)

print(f"프로젝트 루트: {project_root}")
print(f"Python 경로에 추가됨")

from utils.pdf_processor import PDFProcessor
from utils.logger import logger
import logging

# 로깅 설정 - 더 자세한 정보 출력
logging.basicConfig(
    level=logging.INFO,  # DEBUG에서 INFO로 변경 (로그 정리)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

print("=== PDF OCR 테스트 시작 (EasyOCR 버전) ===")

# PDF 파일 경로
pdf_path = 'Altair.pdf'

# 파일 존재 확인
if not os.path.exists(pdf_path):
    print(f"오류: {pdf_path} 파일을 찾을 수 없습니다.")
    sys.exit(1)

print(f"PDF 파일: {pdf_path} 발견")

try:
    # PDF 처리기 인스턴스 생성
    print("PDF 처리기 생성 중...")
    pdf_processor = PDFProcessor()
    
    print("EasyOCR로 OCR 처리 시작...")
    # PDF 파일 OCR 처리
    success, result = pdf_processor.ocr_pdf_to_text(pdf_path)
    
    # 결과 출력
    if success:
        print(f"\n🎉 OCR 성공! 🎉")
        print(f"추출된 텍스트 길이: {len(result)} 글자")
        print(f"\n📄 추출된 전체 텍스트:")
        print("=" * 80)
        print(result)
        print("=" * 80)
            
        # 전체 텍스트 파일로 저장
        with open('extracted_text.txt', 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"\n💾 전체 텍스트가 'extracted_text.txt' 파일로도 저장되었습니다!")
        
        # 텍스트 분석 정보 출력
        lines = result.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        print(f"\n📊 텍스트 분석:")
        print(f"- 총 줄 수: {len(lines)}")
        print(f"- 빈 줄 제외 줄 수: {len(non_empty_lines)}")
        print(f"- 평균 줄 길이: {len(result) / len(non_empty_lines):.1f} 글자" if non_empty_lines else "- 텍스트 없음")
        
        # 각 줄 표시 (디버깅용)
        print(f"\n🔍 줄별 분석 (빈 줄 제외):")
        for i, line in enumerate(non_empty_lines[:10], 1):  # 처음 10줄만
            print(f"  [{i:2d}] {line}")
        if len(non_empty_lines) > 10:
            print(f"  ... ({len(non_empty_lines) - 10}개 줄 더 있음)")
        
    else:
        print(f"\n❌ OCR 실패!")
        print(f"오류 메시지: {result}")
        
except Exception as e:
    print(f"\n💥 예외 발생!")
    print(f"오류: {e}")
    import traceback
    traceback.print_exc()

print("\n=== 테스트 완료 ===")
print("현재 EasyOCR 사용 중 - 정확한 텍스트 추출 결과를 확인하세요! 🚀") 