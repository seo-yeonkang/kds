import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple
import fitz  # PyMuPDF
import numpy as np
import re
from utils.logger import logger
import io

class PDFProcessor:
    """PDF 문서 OCR 처리기 (PyMuPDF + EasyOCR)"""
    
    def __init__(self):
        """PDF 처리기 초기화"""
        self.supported_formats = ['.pdf']
        self._easy_ocr = None
        logger.info("PDF 처리기 초기화 완료")
    
    def _get_easy_ocr(self):
        """EasyOCR 인스턴스 지연 로딩"""
        if self._easy_ocr is None:
            try:
                logger.info("EasyOCR 모델 로딩 중... (최초 실행 시 모델 다운로드로 시간이 소요될 수 있습니다)")
                
                import easyocr
                
                # 한국어 + 영어 동시 지원
                self._easy_ocr = easyocr.Reader(['ko', 'en'], gpu=False)
                logger.info("EasyOCR 초기화 완료 (한국어 + 영어 지원)")
                        
            except Exception as e:
                logger.error(f"EasyOCR 초기화 실패: {e}")
                raise ImportError("EasyOCR 모델 로딩에 실패했습니다. 'pip install easyocr' 실행해보세요.")
        
        return self._easy_ocr
    
    def _pdf_page_to_image_array(self, page, dpi=250):
        """PDF 페이지를 numpy 이미지 배열로 변환 (EasyOCR 호환)"""
        from PIL import Image
        
        # PyMuPDF로 픽스맵 생성
        mat = fitz.Matrix(dpi/72, dpi/72)
        pix = page.get_pixmap(matrix=mat)
        
        # PIL Image로 변환
        img_data = pix.tobytes("png")
        pil_image = Image.open(io.BytesIO(img_data))
        
        # RGB로 변환 (RGBA인 경우)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # 이미지 크기 제한 (메모리 절약)
        max_size = 1600  # 적당한 크기로 설정
        if max(pil_image.size) > max_size:
            pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            logger.info(f"이미지 크기 조정: {pil_image.size}")
        
        # numpy 배열로 변환 (EasyOCR가 요구하는 형식)
        img_array = np.array(pil_image)
        
        return img_array

    def ocr_pdf_to_text(self, pdf_path: str) -> Tuple[bool, str]:
        """
        PDF 파일을 OCR하여 텍스트로 변환
        
        Args:
            pdf_path: PDF 파일 경로
            
        Returns:
            (success: bool, result: str) - 성공 여부와 텍스트 또는 에러 메시지
        """
        try:
            logger.info(f"PDF OCR 시작: {pdf_path}")
            
            # 1. PDF 문서 열기 (PyMuPDF)
            doc = fitz.open(pdf_path)
            logger.info(f"PDF 로드 완료: {len(doc)} 페이지")
            
            # 2. EasyOCR 초기화 (시간 소요 가능)
            ocr = self._get_easy_ocr()
            
            extracted_text = ""
            
            # 3. 각 페이지 처리
            for page_num in range(len(doc)):
                logger.info(f"페이지 {page_num + 1}/{len(doc)} OCR 처리 중...")
                
                page = doc.load_page(page_num)
                
                # 4. PDF 페이지 → numpy 이미지 배열 변환
                img_array = self._pdf_page_to_image_array(page, dpi=150)
                logger.debug(f"이미지 배열 크기: {img_array.shape}")
                
                # 5. 이미지 → 텍스트 OCR (EasyOCR) - 설정 최적화
                try:
                    # EasyOCR로 텍스트 추출 (최적화된 설정)
                    ocr_results = ocr.readtext(
                        img_array,
                        paragraph=False,    # 줄별로 텍스트 인식
                        width_ths=0.7,     # 단어 간격 조정
                        height_ths=0.7,    # 줄 간격 조정  
                        detail=1           # 더 자세한 결과 반환
                    )
                    
                    # 6. OCR 결과 파싱
                    page_text = self._parse_easyocr_results(ocr_results)
                    
                    if page_text.strip():
                        extracted_text += page_text + "\n"
                        logger.info(f"페이지 {page_num + 1} OCR 완료: {len(page_text)} 글자 추출")
                    else:
                        logger.warning(f"페이지 {page_num + 1}에서 텍스트를 찾을 수 없습니다")
                        
                except Exception as page_error:
                    logger.error(f"페이지 {page_num + 1} OCR 실패: {page_error}")
                    # 개별 페이지 실패해도 계속 진행
                    continue
            
            doc.close()
            
            # 7. 최종 결과 검증
            extracted_text = extracted_text.strip()
            if not extracted_text:
                logger.warning("OCR 결과가 비어있습니다.")
                return False, "OCR 처리 결과가 비어있습니다. PDF가 이미지 기반이거나 텍스트 품질이 낮을 수 있습니다."
            
            logger.info(f"PDF OCR 완료: 총 {len(extracted_text)} 글자 추출")
            logger.debug(f"추출 텍스트 미리보기: {extracted_text[:200]}...")
            return True, extracted_text
            
        except Exception as e:
            error_msg = f"PDF OCR 처리 실패: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg
    
    def _parse_easyocr_results(self, ocr_results, confidence_threshold=0.4):
        """EasyOCR 결과를 파싱하여 텍스트 추출"""
        text_lines = []
        
        try:
            # EasyOCR 결과 구조: [(bbox, text, confidence), ...]
            for result in ocr_results:
                if len(result) >= 3:
                    bbox = result[0]      # 바운딩 박스
                    text = result[1]      # 텍스트
                    confidence = result[2]  # 신뢰도
                    
                    # 신뢰도 필터링
                    if confidence >= confidence_threshold:
                        # 텍스트 정리
                        cleaned_text = re.sub(r'\s+', ' ', str(text)).strip()
                        if cleaned_text:
                            text_lines.append(cleaned_text)
                            logger.debug(f"추출된 텍스트: '{cleaned_text}' (신뢰도: {confidence:.3f})")
        
        except Exception as e:
            logger.error(f"OCR 결과 파싱 오류: {e}")
            logger.debug(f"OCR 결과 구조: {type(ocr_results)} - {ocr_results}")
            return ""
        
        # 줄바꿈으로 연결
        result_text = ' '.join(text_lines)
        logger.debug(f"파싱된 텍스트 길이: {len(result_text)}")
        return result_text
    
    def process_uploaded_file(self, uploaded_file) -> Tuple[bool, str, Optional[str]]:
        """
        업로드된 PDF 파일을 처리하여 텍스트 추출
        
        Args:
            uploaded_file: Streamlit 업로드 파일 객체
            
        Returns:
            (success: bool, text_or_error: str, filename: Optional[str])
        """
        try:
            # 파일 확장자 확인
            filename = uploaded_file.name
            file_ext = Path(filename).suffix.lower()
            
            if file_ext not in self.supported_formats:
                return False, f"지원하지 않는 파일 형식입니다: {file_ext}", None
            
            logger.info(f"PDF 파일 처리 시작: {filename}")
            
            # 임시 파일로 저장
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                # PyMuPDF + EasyOCR 처리
                success, result = self.ocr_pdf_to_text(tmp_path)
                
                if success:
                    logger.info(f"파일 처리 성공: {filename} ({len(result)} 글자 추출)")
                    return True, result, filename
                else:
                    logger.error(f"파일 처리 실패: {filename}")
                    return False, result, None
                    
            finally:
                # 임시 파일 정리
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                    
        except Exception as e:
            error_msg = f"파일 처리 중 오류 발생: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg, None
    
    def get_file_info(self, uploaded_file) -> dict:
        """업로드된 파일 정보 반환"""
        return {
            'name': uploaded_file.name,
            'size': uploaded_file.size,
            'type': uploaded_file.type,
            'size_mb': round(uploaded_file.size / 1024 / 1024, 2)
        }

# 전역 인스턴스
_pdf_processor = None

def get_pdf_processor() -> PDFProcessor:
    """PDF 처리기 싱글톤 인스턴스 반환"""
    global _pdf_processor
    if _pdf_processor is None:
        _pdf_processor = PDFProcessor()
    return _pdf_processor 