import torch
import re
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from peft import PeftModel
from typing import Optional

class M2M100TranslationService:
    def __init__(self, model_path: str):
        """
        Args:
            model_path: 파인튜닝된 모델이 저장된 경로
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """모델과 토크나이저 로딩"""
        print(f"🔄 모델 로딩: {self.model_path}")
        
        # 베이스 모델 로딩
        base_model = M2M100ForConditionalGeneration.from_pretrained(
            "facebook/m2m100_418M",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # LoRA 모델 로딩
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        self.model.eval()
        
        # 토크나이저 로딩py 
        self.tokenizer = M2M100Tokenizer.from_pretrained(self.model_path)
        
        print(f"✅ 모델 로딩 완료 (device: {self.device})")
    
    def _detect_language(self, text: str) -> str:
        """간단한 언어 감지 (한글/중국어)"""
        # 한글 문자 패턴
        korean_pattern = re.compile(r'[가-힣]')
        # 중국어 문자 패턴  
        chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
        
        korean_count = len(korean_pattern.findall(text))
        chinese_count = len(chinese_pattern.findall(text))
        
        if korean_count > chinese_count:
            return "ko"
        else:
            return "zh"
    
    def translate(self, text: str, source_lang: Optional[str] = None, target_lang: Optional[str] = None) -> str:
        """
        텍스트 번역
        
        Args:
            text: 번역할 텍스트
            source_lang: 소스 언어 ("ko" 또는 "zh", None이면 자동 감지)
            target_lang: 타겟 언어 ("ko" 또는 "zh", None이면 자동 결정)
        
        Returns:
            번역된 텍스트
        """
        if not text or not text.strip():
            return text
        
        # 언어 자동 감지
        if source_lang is None:
            source_lang = self._detect_language(text)
        
        # 타겟 언어 자동 결정
        if target_lang is None:
            target_lang = "zh" if source_lang == "ko" else "ko"
        
        # 같은 언어면 그대로 반환
        if source_lang == target_lang:
            return text
        
        try:
            # 토큰화
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=128,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # 번역 생성
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    num_beams=3,
                    length_penalty=1.0,
                    early_stopping=True,
                    do_sample=False
                )
            
            # 디코딩
            translated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 입력 텍스트가 출력에 포함된 경우 제거
            if text in translated:
                translated = translated.replace(text, "").strip()
            
            return translated if translated else text
            
        except Exception as e:
            print(f"❌ 번역 오류: {e}")
            return text  # 오류 시 원본 반환

# 사용 예시
if __name__ == "__main__":
    # 번역 서비스 초기화
    translator = M2M100TranslationService("translation_m2m") #lora 경로로 변경
    
    # 테스트
    korean_text = "근로계약서에서 초과근무수당은 어떻게 계산되나요?"
    chinese_text = "加班费是怎么计算的？"
    
    # 한국어 → 중국어
    result1 = translator.translate(korean_text)
    print(f"🇰🇷 → 🇨🇳: {korean_text}")
    print(f"결과: {result1}")
    
    # 중국어 → 한국어  
    result2 = translator.translate(chinese_text)
    print(f"🇨🇳 → 🇰🇷: {chinese_text}")
    print(f"결과: {result2}")
    
    # 명시적 언어 지정
    result3 = translator.translate("你好", source_lang="zh", target_lang="ko")
    print(f"명시적 번역: {result3}")