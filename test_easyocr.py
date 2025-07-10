import numpy as np
from PIL import Image, ImageDraw, ImageFont

print("=== EasyOCR 간단 테스트 ===")

# 1. 간단한 텍스트 이미지 생성
print("1. 테스트 이미지 생성...")
img = Image.new('RGB', (400, 150), color='white')
draw = ImageDraw.Draw(img)

# 간단한 텍스트 그리기
try:
    # 시스템 폰트 시도
    font = ImageFont.truetype("arial.ttf", 20)
except:
    # 기본 폰트 사용
    font = ImageFont.load_default()

draw.text((20, 30), "Hello World", fill='black', font=font)
draw.text((20, 70), "안녕하세요 한국어 테스트", fill='black', font=font)
draw.text((20, 110), "EasyOCR Test 123", fill='black', font=font)

# 이미지 저장 (확인용)
img.save('test_image.png')
img_array = np.array(img)
print(f"   ✅ 테스트 이미지 생성: {img_array.shape}")
print("   📁 test_image.png 파일로 저장됨")

# 2. EasyOCR 테스트
print("2. EasyOCR 테스트...")
try:
    import easyocr
    
    print("   EasyOCR 모델 로딩 중...")
    reader = easyocr.Reader(['ko', 'en'], gpu=False, verbose=False)
    print("   ✅ EasyOCR 초기화 성공!")
    
    print("   OCR 처리 중...")
    results = reader.readtext(img_array)
    
    print(f"   ✅ OCR 처리 성공! {len(results)}개 텍스트 발견")
    
    print("\n📝 추출된 텍스트:")
    print("-" * 40)
    for i, (bbox, text, confidence) in enumerate(results):
        print(f"{i+1}. '{text}' (신뢰도: {confidence:.3f})")
    print("-" * 40)
    
    if len(results) > 0:
        print("✅ EasyOCR이 정상 작동합니다!")
    else:
        print("⚠️  텍스트가 인식되지 않았습니다.")
    
except ImportError:
    print("   ❌ EasyOCR이 설치되지 않았습니다!")
    print("   해결: 'pip install easyocr' 실행하세요")
    
except Exception as e:
    print(f"   ❌ EasyOCR 테스트 실패: {e}")

print("\n=== 테스트 완료 ===")
print("이제 실제 PDF 테스트를 해보세요! 🚀") 