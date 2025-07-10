# utils/translation_service.py
from pathlib import Path
from functools import lru_cache
from translation import M2M100TranslationService  # ← 업로드된 파일 재사용

_MODEL_DIR = Path("models/translation_m2m")  # config에서 override 가능

@lru_cache(maxsize=1)
def get_translator():
    try:
        return M2M100TranslationService(str(_MODEL_DIR))
    except Exception as e:  # GPU 미탑재·모델 미존재 등
        print(f"[Translator] Disabled ⇒ {e}")
        return None
