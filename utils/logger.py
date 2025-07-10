import logging
import sys
from datetime import datetime
from pathlib import Path
from config import Config

def setup_logger(name: str = "law_agent"):
    """로거 설정"""
    
    # 로그 디렉토리 생성
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # 로거 생성
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, Config.LOG_LEVEL))
    
    # 이미 핸들러가 있으면 제거``
    if logger.handlers:
        logger.handlers.clear()
    
    # 포맷터 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 파일 핸들러
    today = datetime.now().strftime("%Y%m%d")
    file_handler = logging.FileHandler(
        log_dir / f"law_agent_{today}.log",
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

# 기본 로거 인스턴스
logger = setup_logger()

def log_conversation(user_message: str, ai_response: str, session_id: str = None):
    """대화 로그 기록"""
    logger.info(f"[CONVERSATION] Session: {session_id}")
    logger.info(f"[USER] {user_message}")
    logger.info(f"[AI] {ai_response}")

def log_error(error: Exception, context: str = ""):
    """에러 로그 기록"""
    logger.error(f"[ERROR] {context}: {str(error)}", exc_info=True)

def log_agent_action(action: str, details: dict = None):
    """Agent 액션 로그 기록"""
    log_msg = f"[AGENT] {action}"
    if details:
        log_msg += f" - Details: {details}"
    logger.info(log_msg) 