import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
from config import Config
from utils.logger import logger

class ConversationManager:
    """대화 기록 관리 클래스"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or Config.CONVERSATION_DB
        self.init_database()
    
    def init_database(self):
        """데이터베이스 초기화"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS conversations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        user_message TEXT NOT NULL,
                        ai_response TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        metadata TEXT
                    )
                ''')
                conn.commit()
                logger.info("데이터베이스 초기화 완료")
        except Exception as e:
            logger.error(f"데이터베이스 초기화 실패: {e}")
            raise
    
    def save_conversation(
        self, 
        session_id: str,
        user_message: str, 
        ai_response: str,
        metadata: Dict = None
    ) -> int:
        """대화 기록 저장"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO conversations 
                    (session_id, user_message, ai_response, metadata)
                    VALUES (?, ?, ?, ?)
                ''', (
                    session_id,
                    user_message,
                    ai_response,
                    json.dumps(metadata) if metadata else None
                ))
                
                conversation_id = cursor.lastrowid
                conn.commit()
                
                logger.info(f"대화 기록 저장 완료 - ID: {conversation_id}")
                return conversation_id
                
        except Exception as e:
            logger.error(f"대화 기록 저장 실패: {e}")
            raise
    
    def get_conversation_history(
        self, 
        session_id: str, 
        limit: int = 10
    ) -> List[Dict]:
        """세션의 대화 기록 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT user_message, ai_response, timestamp, metadata
                    FROM conversations 
                    WHERE session_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (session_id, limit))
                
                rows = cursor.fetchall()
                conversations = []
                
                for row in rows:
                    conversations.append({
                        'user_message': row[0],
                        'ai_response': row[1],
                        'timestamp': row[2],
                        'metadata': json.loads(row[3]) if row[3] else None
                    })
                
                # 시간순으로 정렬 (오래된 것부터)
                conversations.reverse()
                return conversations
                
        except Exception as e:
            logger.error(f"대화 기록 조회 실패: {e}")
            return []
    
    def get_all_sessions(self) -> List[str]:
        """모든 세션 ID 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT DISTINCT session_id
                    FROM conversations
                    ORDER BY MAX(timestamp) DESC
                ''')
                
                return [row[0] for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"세션 목록 조회 실패: {e}")
            return []
    
    def delete_session(self, session_id: str) -> bool:
        """특정 세션의 모든 대화 기록 삭제"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    DELETE FROM conversations 
                    WHERE session_id = ?
                ''', (session_id,))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                logger.info(f"세션 {session_id} 대화 기록 {deleted_count}개 삭제")
                return deleted_count > 0
                
        except Exception as e:
            logger.error(f"세션 삭제 실패: {e}")
            return False
    
    def export_conversations(self, session_id: str, output_path: str = None):
        """대화 기록을 JSON 파일로 내보내기"""
        try:
            conversations = self.get_conversation_history(session_id, limit=1000)
            
            if not conversations:
                logger.warning(f"세션 {session_id}에 대화 기록이 없습니다.")
                return None
            
            output_path = output_path or f"conversations_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(conversations, f, ensure_ascii=False, indent=2)
            
            logger.info(f"대화 기록 내보내기 완료: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"대화 기록 내보내기 실패: {e}")
            return None 