"""
Memory service implementations for cross-session knowledge management.
"""

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod

from google.adk.memory import BaseMemoryService
from google.adk.memory.base_memory_service import SearchMemoryResponse
from google.adk.memory.memory_entry import MemoryEntry
from google.adk.events.event import types
from google.genai.types import Part
from google.adk.sessions import Session

from .base import BaseService
from ..config.services import MemoryServiceConfig


class MemoryService(BaseService, BaseMemoryService, ABC):
    """
    Abstract base class for memory services with platform integration.
    
    Extends ADK's BaseMemoryService with platform-specific features.
    """
    
    def __init__(self, name: str, config: MemoryServiceConfig):
        BaseService.__init__(self, name, config.model_dump())
        self.config = config
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = threading.RLock()
    
    async def _start_impl(self) -> None:
        """Initialize the memory service."""
        await self._initialize_storage()
        self._log_info("Memory service initialized")
    
    async def _stop_impl(self) -> None:
        """Cleanup memory service resources."""
        await self._cleanup_storage()
        self._log_info("Memory service cleaned up")
    
    async def _health_check_impl(self) -> tuple[bool, Dict[str, Any]]:
        """Check memory service health."""
        try:
            # Test basic search operation
            test_query = "health check test query"
            search_result = await self.search_memory("test_app", "test_user", test_query)
            
            return True, {
                "memory_entries": len(self._memory_cache),
                "storage_type": self.config.service_type.value,
                "search_functional": True,
                "last_search_results": len(search_result.memories) if search_result else 0
            }
            
        except Exception as e:
            return False, {"error": str(e)}
    
    @abstractmethod
    async def _initialize_storage(self) -> None:
        """Initialize storage backend."""
        pass
    
    @abstractmethod
    async def _cleanup_storage(self) -> None:
        """Cleanup storage backend."""
        pass
    
    def _extract_session_content(self, session: Session) -> List[Dict[str, Any]]:
        """Extract searchable content from a session."""
        content_entries = []
        
        # Extract from events
        for event in session.events:
            if hasattr(event, 'content') and event.content:
                for part in event.content.parts:
                    if part.text and len(part.text.strip()) > 10:  # Minimum content length
                        content_entries.append({
                            'type': 'event_text',
                            'content': part.text.strip(),
                            'author': getattr(event, 'author', 'unknown'),
                            'timestamp': getattr(event, 'timestamp', datetime.utcnow()).isoformat()
                        })
                    
                    # Extract function call/response information
                    if part.function_call:
                        content_entries.append({
                            'type': 'function_call',
                            'content': f"Called {part.function_call.name} with args: {part.function_call.args}",
                            'author': getattr(event, 'author', 'unknown'),
                            'timestamp': getattr(event, 'timestamp', datetime.utcnow()).isoformat()
                        })
                    
                    if part.function_response:
                        response_text = str(part.function_response.response)
                        if len(response_text) > 10:
                            content_entries.append({
                                'type': 'function_response',
                                'content': f"Tool {part.function_response.name} returned: {response_text[:500]}",
                                'author': getattr(event, 'author', 'unknown'),
                                'timestamp': getattr(event, 'timestamp', datetime.utcnow()).isoformat()
                            })
        
        # Extract important state information
        for key, value in session.state.items():
            if not key.startswith('temp:') and isinstance(value, str) and len(value) > 10:
                content_entries.append({
                    'type': 'session_state',
                    'content': f"Session state {key}: {value}",
                    'author': 'session',
                    'timestamp': session.last_update_time.isoformat()
                })
        
        return content_entries
    
    def _should_ingest_session(self, session: Session) -> bool:
        """Determine if a session should be ingested into memory."""
        criteria = self.config.ingestion_criteria
        
        # Check minimum events
        if len(session.events) < criteria.get('min_events', 5):
            return False
        
        # Check minimum duration
        if hasattr(session, 'created_time'):
            duration = (session.last_update_time - session.created_time).total_seconds()
            if duration < criteria.get('min_duration_seconds', 30):
                return False
        
        # Check for success (no error events)
        if criteria.get('exclude_error_sessions', True):
            error_events = [
                event for event in session.events
                if hasattr(event, 'error_code') and event.error_code
            ]
            if error_events:
                return False
        
        return True
    
    def _create_keywords(self, content: str) -> List[str]:
        """Extract keywords from content for search indexing."""
        if not self.config.extract_keywords:
            return []
        
        # Simple keyword extraction (in production, use more sophisticated NLP)
        words = content.lower().split()
        
        # Filter out common words and short words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
        }
        
        keywords = []
        for word in words:
            # Remove punctuation
            clean_word = ''.join(c for c in word if c.isalnum())
            if len(clean_word) > 2 and clean_word not in stop_words:
                keywords.append(clean_word)
        
        # Return top keywords (by frequency)
        from collections import Counter
        word_counts = Counter(keywords)
        return [word for word, count in word_counts.most_common(20)]


class InMemoryMemoryService(MemoryService):
    """
    In-memory implementation of memory service.
    
    Suitable for development and testing. Uses simple keyword matching for search.
    """
    
    def __init__(self, config: Optional[MemoryServiceConfig] = None):
        config = config or MemoryServiceConfig()
        super().__init__("in_memory_memory", config)
        self._storage: Dict[str, List[Dict[str, Any]]] = {}  # app_name -> memory entries
        self._keyword_index: Dict[str, Dict[str, List[str]]] = {}  # app_name -> keyword -> entry_ids
    
    async def _initialize_storage(self) -> None:
        """Initialize in-memory storage."""
        self._storage.clear()
        self._keyword_index.clear()
    
    async def _cleanup_storage(self) -> None:
        """Clear in-memory storage."""
        self._storage.clear()
        self._keyword_index.clear()
    
    async def add_session_to_memory(self, session: Session) -> None:
        """Add session content to memory."""
        if not self._should_ingest_session(session):
            self._log_debug(f"Skipping session {session.id} - doesn't meet ingestion criteria")
            return
        
        with self._cache_lock:
            app_name = session.app_name
            
            # Initialize app storage
            if app_name not in self._storage:
                self._storage[app_name] = []
                self._keyword_index[app_name] = {}
            
            # Extract content from session
            content_entries = self._extract_session_content(session)
            
            for content_entry in content_entries:
                # Create memory entry
                entry_id = str(uuid.uuid4())
                memory_entry = {
                    'id': entry_id,
                    'session_id': session.id,
                    'user_id': session.user_id,
                    'content': content_entry['content'],
                    'content_type': content_entry['type'],
                    'author': content_entry['author'],
                    'timestamp': content_entry['timestamp'],
                    'created_at': datetime.utcnow().isoformat(),
                    'keywords': self._create_keywords(content_entry['content'])
                }
                
                # Store entry
                self._storage[app_name].append(memory_entry)
                
                # Index keywords
                for keyword in memory_entry['keywords']:
                    if keyword not in self._keyword_index[app_name]:
                        self._keyword_index[app_name][keyword] = []
                    self._keyword_index[app_name][keyword].append(entry_id)
            
            # Cleanup old entries if needed
            await self._cleanup_old_entries(app_name)
            
            self._log_info(f"Added {len(content_entries)} memory entries from session {session.id}")
    
    async def search_memory(self, app_name: str, user_id: str, query: str) -> SearchMemoryResponse:
        """Search memory using keyword matching."""
        with self._cache_lock:
            if app_name not in self._storage:
                return SearchMemoryResponse()
            
            # Extract keywords from query
            query_keywords = self._create_keywords(query.lower())
            
            # Find matching entries
            entry_scores: Dict[str, float] = {}
            app_entries = self._storage[app_name]
            keyword_index = self._keyword_index[app_name]
            
            # Score entries based on keyword matches
            for keyword in query_keywords:
                if keyword in keyword_index:
                    for entry_id in keyword_index[keyword]:
                        if entry_id not in entry_scores:
                            entry_scores[entry_id] = 0
                        entry_scores[entry_id] += 1
            
            # Get top scoring entries
            sorted_entries = sorted(entry_scores.items(), key=lambda x: x[1], reverse=True)
            top_entries = sorted_entries[:self.config.similarity_top_k]
            
            # Create response entries
            results = []
            for entry_id, score in top_entries:
                # Find the actual entry
                memory_entry = None
                for entry in app_entries:
                    if entry['id'] == entry_id:
                        memory_entry = entry
                        break
                
                if memory_entry:
                    # Convert content to proper Content type
                    content = types.Content(parts=[Part(text=memory_entry['content'])])
                    
                    # Create MemoryEntry
                    memory_entry_obj = MemoryEntry(
                        content=content,
                        author=memory_entry['author'],
                        timestamp=memory_entry['timestamp']
                    )
                    results.append(memory_entry_obj)
            
            self._log_debug(f"Memory search for '{query}' returned {len(results)} results")
            return SearchMemoryResponse(memories=results)
    
    async def _cleanup_old_entries(self, app_name: str) -> None:
        """Remove old memory entries based on retention policy."""
        if not self.config.auto_cleanup_enabled:
            return
        
        cutoff_date = datetime.utcnow() - timedelta(days=self.config.memory_retention_days)
        cutoff_iso = cutoff_date.isoformat()
        
        entries = self._storage[app_name]
        original_count = len(entries)
        
        # Filter out old entries
        self._storage[app_name] = [
            entry for entry in entries
            if entry['created_at'] > cutoff_iso
        ]
        
        # Rebuild keyword index for this app
        self._keyword_index[app_name] = {}
        for entry in self._storage[app_name]:
            for keyword in entry['keywords']:
                if keyword not in self._keyword_index[app_name]:
                    self._keyword_index[app_name][keyword] = []
                self._keyword_index[app_name][keyword].append(entry['id'])
        
        removed_count = original_count - len(self._storage[app_name])
        if removed_count > 0:
            self._log_info(f"Cleaned up {removed_count} old memory entries for {app_name}")


class DatabaseMemoryService(MemoryService):
    """
    Database-backed memory service implementation.
    
    Provides persistent storage for memory entries with SQL-based search.
    """
    
    def __init__(self, config: MemoryServiceConfig):
        super().__init__("database_memory", config)
        self._connection: Optional[sqlite3.Connection] = None
        self._db_lock = threading.RLock()
    
    async def _initialize_storage(self) -> None:
        """Initialize database storage."""
        db_path = Path("data/memory.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._connection = sqlite3.connect(str(db_path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        
        # Create tables
        with self._db_lock:
            cursor = self._connection.cursor()
            
            # Memory entries table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory_entries (
                    id TEXT PRIMARY KEY,
                    app_name TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    author TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    keywords TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Full-text search virtual table
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts 
                USING fts5(content, tokenize = 'unicode61 remove_diacritics 2')
            """)
            
            # Indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memory_app_user 
                ON memory_entries (app_name, user_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memory_created 
                ON memory_entries (created_at)
            """)
            
            self._connection.commit()
    
    async def _cleanup_storage(self) -> None:
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
    
    async def add_session_to_memory(self, session: Session) -> None:
        """Add session content to database memory."""
        if not self._should_ingest_session(session):
            self._log_debug(f"Skipping session {session.id} - doesn't meet ingestion criteria")
            return
        
        with self._db_lock:
            cursor = self._connection.cursor()
            content_entries = self._extract_session_content(session)
            
            for content_entry in content_entries:
                entry_id = str(uuid.uuid4())
                keywords = self._create_keywords(content_entry['content'])
                
                # Insert into memory_entries
                cursor.execute("""
                    INSERT INTO memory_entries 
                    (id, app_name, session_id, user_id, content, content_type, author, timestamp, keywords)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry_id,
                    session.app_name,
                    session.id,
                    session.user_id,
                    content_entry['content'],
                    content_entry['type'],
                    content_entry['author'],
                    content_entry['timestamp'],
                    json.dumps(keywords)
                ))
                
                # Insert into FTS table
                cursor.execute("""
                    INSERT INTO memory_fts (content) VALUES (?)
                """, (content_entry['content'],))
            
            self._connection.commit()
            
            # Cleanup old entries
            await self._cleanup_old_entries_db(session.app_name)
            
            self._log_info(f"Added {len(content_entries)} memory entries from session {session.id}")
    
    async def search_memory(self, app_name: str, user_id: str, query: str) -> SearchMemoryResponse:
        """Search memory using database full-text search."""
        with self._db_lock:
            cursor = self._connection.cursor()
            
            # Use FTS for content search
            cursor.execute("""
                SELECT me.*, 
                       bm25(mf.content) as score
                FROM memory_entries me
                JOIN memory_fts mf ON me.content = mf.content
                WHERE me.app_name = ? 
                AND mf.content MATCH ?
                ORDER BY score DESC
                LIMIT ?
            """, (app_name, query, self.config.similarity_top_k))
            
            results = []
            for row in cursor.fetchall():
                # Normalize score (FTS scores are negative, lower is better)
                normalized_score = max(0, 1.0 + (row['score'] / 10.0))
                
                # Convert content to proper Content type
                content = types.Content(parts=[Part(text=row['content'])])
                
                # Create MemoryEntry
                memory_entry = MemoryEntry(
                    content=content,
                    author=row['author'],
                    timestamp=row['timestamp']
                )
                results.append(memory_entry)
            
            self._log_debug(f"Database memory search for '{query}' returned {len(results)} results")
            return SearchMemoryResponse(memories=results)
    
    async def _cleanup_old_entries_db(self, app_name: str) -> None:
        """Clean up old entries from database."""
        if not self.config.auto_cleanup_enabled:
            return
        
        with self._db_lock:
            cursor = self._connection.cursor()
            cutoff_date = datetime.utcnow() - timedelta(days=self.config.memory_retention_days)
            
            # Delete old entries
            cursor.execute("""
                DELETE FROM memory_entries 
                WHERE app_name = ? AND created_at < ?
            """, (app_name, cutoff_date))
            
            # Clean up FTS table (remove orphaned entries)
            cursor.execute("""
                DELETE FROM memory_fts 
                WHERE content NOT IN (SELECT content FROM memory_entries)
            """)
            
            deleted_count = cursor.rowcount
            self._connection.commit()
            
            if deleted_count > 0:
                self._log_info(f"Cleaned up {deleted_count} old memory entries for {app_name}")


class VertexAIRagMemoryService(MemoryService):
    """
    Vertex AI RAG-backed memory service implementation.
    
    Uses Google Cloud Vertex AI RAG for semantic search capabilities.
    """
    
    def __init__(self, config: MemoryServiceConfig):
        super().__init__("vertex_ai_rag_memory", config)
        self._rag_client = None
    
    async def _initialize_storage(self) -> None:
        """Initialize Vertex AI RAG client."""
        try:
            # Import Vertex AI libraries
            # from google.cloud import aiplatform
            # from vertexai.preview import rag
            
            # Initialize RAG client
            # self._rag_client = rag.RagCorpusClient()
            
            # For now, log that we would initialize Vertex AI RAG
            self._log_info(f"Would initialize Vertex AI RAG with corpus: {self.config.rag_corpus_name}")
            
        except ImportError:
            self._log_warning("Vertex AI libraries not available, using fallback memory service")
            # Fall back to in-memory implementation
            self._fallback_service = InMemoryMemoryService(self.config)
            await self._fallback_service._initialize_storage()
    
    async def _cleanup_storage(self) -> None:
        """Cleanup Vertex AI RAG resources."""
        if hasattr(self, '_fallback_service'):
            await self._fallback_service._cleanup_storage()
    
    async def add_session_to_memory(self, session: Session) -> None:
        """Add session to Vertex AI RAG corpus."""
        if not self._should_ingest_session(session):
            return
        
        if hasattr(self, '_fallback_service'):
            await self._fallback_service.add_session_to_memory(session)
            return
        
        # Implementation would use Vertex AI RAG APIs
        content_entries = self._extract_session_content(session)
        self._log_info(f"Would add {len(content_entries)} entries to Vertex AI RAG corpus")
    
    async def search_memory(self, app_name: str, user_id: str, query: str) -> SearchMemoryResponse:
        """Search using Vertex AI RAG semantic search."""
        if hasattr(self, '_fallback_service'):
            return await self._fallback_service.search_memory(app_name, user_id, query)
        
        # Implementation would use Vertex AI RAG search APIs
        self._log_info(f"Would search Vertex AI RAG corpus for: {query}")
        return SearchMemoryResponse()