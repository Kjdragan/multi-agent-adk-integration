"""
Session service implementations for managing ADK sessions, state, and events.
"""

import json
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from abc import ABC, abstractmethod

from google.adk.sessions import BaseSessionService, Session
from google.adk.events import Event, EventActions
from google.genai import types

from .base import BaseService
from ..config.services import SessionServiceConfig
from ..platform_logging.models import EventLog, StateChangeLog


class SessionService(BaseService, BaseSessionService, ABC):
    """
    Abstract base class for session services with platform integration.
    
    Extends ADK's BaseSessionService with platform-specific features like
    logging integration and configuration management.
    """
    
    def __init__(self, name: str, config: SessionServiceConfig):
        BaseService.__init__(self, name, config.model_dump())
        self.config = config
        self._sessions_cache: Dict[str, Session] = {}
        self._cache_lock = threading.RLock()
    
    async def _start_impl(self) -> None:
        """Initialize the session service."""
        await self._initialize_storage()
        self._log_info("Session service initialized")
    
    async def _stop_impl(self) -> None:
        """Cleanup session service resources."""
        await self._cleanup_storage()
        self._log_info("Session service cleaned up")
    
    async def _health_check_impl(self) -> tuple[bool, Dict[str, Any]]:
        """Check session service health."""
        try:
            # Test basic operations
            test_session = await self.create_session(
                app_name="health_check",
                user_id="test_user",
                session_id="test_session"
            )
            
            # Test state operations
            test_session.state["health_check"] = "ok"
            
            # Test event operations
            test_event = Event(
                author="health_check",
                invocation_id="test_invocation",
                content=types.Content(parts=[types.Part(text="Health check")])
            )
            
            await self.append_event(test_session, test_event)
            
            # Cleanup
            await self._cleanup_test_session(test_session)
            
            return True, {
                "cached_sessions": len(self._sessions_cache),
                "storage_type": self.config.service_type.value
            }
            
        except Exception as e:
            return False, {"error": str(e)}
    
    def log_event(self, session: Session, event: Event) -> None:
        """Log an event with platform logging system."""
        if not self._logger:
            return
        
        # Create event log
        event_log = EventLog(
            event_id=getattr(event, 'id', 'unknown'),
            invocation_id=getattr(event, 'invocation_id', 'unknown'),
            author=getattr(event, 'author', 'unknown'),
            event_type=self._determine_event_type(event),
            partial=getattr(event, 'partial', False),
            content_summary=self._summarize_content(event.content),
            function_calls=self._extract_function_calls(event),
            function_responses=self._extract_function_responses(event),
            state_changes=getattr(event.actions, 'state_delta', {}) if event.actions else {},
            artifacts_saved=list(getattr(event.actions, 'artifact_delta', {}).keys()) if event.actions else [],
            control_signals=self._extract_control_signals(event.actions) if event.actions else {}
        )
        
        self._logger.log_event(event_log)
    
    def log_state_change(self, session: Session, key: str, old_value: Any, 
                        new_value: Any, agent_name: str = "unknown") -> None:
        """Log a state change with platform logging system."""
        if not self._logger:
            return
        
        # Determine context type from key prefix
        context_type = "session"
        if key.startswith("user:"):
            context_type = "user"
        elif key.startswith("app:"):
            context_type = "app"
        elif key.startswith("temp:"):
            context_type = "temp"
        
        state_log = StateChangeLog(
            invocation_id=getattr(session, 'current_invocation_id', 'unknown'),
            event_id="state_change",
            state_key=key,
            old_value=old_value,
            new_value=new_value,
            change_type="update" if old_value is not None else "create",
            agent_name=agent_name,
            context_type=context_type
        )
        
        self._logger.log_state_change(state_log)
    
    @abstractmethod
    async def _initialize_storage(self) -> None:
        """Initialize storage backend."""
        pass
    
    @abstractmethod
    async def _cleanup_storage(self) -> None:
        """Cleanup storage backend."""
        pass
    
    @abstractmethod
    async def _cleanup_test_session(self, session: Session) -> None:
        """Cleanup test session created during health check."""
        pass
    
    def _determine_event_type(self, event: Event) -> str:
        """Determine the type of event for logging."""
        if hasattr(event, 'content') and event.content:
            if event.get_function_calls():
                return "function_call"
            elif event.get_function_responses():
                return "function_response"
            elif event.content.parts and event.content.parts[0].text:
                return "text"
        
        return "unknown"
    
    def _summarize_content(self, content: Optional[types.Content]) -> Optional[str]:
        """Create a summary of event content for logging."""
        if not content or not content.parts:
            return None
        
        summaries = []
        for part in content.parts[:3]:  # Limit to first 3 parts
            if part.text:
                text = part.text[:100] + "..." if len(part.text) > 100 else part.text
                summaries.append(f"text: {text}")
            elif part.function_call:
                summaries.append(f"function_call: {part.function_call.name}")
            elif part.function_response:
                summaries.append(f"function_response: {part.function_response.name}")
        
        return " | ".join(summaries)
    
    def _extract_function_calls(self, event: Event) -> List[str]:
        """Extract function call names from event."""
        calls = event.get_function_calls() if hasattr(event, 'get_function_calls') else []
        return [call.name for call in calls]
    
    def _extract_function_responses(self, event: Event) -> List[str]:
        """Extract function response names from event."""
        responses = event.get_function_responses() if hasattr(event, 'get_function_responses') else []
        return [response.name for response in responses]
    
    def _extract_control_signals(self, actions: EventActions) -> Dict[str, Any]:
        """Extract control flow signals from event actions."""
        signals = {}
        
        if hasattr(actions, 'transfer_to_agent') and actions.transfer_to_agent:
            signals["transfer_to_agent"] = actions.transfer_to_agent
        
        if hasattr(actions, 'escalate') and actions.escalate:
            signals["escalate"] = True
        
        if hasattr(actions, 'skip_summarization') and actions.skip_summarization:
            signals["skip_summarization"] = True
        
        return signals


class InMemorySessionService(SessionService):
    """
    In-memory implementation of session service.
    
    Suitable for development and testing. All data is lost on restart.
    """
    
    def __init__(self, config: Optional[SessionServiceConfig] = None):
        config = config or SessionServiceConfig()
        super().__init__("in_memory_session", config)
        self._storage: Dict[str, Dict[str, Session]] = {}  # app_name -> user_id -> session
        self._global_state: Dict[str, Dict[str, Any]] = {}  # app_name -> state
    
    async def _initialize_storage(self) -> None:
        """Initialize in-memory storage."""
        self._storage.clear()
        self._global_state.clear()
    
    async def _cleanup_storage(self) -> None:
        """Clear in-memory storage."""
        self._storage.clear()
        self._global_state.clear()
    
    async def _cleanup_test_session(self, session: Session) -> None:
        """Remove test session from memory."""
        app_sessions = self._storage.get(session.app_name, {})
        if session.user_id in app_sessions:
            del app_sessions[session.user_id]
    
    async def create_session(self, app_name: str, user_id: str, session_id: str,
                           state: Optional[Dict[str, Any]] = None) -> Session:
        """Create a new session."""
        with self._cache_lock:
            # Initialize app storage if needed
            if app_name not in self._storage:
                self._storage[app_name] = {}
                self._global_state[app_name] = {}
            
            # Create session
            session = Session(
                app_name=app_name,
                user_id=user_id,
                id=session_id,
                state=state or {},
                events=[],
                last_update_time=datetime.utcnow()
            )
            
            # Store session
            self._storage[app_name][f"{user_id}:{session_id}"] = session
            self._sessions_cache[session_id] = session
            
            self._log_debug(f"Created session {session_id} for user {user_id}")
            return session
    
    async def get_session(self, app_name: str, user_id: str, session_id: str) -> Optional[Session]:
        """Get an existing session."""
        with self._cache_lock:
            session_key = f"{user_id}:{session_id}"
            app_sessions = self._storage.get(app_name, {})
            return app_sessions.get(session_key)
    
    async def append_event(self, session: Session, event: Event) -> None:
        """Append an event to the session."""
        with self._cache_lock:
            # Add event to session
            session.events.append(event)
            session.last_update_time = datetime.utcnow()
            
            # Apply state changes if present
            if event.actions and hasattr(event.actions, 'state_delta'):
                old_state = session.state.copy()
                for key, value in event.actions.state_delta.items():
                    old_value = session.state.get(key)
                    session.state[key] = value
                    
                    # Log state change
                    self.log_state_change(
                        session, key, old_value, value,
                        getattr(event, 'author', 'unknown')
                    )
            
            # Log the event
            self.log_event(session, event)
    
    async def update_session_state(self, session: Session, state_delta: Dict[str, Any]) -> None:
        """Update session state."""
        with self._cache_lock:
            for key, value in state_delta.items():
                old_value = session.state.get(key)
                session.state[key] = value
                
                # Log state change
                self.log_state_change(session, key, old_value, value, "session_service")
            
            session.last_update_time = datetime.utcnow()
    
    async def delete_session(self, app_name: str, user_id: str, session_id: str) -> bool:
        """Delete a session."""
        with self._cache_lock:
            session_key = f"{user_id}:{session_id}"
            app_sessions = self._storage.get(app_name, {})
            
            if session_key in app_sessions:
                del app_sessions[session_key]
                self._sessions_cache.pop(session_id, None)
                self._log_debug(f"Deleted session {session_id}")
                return True
            
            return False
    
    async def list_sessions(self, app_name: str, user_id: Optional[str] = None) -> List[Session]:
        """List sessions for an app/user."""
        with self._cache_lock:
            app_sessions = self._storage.get(app_name, {})
            
            if user_id:
                # Filter by user
                return [
                    session for key, session in app_sessions.items()
                    if key.startswith(f"{user_id}:")
                ]
            else:
                # All sessions for app
                return list(app_sessions.values())


class DatabaseSessionService(SessionService):
    """
    Database-backed session service implementation.
    
    Provides persistent storage for sessions, state, and events.
    """
    
    def __init__(self, config: SessionServiceConfig):
        super().__init__("database_session", config)
        self._connection: Optional[sqlite3.Connection] = None
        self._db_lock = threading.RLock()
    
    async def _initialize_storage(self) -> None:
        """Initialize database storage."""
        # For now, use SQLite for simplicity
        # In production, this would connect to PostgreSQL/Cloud SQL
        db_path = Path("data/sessions.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._connection = sqlite3.connect(str(db_path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        
        # Create tables
        with self._db_lock:
            cursor = self._connection.cursor()
            
            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    app_name TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    state TEXT NOT NULL DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (app_name, user_id, session_id)
                )
            """)
            
            # Events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS session_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    app_name TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    event_data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (app_name, session_id) REFERENCES sessions (app_name, session_id)
                )
            """)
            
            # Indexes for better performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_user 
                ON sessions (app_name, user_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_session 
                ON session_events (app_name, session_id)
            """)
            
            self._connection.commit()
    
    async def _cleanup_storage(self) -> None:
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
    
    async def _cleanup_test_session(self, session: Session) -> None:
        """Remove test session from database."""
        await self.delete_session(session.app_name, session.user_id, session.id)
    
    async def create_session(self, app_name: str, user_id: str, session_id: str,
                           state: Optional[Dict[str, Any]] = None) -> Session:
        """Create a new session in database."""
        with self._db_lock:
            cursor = self._connection.cursor()
            
            state_json = json.dumps(state or {})
            now = datetime.utcnow()
            
            cursor.execute("""
                INSERT OR REPLACE INTO sessions 
                (app_name, user_id, session_id, state, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (app_name, user_id, session_id, state_json, now, now))
            
            self._connection.commit()
            
            # Create session object
            session = Session(
                app_name=app_name,
                user_id=user_id,
                id=session_id,
                state=state or {},
                events=[],
                last_update_time=now
            )
            
            # Load existing events
            await self._load_session_events(session)
            
            self._sessions_cache[session_id] = session
            return session
    
    async def get_session(self, app_name: str, user_id: str, session_id: str) -> Optional[Session]:
        """Get session from database."""
        # Check cache first
        if session_id in self._sessions_cache:
            return self._sessions_cache[session_id]
        
        with self._db_lock:
            cursor = self._connection.cursor()
            cursor.execute("""
                SELECT * FROM sessions 
                WHERE app_name = ? AND user_id = ? AND session_id = ?
            """, (app_name, user_id, session_id))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            # Create session object
            state = json.loads(row['state'])
            session = Session(
                app_name=app_name,
                user_id=user_id,
                id=session_id,
                state=state,
                events=[],
                last_update_time=datetime.fromisoformat(row['updated_at'])
            )
            
            # Load events
            await self._load_session_events(session)
            
            self._sessions_cache[session_id] = session
            return session
    
    async def append_event(self, session: Session, event: Event) -> None:
        """Append event to session in database."""
        with self._db_lock:
            cursor = self._connection.cursor()
            
            # Serialize event
            event_data = {
                'id': getattr(event, 'id', None),
                'author': getattr(event, 'author', None),
                'invocation_id': getattr(event, 'invocation_id', None),
                'content': event.content.model_dump() if event.content else None,
                'actions': event.actions.model_dump() if event.actions else None,
                'timestamp': getattr(event, 'timestamp', datetime.utcnow().isoformat())
            }
            
            # Insert event
            cursor.execute("""
                INSERT INTO session_events (app_name, session_id, event_data)
                VALUES (?, ?, ?)
            """, (session.app_name, session.id, json.dumps(event_data)))
            
            # Apply state changes
            if event.actions and hasattr(event.actions, 'state_delta'):
                for key, value in event.actions.state_delta.items():
                    old_value = session.state.get(key)
                    session.state[key] = value
                    
                    # Log state change
                    self.log_state_change(
                        session, key, old_value, value,
                        getattr(event, 'author', 'unknown')
                    )
                
                # Update session state in database
                state_json = json.dumps(session.state)
                cursor.execute("""
                    UPDATE sessions 
                    SET state = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE app_name = ? AND user_id = ? AND session_id = ?
                """, (state_json, session.app_name, session.user_id, session.id))
            
            self._connection.commit()
            
            # Add to session object
            session.events.append(event)
            session.last_update_time = datetime.utcnow()
            
            # Log the event
            self.log_event(session, event)
    
    async def _load_session_events(self, session: Session) -> None:
        """Load events for a session from database."""
        with self._db_lock:
            cursor = self._connection.cursor()
            cursor.execute("""
                SELECT event_data FROM session_events 
                WHERE app_name = ? AND session_id = ?
                ORDER BY id
            """, (session.app_name, session.id))
            
            for row in cursor.fetchall():
                event_data = json.loads(row['event_data'])
                
                # Reconstruct event (simplified)
                # In a full implementation, you'd properly deserialize all event types
                event = Event(
                    author=event_data.get('author', 'unknown'),
                    invocation_id=event_data.get('invocation_id', ''),
                    content=types.Content.model_validate(event_data['content']) if event_data.get('content') else None
                )
                
                session.events.append(event)
    
    async def delete_session(self, app_name: str, user_id: str, session_id: str) -> bool:
        """Delete session from database."""
        with self._db_lock:
            cursor = self._connection.cursor()
            
            # Delete events first (foreign key constraint)
            cursor.execute("""
                DELETE FROM session_events 
                WHERE app_name = ? AND session_id = ?
            """, (app_name, session_id))
            
            # Delete session
            cursor.execute("""
                DELETE FROM sessions 
                WHERE app_name = ? AND user_id = ? AND session_id = ?
            """, (app_name, user_id, session_id))
            
            deleted = cursor.rowcount > 0
            self._connection.commit()
            
            # Remove from cache
            self._sessions_cache.pop(session_id, None)
            
            return deleted
    
    async def list_sessions(self, app_name: str, user_id: Optional[str] = None) -> List[Session]:
        """List sessions from database."""
        with self._db_lock:
            cursor = self._connection.cursor()
            
            if user_id:
                cursor.execute("""
                    SELECT * FROM sessions 
                    WHERE app_name = ? AND user_id = ?
                    ORDER BY updated_at DESC
                """, (app_name, user_id))
            else:
                cursor.execute("""
                    SELECT * FROM sessions 
                    WHERE app_name = ?
                    ORDER BY updated_at DESC
                """, (app_name,))
            
            sessions = []
            for row in cursor.fetchall():
                state = json.loads(row['state'])
                session = Session(
                    app_name=row['app_name'],
                    user_id=row['user_id'],
                    id=row['session_id'],
                    state=state,
                    events=[],
                    last_update_time=datetime.fromisoformat(row['updated_at'])
                )
                sessions.append(session)
            
            return sessions


class VertexAISessionService(DatabaseSessionService):
    """
    Vertex AI-enhanced session service.
    
    Extends database session service with Vertex AI integrations.
    """
    
    def __init__(self, config: SessionServiceConfig):
        super().__init__(config)
        self.name = "vertex_ai_session"
        # Additional Vertex AI specific initialization would go here
    
    async def _health_check_impl(self) -> tuple[bool, Dict[str, Any]]:
        """Enhanced health check with Vertex AI status."""
        base_healthy, base_details = await super()._health_check_impl()
        
        # Add Vertex AI specific health checks here
        vertex_details = {
            "vertex_ai_project": self.config.vertex_ai_project,
            "vertex_ai_location": self.config.vertex_ai_location
        }
        
        return base_healthy, {**base_details, **vertex_details}