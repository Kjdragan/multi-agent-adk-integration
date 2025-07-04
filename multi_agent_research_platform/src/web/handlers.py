"""
WebSocket and Event Handlers

Real-time communication handlers for the web interface including
WebSocket connections, event broadcasting, and log handling.
"""

import json
import time
import asyncio
from typing import Any, Dict, List, Optional, Set, Callable
from dataclasses import dataclass, field
from fastapi import WebSocket, WebSocketDisconnect

from .config import WebSocketConfig, DebugConfig
from ..platform_logging import RunLogger


@dataclass
class WebSocketConnection:
    """Represents a WebSocket connection."""
    websocket: WebSocket
    client_id: str
    connected_at: float = field(default_factory=time.time)
    last_ping: Optional[float] = None
    subscribed_channels: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WebSocketMessage:
    """WebSocket message structure."""
    type: str
    channel: str
    data: Any
    timestamp: float = field(default_factory=time.time)
    client_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "type": self.type,
            "channel": self.channel,
            "data": self.data,
            "timestamp": self.timestamp,
            "client_id": self.client_id,
        }


class WebSocketHandler:
    """Handles WebSocket connections and real-time communication."""
    
    def __init__(self,
                 config: WebSocketConfig,
                 logger: Optional[RunLogger] = None):
        
        self.config = config
        self.logger = logger
        
        # Connection management
        self.connections: Dict[str, WebSocketConnection] = {}
        self.channels: Dict[str, Set[str]] = {}  # channel -> set of client_ids
        
        # Message handling
        self.message_handlers: Dict[str, Callable] = {}
        self.message_queue: List[WebSocketMessage] = []
        
        # State tracking
        self.is_running = False
        self.total_connections = 0
        self.total_messages_sent = 0
        self.total_messages_received = 0
        
        # Background tasks
        self.ping_task = None
        self.message_processor_task = None
        
        # Register default message handlers
        self._register_default_handlers()
    
    async def start(self) -> None:
        """Start the WebSocket handler."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start background tasks
        if self.config.ping_interval_seconds > 0:
            self.ping_task = asyncio.create_task(self._ping_loop())
        
        self.message_processor_task = asyncio.create_task(self._message_processor_loop())
        
        if self.logger:
            self.logger.info("WebSocket handler started")
    
    async def stop(self) -> None:
        """Stop the WebSocket handler."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel background tasks
        if self.ping_task:
            self.ping_task.cancel()
        
        if self.message_processor_task:
            self.message_processor_task.cancel()
        
        # Close all connections
        for connection in list(self.connections.values()):
            try:
                await connection.websocket.close()
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Error closing WebSocket connection: {e}")
        
        self.connections.clear()
        self.channels.clear()
        
        if self.logger:
            self.logger.info("WebSocket handler stopped")
    
    async def handle_connection(self, websocket: WebSocket) -> None:
        """Handle a new WebSocket connection."""
        # Check connection limit
        if len(self.connections) >= self.config.max_connections:
            await websocket.close(code=1013, reason="Too many connections")
            return
        
        await websocket.accept()
        
        # Create connection
        client_id = f"client_{int(time.time() * 1000)}_{len(self.connections)}"
        connection = WebSocketConnection(
            websocket=websocket,
            client_id=client_id
        )
        
        self.connections[client_id] = connection
        self.total_connections += 1
        
        if self.logger:
            self.logger.info(f"WebSocket client connected: {client_id}")
        
        try:
            # Send welcome message
            await self._send_message_to_client(
                client_id,
                WebSocketMessage(
                    type="connection",
                    channel="system",
                    data={
                        "status": "connected",
                        "client_id": client_id,
                        "server_time": time.time(),
                        "supported_events": self.config.supported_events,
                    }
                )
            )
            
            # Handle incoming messages
            while self.is_running:
                try:
                    # Receive message with timeout
                    message_data = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=self.config.ping_timeout_seconds
                    )
                    
                    await self._handle_incoming_message(client_id, message_data)
                    self.total_messages_received += 1
                    
                except asyncio.TimeoutError:
                    # Send ping to keep connection alive
                    await self._ping_client(client_id)
                    
                except WebSocketDisconnect:
                    break
                    
        except Exception as e:
            if self.logger:
                self.logger.error(f"WebSocket connection error for {client_id}: {e}")
        
        finally:
            # Clean up connection
            await self._disconnect_client(client_id)
    
    async def _handle_incoming_message(self, client_id: str, message_data: str) -> None:
        """Handle incoming message from client."""
        try:
            message_json = json.loads(message_data)
            message_type = message_json.get("type")
            
            if message_type in self.message_handlers:
                await self.message_handlers[message_type](client_id, message_json)
            else:
                if self.logger:
                    self.logger.warning(f"Unknown message type: {message_type}")
                
        except json.JSONDecodeError as e:
            if self.logger:
                self.logger.warning(f"Invalid JSON from client {client_id}: {e}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error handling message from {client_id}: {e}")
    
    async def _disconnect_client(self, client_id: str) -> None:
        """Disconnect and clean up a client."""
        if client_id not in self.connections:
            return
        
        connection = self.connections[client_id]
        
        # Remove from all channels
        for channel_clients in self.channels.values():
            channel_clients.discard(client_id)
        
        # Remove connection
        del self.connections[client_id]
        
        if self.logger:
            self.logger.info(f"WebSocket client disconnected: {client_id}")
    
    async def _send_message_to_client(self, client_id: str, message: WebSocketMessage) -> bool:
        """Send message to specific client."""
        if client_id not in self.connections:
            return False
        
        connection = self.connections[client_id]
        
        try:
            message_dict = message.to_dict()
            await connection.websocket.send_text(json.dumps(message_dict))
            self.total_messages_sent += 1
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to send message to {client_id}: {e}")
            
            # Remove disconnected client
            await self._disconnect_client(client_id)
            return False
    
    async def broadcast_message(self, message: WebSocketMessage) -> int:
        """Broadcast message to all clients or specific channel."""
        sent_count = 0
        
        # Determine target clients
        if message.channel == "all" or message.channel not in self.channels:
            target_clients = list(self.connections.keys())
        else:
            target_clients = list(self.channels[message.channel])
        
        # Send to all target clients
        for client_id in target_clients:
            success = await self._send_message_to_client(client_id, message)
            if success:
                sent_count += 1
        
        return sent_count
    
    async def send_event(self, event_type: str, data: Any, channel: str = "events") -> int:
        """Send event to all subscribed clients."""
        message = WebSocketMessage(
            type="event",
            channel=channel,
            data={
                "event_type": event_type,
                "payload": data
            }
        )
        
        return await self.broadcast_message(message)
    
    def register_message_handler(self, message_type: str, handler: Callable) -> None:
        """Register a message handler for specific message type."""
        self.message_handlers[message_type] = handler
    
    def _register_default_handlers(self) -> None:
        """Register default message handlers."""
        
        async def handle_subscribe(client_id: str, message_data: Dict[str, Any]) -> None:
            """Handle channel subscription."""
            channel = message_data.get("channel")
            if not channel:
                return
            
            if channel not in self.channels:
                self.channels[channel] = set()
            
            self.channels[channel].add(client_id)
            
            if client_id in self.connections:
                self.connections[client_id].subscribed_channels.add(channel)
            
            # Send confirmation
            await self._send_message_to_client(
                client_id,
                WebSocketMessage(
                    type="subscription_confirmed",
                    channel="system",
                    data={"channel": channel}
                )
            )
        
        async def handle_unsubscribe(client_id: str, message_data: Dict[str, Any]) -> None:
            """Handle channel unsubscription."""
            channel = message_data.get("channel")
            if not channel:
                return
            
            if channel in self.channels:
                self.channels[channel].discard(client_id)
            
            if client_id in self.connections:
                self.connections[client_id].subscribed_channels.discard(channel)
            
            # Send confirmation
            await self._send_message_to_client(
                client_id,
                WebSocketMessage(
                    type="unsubscription_confirmed",
                    channel="system",
                    data={"channel": channel}
                )
            )
        
        async def handle_ping(client_id: str, message_data: Dict[str, Any]) -> None:
            """Handle ping message."""
            await self._send_message_to_client(
                client_id,
                WebSocketMessage(
                    type="pong",
                    channel="system",
                    data={"timestamp": time.time()}
                )
            )
        
        # Register handlers
        self.register_message_handler("subscribe", handle_subscribe)
        self.register_message_handler("unsubscribe", handle_unsubscribe)
        self.register_message_handler("ping", handle_ping)
    
    async def _ping_loop(self) -> None:
        """Background task to ping clients periodically."""
        while self.is_running:
            try:
                current_time = time.time()
                
                for client_id in list(self.connections.keys()):
                    connection = self.connections.get(client_id)
                    if not connection:
                        continue
                    
                    # Check if ping is needed
                    if (connection.last_ping is None or 
                        current_time - connection.last_ping > self.config.ping_interval_seconds):
                        
                        await self._ping_client(client_id)
                
                await asyncio.sleep(self.config.ping_interval_seconds / 2)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Ping loop error: {e}")
                await asyncio.sleep(5)
    
    async def _ping_client(self, client_id: str) -> None:
        """Send ping to specific client."""
        if client_id not in self.connections:
            return
        
        connection = self.connections[client_id]
        connection.last_ping = time.time()
        
        success = await self._send_message_to_client(
            client_id,
            WebSocketMessage(
                type="ping",
                channel="system",
                data={"timestamp": connection.last_ping}
            )
        )
        
        if not success:
            await self._disconnect_client(client_id)
    
    async def _message_processor_loop(self) -> None:
        """Background task to process queued messages."""
        while self.is_running:
            try:
                if self.message_queue:
                    # Process queued messages
                    messages_to_process = self.message_queue.copy()
                    self.message_queue.clear()
                    
                    for message in messages_to_process:
                        await self.broadcast_message(message)
                
                await asyncio.sleep(0.1)  # Process queue every 100ms
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Message processor error: {e}")
                await asyncio.sleep(1)
    
    def queue_message(self, message: WebSocketMessage) -> None:
        """Queue message for broadcasting."""
        if len(self.message_queue) < 1000:  # Prevent memory issues
            self.message_queue.append(message)
    
    def get_status(self) -> Dict[str, Any]:
        """Get WebSocket handler status."""
        return {
            "is_running": self.is_running,
            "connected_clients": len(self.connections),
            "total_connections": self.total_connections,
            "active_channels": len(self.channels),
            "total_messages_sent": self.total_messages_sent,
            "total_messages_received": self.total_messages_received,
            "queued_messages": len(self.message_queue),
            "config": {
                "max_connections": self.config.max_connections,
                "ping_interval": self.config.ping_interval_seconds,
                "supported_events": self.config.supported_events,
            }
        }


class EventHandler:
    """Handles event broadcasting and notification."""
    
    def __init__(self,
                 websocket_handler: Optional[WebSocketHandler] = None,
                 logger: Optional[RunLogger] = None):
        
        self.websocket_handler = websocket_handler
        self.logger = logger
        
        # Event tracking
        self.event_history: List[Dict[str, Any]] = []
        self.event_stats: Dict[str, int] = {}
        self.max_history_length = 1000
    
    async def send_event(self, event_type: str, data: Any, channel: str = "events") -> None:
        """Send event through available channels."""
        event = {
            "event_type": event_type,
            "data": data,
            "timestamp": time.time(),
            "channel": channel,
        }
        
        # Add to history
        self.event_history.append(event)
        if len(self.event_history) > self.max_history_length:
            self.event_history = self.event_history[-self.max_history_length:]
        
        # Update stats
        self.event_stats[event_type] = self.event_stats.get(event_type, 0) + 1
        
        # Send through WebSocket if available
        if self.websocket_handler:
            try:
                await self.websocket_handler.send_event(event_type, data, channel)
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to send event via WebSocket: {e}")
        
        # Log event if logger available
        if self.logger:
            self.logger.debug(f"Event sent: {event_type} -> {channel}")
    
    async def send_agent_status_update(self, agent_id: str, status: Dict[str, Any]) -> None:
        """Send agent status update event."""
        await self.send_event(
            "agent_status_update",
            {"agent_id": agent_id, "status": status},
            "agents"
        )
    
    async def send_task_progress_update(self, task_id: str, progress: Dict[str, Any]) -> None:
        """Send task progress update event."""
        await self.send_event(
            "task_progress_update",
            {"task_id": task_id, "progress": progress},
            "tasks"
        )
    
    async def send_orchestration_event(self, orchestration_data: Dict[str, Any]) -> None:
        """Send orchestration event."""
        await self.send_event(
            "orchestration_event",
            orchestration_data,
            "orchestration"
        )
    
    async def send_performance_metric_update(self, metrics: Dict[str, Any]) -> None:
        """Send performance metrics update event."""
        await self.send_event(
            "performance_metric_update",
            metrics,
            "performance"
        )
    
    async def send_error_notification(self, error_data: Dict[str, Any]) -> None:
        """Send error notification event."""
        await self.send_event(
            "error_notification",
            error_data,
            "errors"
        )
    
    async def send_debug_event(self, debug_data: Dict[str, Any]) -> None:
        """Send debug event."""
        await self.send_event(
            "debug_event",
            debug_data,
            "debug"
        )
    
    def get_event_history(self, 
                         event_type: Optional[str] = None,
                         limit: int = 100) -> List[Dict[str, Any]]:
        """Get event history with optional filtering."""
        events = self.event_history.copy()
        
        if event_type:
            events = [e for e in events if e["event_type"] == event_type]
        
        return events[-limit:] if limit > 0 else events
    
    def get_event_stats(self) -> Dict[str, Any]:
        """Get event statistics."""
        return {
            "total_events": len(self.event_history),
            "event_type_counts": self.event_stats.copy(),
            "recent_events": len([e for e in self.event_history if time.time() - e["timestamp"] < 300]),  # Last 5 minutes
        }


class LogHandler:
    """Handles log capture and real-time log streaming."""
    
    def __init__(self,
                 config: DebugConfig,
                 websocket_handler: Optional[WebSocketHandler] = None,
                 logger: Optional[RunLogger] = None):
        
        self.config = config
        self.websocket_handler = websocket_handler
        self.logger = logger
        
        # Log management
        self.captured_logs: List[Dict[str, Any]] = []
        self.log_filters: Dict[str, Any] = {}
        self.real_time_streaming = False
        
        # Log stats
        self.log_stats: Dict[str, int] = {}
    
    def capture_log(self, log_entry: Dict[str, Any]) -> None:
        """Capture and optionally stream a log entry."""
        # Add capture metadata
        enhanced_entry = {
            **log_entry,
            "captured_at": time.time(),
            "handler": "web_log_handler"
        }
        
        # Apply filters
        if self._should_capture_log(enhanced_entry):
            self.captured_logs.append(enhanced_entry)
            
            # Update stats
            level = enhanced_entry.get("level", "unknown").upper()
            self.log_stats[level] = self.log_stats.get(level, 0) + 1
            
            # Trim to max entries
            if len(self.captured_logs) > self.config.max_log_entries:
                self.captured_logs = self.captured_logs[-self.config.max_log_entries:]
            
            # Stream in real-time if enabled
            if self.real_time_streaming and self.websocket_handler:
                asyncio.create_task(self._stream_log_entry(enhanced_entry))
    
    async def _stream_log_entry(self, log_entry: Dict[str, Any]) -> None:
        """Stream log entry via WebSocket."""
        try:
            await self.websocket_handler.send_event(
                "log_entry",
                log_entry,
                "logs"
            )
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to stream log entry: {e}")
    
    def _should_capture_log(self, log_entry: Dict[str, Any]) -> bool:
        """Check if log entry should be captured based on filters."""
        # Check log level
        log_level = log_entry.get("level", "INFO").upper()
        config_level = self.config.log_level.value.upper()
        
        level_priority = {
            "DEBUG": 0,
            "INFO": 1,
            "WARNING": 2,
            "ERROR": 3,
            "CRITICAL": 4,
        }
        
        if level_priority.get(log_level, 1) < level_priority.get(config_level, 1):
            return False
        
        # Apply custom filters
        for filter_key, filter_value in self.log_filters.items():
            if filter_key in log_entry:
                if isinstance(filter_value, list):
                    if log_entry[filter_key] not in filter_value:
                        return False
                elif log_entry[filter_key] != filter_value:
                    return False
        
        return True
    
    def set_real_time_streaming(self, enabled: bool) -> None:
        """Enable or disable real-time log streaming."""
        self.real_time_streaming = enabled
        
        if self.logger:
            self.logger.info(f"Real-time log streaming {'enabled' if enabled else 'disabled'}")
    
    def add_log_filter(self, filter_key: str, filter_value: Any) -> None:
        """Add a log filter."""
        self.log_filters[filter_key] = filter_value
    
    def remove_log_filter(self, filter_key: str) -> None:
        """Remove a log filter."""
        self.log_filters.pop(filter_key, None)
    
    def clear_log_filters(self) -> None:
        """Clear all log filters."""
        self.log_filters.clear()
    
    def get_captured_logs(self,
                         level: Optional[str] = None,
                         limit: int = 100,
                         start_time: Optional[float] = None,
                         end_time: Optional[float] = None) -> List[Dict[str, Any]]:
        """Get captured logs with optional filtering."""
        logs = self.captured_logs.copy()
        
        # Apply level filter
        if level:
            logs = [log for log in logs if log.get("level", "").upper() == level.upper()]
        
        # Apply time filters
        if start_time:
            logs = [log for log in logs if log.get("captured_at", 0) >= start_time]
        
        if end_time:
            logs = [log for log in logs if log.get("captured_at", 0) <= end_time]
        
        # Apply limit
        return logs[-limit:] if limit > 0 else logs
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get log statistics."""
        return {
            "total_captured": len(self.captured_logs),
            "by_level": self.log_stats.copy(),
            "real_time_streaming": self.real_time_streaming,
            "active_filters": len(self.log_filters),
            "filters": self.log_filters.copy(),
        }
    
    def clear_captured_logs(self) -> None:
        """Clear all captured logs."""
        self.captured_logs.clear()
        self.log_stats.clear()
        
        if self.logger:
            self.logger.info("Cleared all captured logs")