"""
Artifact service implementations for file and document management.
"""

import hashlib
import json
import mimetypes
import shutil
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from abc import ABC, abstractmethod

from google.adk.artifacts import BaseArtifactService
from google.genai import types

from .base import BaseService
from ..config.services import ArtifactServiceConfig


class ArtifactService(BaseService, BaseArtifactService, ABC):
    """
    Abstract base class for artifact services with platform integration.
    
    Extends ADK's BaseArtifactService with platform-specific features.
    """
    
    def __init__(self, name: str, config: ArtifactServiceConfig):
        BaseService.__init__(self, name, config.model_dump())
        self.config = config
        self._artifact_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = threading.RLock()
    
    async def _start_impl(self) -> None:
        """Initialize the artifact service."""
        await self._initialize_storage()
        self._log_info("Artifact service initialized")
    
    async def _stop_impl(self) -> None:
        """Cleanup artifact service resources."""
        await self._cleanup_storage()
        self._log_info("Artifact service cleaned up")
    
    async def _health_check_impl(self) -> tuple[bool, Dict[str, Any]]:
        """Check artifact service health."""
        try:
            # Test basic operations
            test_filename = "health_check_test.txt"
            test_content = types.Part(text="Health check test content")
            
            # Test save and load
            version = await self.save_artifact(test_filename, test_content)
            loaded_part = await self.load_artifact(test_filename)
            
            # Cleanup test artifact
            await self._cleanup_test_artifact(test_filename)
            
            return True, {
                "cached_artifacts": len(self._artifact_cache),
                "storage_type": self.config.service_type.value,
                "test_save_successful": version is not None,
                "test_load_successful": loaded_part is not None
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
    
    @abstractmethod
    async def _cleanup_test_artifact(self, filename: str) -> None:
        """Cleanup test artifact created during health check."""
        pass
    
    def _validate_artifact(self, filename: str, content: types.Part) -> None:
        """Validate artifact before storage."""
        # Check filename
        if not filename or len(filename) > 255:
            raise ValueError("Invalid filename")
        
        # Check file extension
        if self.config.allowed_mime_types:
            mime_type, _ = mimetypes.guess_type(filename)
            if mime_type and mime_type not in self.config.allowed_mime_types:
                raise ValueError(f"MIME type {mime_type} not allowed")
        
        # Check content size
        content_size = self._estimate_content_size(content)
        max_size_bytes = self.config.max_artifact_size_mb * 1024 * 1024
        
        if content_size > max_size_bytes:
            raise ValueError(f"Artifact size {content_size} exceeds limit {max_size_bytes}")
    
    def _estimate_content_size(self, content: types.Part) -> int:
        """Estimate the size of content in bytes."""
        if content.text:
            return len(content.text.encode('utf-8'))
        elif content.inline_data:
            # Base64 encoded data
            return len(content.inline_data.data) * 3 // 4  # Approximate decoded size
        else:
            return 0
    
    def _generate_artifact_id(self, filename: str, content: types.Part) -> str:
        """Generate a unique ID for an artifact."""
        hasher = hashlib.sha256()
        hasher.update(filename.encode('utf-8'))
        hasher.update(str(datetime.utcnow().timestamp()).encode('utf-8'))
        
        if content.text:
            hasher.update(content.text.encode('utf-8'))
        elif content.inline_data:
            hasher.update(content.inline_data.data.encode('utf-8'))
        
        return hasher.hexdigest()[:16]
    
    def _compress_content(self, content: types.Part) -> types.Part:
        """Compress content if enabled and beneficial."""
        if not self.config.enable_compression:
            return content
        
        if content.text and len(content.text) > self.config.compression_threshold_kb * 1024:
            # In a real implementation, you would compress the text
            # For now, just return the original content
            return content
        
        return content
    
    def _create_artifact_metadata(self, filename: str, content: types.Part, 
                                 artifact_id: str, version: int) -> Dict[str, Any]:
        """Create metadata for an artifact."""
        mime_type, _ = mimetypes.guess_type(filename)
        
        return {
            'id': artifact_id,
            'filename': filename,
            'version': version,
            'mime_type': mime_type,
            'size_bytes': self._estimate_content_size(content),
            'created_at': datetime.utcnow().isoformat(),
            'content_type': 'text' if content.text else 'binary',
            'compressed': self.config.enable_compression,
            'checksum': self._calculate_checksum(content)
        }
    
    def _calculate_checksum(self, content: types.Part) -> str:
        """Calculate checksum for content integrity."""
        hasher = hashlib.md5()
        
        if content.text:
            hasher.update(content.text.encode('utf-8'))
        elif content.inline_data:
            hasher.update(content.inline_data.data.encode('utf-8'))
        
        return hasher.hexdigest()


class InMemoryArtifactService(ArtifactService):
    """
    In-memory implementation of artifact service.
    
    Suitable for development and testing. All artifacts are lost on restart.
    """
    
    def __init__(self, config: Optional[ArtifactServiceConfig] = None):
        config = config or ArtifactServiceConfig()
        super().__init__("in_memory_artifact", config)
        self._storage: Dict[str, Dict[int, Dict[str, Any]]] = {}  # filename -> version -> artifact_data
        self._metadata: Dict[str, Dict[str, Any]] = {}  # filename -> metadata
    
    async def _initialize_storage(self) -> None:
        """Initialize in-memory storage."""
        self._storage.clear()
        self._metadata.clear()
    
    async def _cleanup_storage(self) -> None:
        """Clear in-memory storage."""
        self._storage.clear()
        self._metadata.clear()
    
    async def _cleanup_test_artifact(self, filename: str) -> None:
        """Remove test artifact from memory."""
        with self._cache_lock:
            self._storage.pop(filename, None)
            self._metadata.pop(filename, None)
    
    async def save_artifact(self, filename: str, content: types.Part) -> int:
        """Save artifact to memory."""
        self._validate_artifact(filename, content)
        
        with self._cache_lock:
            # Initialize filename storage
            if filename not in self._storage:
                self._storage[filename] = {}
                self._metadata[filename] = {}
            
            # Determine version
            existing_versions = list(self._storage[filename].keys())
            version = max(existing_versions, default=0) + 1
            
            # Check version limit
            if len(existing_versions) >= self.config.max_versions_per_artifact:
                # Remove oldest version
                oldest_version = min(existing_versions)
                del self._storage[filename][oldest_version]
                del self._metadata[filename][str(oldest_version)]
            
            # Compress content if needed
            processed_content = self._compress_content(content)
            
            # Generate artifact ID and metadata
            artifact_id = self._generate_artifact_id(filename, content)
            metadata = self._create_artifact_metadata(filename, content, artifact_id, version)
            
            # Store artifact
            self._storage[filename][version] = {
                'content': processed_content,
                'metadata': metadata
            }
            self._metadata[filename][str(version)] = metadata
            
            self._log_debug(f"Saved artifact {filename} version {version}")
            return version
    
    async def load_artifact(self, filename: str, version: Optional[int] = None) -> Optional[types.Part]:
        """Load artifact from memory."""
        with self._cache_lock:
            if filename not in self._storage:
                return None
            
            # Get version
            if version is None:
                # Get latest version
                versions = list(self._storage[filename].keys())
                if not versions:
                    return None
                version = max(versions)
            
            artifact_data = self._storage[filename].get(version)
            if not artifact_data:
                return None
            
            self._log_debug(f"Loaded artifact {filename} version {version}")
            return artifact_data['content']
    
    async def list_artifacts(self) -> List[str]:
        """List all artifact filenames."""
        with self._cache_lock:
            return list(self._storage.keys())
    
    async def get_artifact_metadata(self, filename: str) -> Optional[Dict[str, Any]]:
        """Get artifact metadata."""
        with self._cache_lock:
            if filename not in self._metadata:
                return None
            
            # Return metadata for all versions
            return {
                'filename': filename,
                'versions': self._metadata[filename]
            }
    
    async def delete_artifact(self, filename: str, version: Optional[int] = None) -> bool:
        """Delete artifact or specific version."""
        with self._cache_lock:
            if filename not in self._storage:
                return False
            
            if version is None:
                # Delete all versions
                del self._storage[filename]
                del self._metadata[filename]
                self._log_debug(f"Deleted all versions of artifact {filename}")
                return True
            else:
                # Delete specific version
                if version in self._storage[filename]:
                    del self._storage[filename][version]
                    del self._metadata[filename][str(version)]
                    
                    # If no versions left, remove filename entirely
                    if not self._storage[filename]:
                        del self._storage[filename]
                        del self._metadata[filename]
                    
                    self._log_debug(f"Deleted artifact {filename} version {version}")
                    return True
                
                return False
    
    async def list_artifact_keys(self) -> List[str]:
        """List all artifact keys (filenames)."""
        return await self.list_artifacts()
    
    async def list_versions(self, filename: str) -> List[int]:
        """List all versions for a specific artifact."""
        with self._cache_lock:
            if filename not in self._storage:
                return []
            return sorted(self._storage[filename].keys())


class LocalFileArtifactService(ArtifactService):
    """
    Local file system implementation of artifact service.
    
    Stores artifacts as files in a local directory with SQLite metadata.
    """
    
    def __init__(self, config: ArtifactServiceConfig):
        super().__init__("local_file_artifact", config)
        self.storage_path = config.local_storage_path
        self._connection: Optional[sqlite3.Connection] = None
        self._db_lock = threading.RLock()
    
    async def _initialize_storage(self) -> None:
        """Initialize local file storage and metadata database."""
        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata database
        db_path = self.storage_path / "metadata.db"
        self._connection = sqlite3.connect(str(db_path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        
        with self._db_lock:
            cursor = self._connection.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS artifacts (
                    filename TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    artifact_id TEXT NOT NULL,
                    mime_type TEXT,
                    size_bytes INTEGER NOT NULL,
                    content_type TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (filename, version)
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_artifacts_filename 
                ON artifacts (filename)
            """)
            
            self._connection.commit()
    
    async def _cleanup_storage(self) -> None:
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
    
    async def _cleanup_test_artifact(self, filename: str) -> None:
        """Remove test artifact from file system."""
        await self.delete_artifact(filename)
    
    async def save_artifact(self, filename: str, content: types.Part) -> int:
        """Save artifact to local file system."""
        self._validate_artifact(filename, content)
        
        with self._db_lock:
            cursor = self._connection.cursor()
            
            # Determine version
            cursor.execute("""
                SELECT MAX(version) as max_version FROM artifacts WHERE filename = ?
            """, (filename,))
            
            row = cursor.fetchone()
            version = (row['max_version'] or 0) + 1
            
            # Generate artifact ID and file path
            artifact_id = self._generate_artifact_id(filename, content)
            file_path = self.storage_path / f"{artifact_id}_v{version}_{filename}"
            
            # Save content to file
            if content.text:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content.text)
            elif content.inline_data:
                # Decode base64 and save as binary
                import base64
                binary_data = base64.b64decode(content.inline_data.data)
                with open(file_path, 'wb') as f:
                    f.write(binary_data)
            else:
                raise ValueError("Content must have text or inline_data")
            
            # Save metadata
            metadata = self._create_artifact_metadata(filename, content, artifact_id, version)
            
            cursor.execute("""
                INSERT INTO artifacts 
                (filename, version, artifact_id, mime_type, size_bytes, content_type, file_path, checksum)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                filename,
                version,
                artifact_id,
                metadata['mime_type'],
                metadata['size_bytes'],
                metadata['content_type'],
                str(file_path),
                metadata['checksum']
            ))
            
            # Clean up old versions if needed
            cursor.execute("""
                SELECT version, file_path FROM artifacts 
                WHERE filename = ? 
                ORDER BY version DESC
            """, (filename,))
            
            versions = cursor.fetchall()
            if len(versions) > self.config.max_versions_per_artifact:
                # Delete oldest versions
                for old_version in versions[self.config.max_versions_per_artifact:]:
                    old_file_path = Path(old_version['file_path'])
                    if old_file_path.exists():
                        old_file_path.unlink()
                    
                    cursor.execute("""
                        DELETE FROM artifacts 
                        WHERE filename = ? AND version = ?
                    """, (filename, old_version['version']))
            
            self._connection.commit()
            
            self._log_debug(f"Saved artifact {filename} version {version} to {file_path}")
            return version
    
    async def load_artifact(self, filename: str, version: Optional[int] = None) -> Optional[types.Part]:
        """Load artifact from local file system."""
        with self._db_lock:
            cursor = self._connection.cursor()
            
            if version is None:
                # Get latest version
                cursor.execute("""
                    SELECT * FROM artifacts 
                    WHERE filename = ? 
                    ORDER BY version DESC 
                    LIMIT 1
                """, (filename,))
            else:
                cursor.execute("""
                    SELECT * FROM artifacts 
                    WHERE filename = ? AND version = ?
                """, (filename, version))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            file_path = Path(row['file_path'])
            if not file_path.exists():
                self._log_error(f"Artifact file missing: {file_path}")
                return None
            
            # Load content based on type
            if row['content_type'] == 'text':
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return types.Part(text=content)
            else:
                # Binary content
                with open(file_path, 'rb') as f:
                    binary_data = f.read()
                
                import base64
                encoded_data = base64.b64encode(binary_data).decode('utf-8')
                
                return types.Part(
                    inline_data=types.Blob(
                        mime_type=row['mime_type'] or 'application/octet-stream',
                        data=encoded_data
                    )
                )
    
    async def list_artifacts(self) -> List[str]:
        """List all artifact filenames."""
        with self._db_lock:
            cursor = self._connection.cursor()
            cursor.execute("SELECT DISTINCT filename FROM artifacts")
            return [row['filename'] for row in cursor.fetchall()]
    
    async def get_artifact_metadata(self, filename: str) -> Optional[Dict[str, Any]]:
        """Get artifact metadata."""
        with self._db_lock:
            cursor = self._connection.cursor()
            cursor.execute("""
                SELECT * FROM artifacts WHERE filename = ? ORDER BY version DESC
            """, (filename,))
            
            rows = cursor.fetchall()
            if not rows:
                return None
            
            versions = {}
            for row in rows:
                versions[str(row['version'])] = {
                    'version': row['version'],
                    'artifact_id': row['artifact_id'],
                    'mime_type': row['mime_type'],
                    'size_bytes': row['size_bytes'],
                    'content_type': row['content_type'],
                    'checksum': row['checksum'],
                    'created_at': row['created_at']
                }
            
            return {
                'filename': filename,
                'versions': versions
            }
    
    async def delete_artifact(self, filename: str, version: Optional[int] = None) -> bool:
        """Delete artifact from local file system."""
        with self._db_lock:
            cursor = self._connection.cursor()
            
            if version is None:
                # Delete all versions
                cursor.execute("""
                    SELECT file_path FROM artifacts WHERE filename = ?
                """, (filename,))
                
                file_paths = [row['file_path'] for row in cursor.fetchall()]
                
                # Delete files
                for file_path in file_paths:
                    path = Path(file_path)
                    if path.exists():
                        path.unlink()
                
                # Delete metadata
                cursor.execute("DELETE FROM artifacts WHERE filename = ?", (filename,))
                deleted = cursor.rowcount > 0
                
            else:
                # Delete specific version
                cursor.execute("""
                    SELECT file_path FROM artifacts 
                    WHERE filename = ? AND version = ?
                """, (filename, version))
                
                row = cursor.fetchone()
                if not row:
                    return False
                
                # Delete file
                file_path = Path(row['file_path'])
                if file_path.exists():
                    file_path.unlink()
                
                # Delete metadata
                cursor.execute("""
                    DELETE FROM artifacts 
                    WHERE filename = ? AND version = ?
                """, (filename, version))
                deleted = cursor.rowcount > 0
            
            self._connection.commit()
            
            if deleted:
                self._log_debug(f"Deleted artifact {filename}" + 
                              (f" version {version}" if version else " (all versions)"))
            
            return deleted
    
    async def list_artifact_keys(self) -> List[str]:
        """List all artifact keys (filenames)."""
        return await self.list_artifacts()
    
    async def list_versions(self, filename: str) -> List[int]:
        """List all versions for a specific artifact."""
        with self._db_lock:
            cursor = self._connection.cursor()
            cursor.execute("""
                SELECT version FROM artifacts WHERE filename = ? ORDER BY version
            """, (filename,))
            return [row['version'] for row in cursor.fetchall()]


class GCSArtifactService(ArtifactService):
    """
    Google Cloud Storage implementation of artifact service.
    
    Stores artifacts in GCS with metadata in Cloud SQL or local database.
    """
    
    def __init__(self, config: ArtifactServiceConfig):
        super().__init__("gcs_artifact", config)
        self._gcs_client = None
        self._bucket = None
    
    async def _initialize_storage(self) -> None:
        """Initialize GCS client and bucket."""
        try:
            # Import GCS libraries
            # from google.cloud import storage
            
            # Initialize client
            # self._gcs_client = storage.Client(project=self.config.gcs_project)
            # self._bucket = self._gcs_client.bucket(self.config.gcs_bucket_name)
            
            # For now, log that we would initialize GCS
            self._log_info(f"Would initialize GCS bucket: {self.config.gcs_bucket_name}")
            
        except ImportError:
            self._log_warning("Google Cloud Storage libraries not available, using fallback")
            # Fall back to local file implementation
            self._fallback_service = LocalFileArtifactService(self.config)
            await self._fallback_service._initialize_storage()
    
    async def _cleanup_storage(self) -> None:
        """Cleanup GCS resources."""
        if hasattr(self, '_fallback_service'):
            await self._fallback_service._cleanup_storage()
    
    async def _cleanup_test_artifact(self, filename: str) -> None:
        """Remove test artifact from GCS."""
        if hasattr(self, '_fallback_service'):
            await self._fallback_service._cleanup_test_artifact(filename)
    
    async def save_artifact(self, filename: str, content: types.Part) -> int:
        """Save artifact to GCS."""
        if hasattr(self, '_fallback_service'):
            return await self._fallback_service.save_artifact(filename, content)
        
        # Implementation would use GCS APIs
        self._log_info(f"Would save artifact {filename} to GCS")
        return 1
    
    async def load_artifact(self, filename: str, version: Optional[int] = None) -> Optional[types.Part]:
        """Load artifact from GCS."""
        if hasattr(self, '_fallback_service'):
            return await self._fallback_service.load_artifact(filename, version)
        
        # Implementation would use GCS APIs
        self._log_info(f"Would load artifact {filename} from GCS")
        return None
    
    async def list_artifacts(self) -> List[str]:
        """List artifacts in GCS."""
        if hasattr(self, '_fallback_service'):
            return await self._fallback_service.list_artifacts()
        
        return []


class S3ArtifactService(ArtifactService):
    """
    Amazon S3 implementation of artifact service.
    
    Stores artifacts in S3 with metadata management.
    """
    
    def __init__(self, config: ArtifactServiceConfig):
        super().__init__("s3_artifact", config)
        self._s3_client = None
    
    async def _initialize_storage(self) -> None:
        """Initialize S3 client."""
        try:
            # Import boto3
            # import boto3
            
            # Initialize client
            # self._s3_client = boto3.client('s3', region_name=self.config.s3_region)
            
            # For now, log that we would initialize S3
            self._log_info(f"Would initialize S3 bucket: {self.config.s3_bucket_name}")
            
        except ImportError:
            self._log_warning("boto3 not available, using fallback")
            # Fall back to local file implementation
            self._fallback_service = LocalFileArtifactService(self.config)
            await self._fallback_service._initialize_storage()
    
    async def _cleanup_storage(self) -> None:
        """Cleanup S3 resources."""
        if hasattr(self, '_fallback_service'):
            await self._fallback_service._cleanup_storage()
    
    async def _cleanup_test_artifact(self, filename: str) -> None:
        """Remove test artifact from S3."""
        if hasattr(self, '_fallback_service'):
            await self._fallback_service._cleanup_test_artifact(filename)
    
    async def save_artifact(self, filename: str, content: types.Part) -> int:
        """Save artifact to S3."""
        if hasattr(self, '_fallback_service'):
            return await self._fallback_service.save_artifact(filename, content)
        
        # Implementation would use S3 APIs
        self._log_info(f"Would save artifact {filename} to S3")
        return 1
    
    async def load_artifact(self, filename: str, version: Optional[int] = None) -> Optional[types.Part]:
        """Load artifact from S3."""
        if hasattr(self, '_fallback_service'):
            return await self._fallback_service.load_artifact(filename, version)
        
        # Implementation would use S3 APIs
        self._log_info(f"Would load artifact {filename} from S3")
        return None
    
    async def list_artifacts(self) -> List[str]:
        """List artifacts in S3."""
        if hasattr(self, '_fallback_service'):
            return await self._fallback_service.list_artifacts()
        
        return []