"""
Code Execution Built-in Tool Integration

Provides comprehensive integration with ADK's built-in code execution capabilities,
including safety measures, result processing, and artifact management.
"""

import time
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from google.adk.tools.tool_context import ToolContext

from .base import (
    BaseTool, 
    BuiltInToolMixin, 
    ToolResult, 
    ToolType, 
    ToolExecutionStatus,
    ToolAuthConfig,
)
from ..platform_logging import RunLogger
from ..services import SessionService, MemoryService, ArtifactService


class CodeLanguage(str, Enum):
    """Supported code execution languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    BASH = "bash"
    R = "r"
    SQL = "sql"


class ExecutionSafety(str, Enum):
    """Code execution safety levels."""
    STRICT = "strict"      # Maximum safety, limited capabilities
    MODERATE = "moderate"  # Balanced safety and functionality
    PERMISSIVE = "permissive"  # Minimal restrictions, maximum functionality


@dataclass
class CodeExecutionConfig:
    """Configuration for code execution."""
    language: CodeLanguage
    safety_level: ExecutionSafety = ExecutionSafety.MODERATE
    timeout_seconds: int = 30
    max_output_length: int = 10000
    allow_network: bool = False
    allow_file_system: bool = True
    allowed_imports: Optional[List[str]] = None
    blocked_imports: Optional[List[str]] = None
    working_directory: Optional[str] = None
    environment_variables: Optional[Dict[str, str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for execution engine."""
        return {
            "language": self.language.value,
            "safety_level": self.safety_level.value,
            "timeout": self.timeout_seconds,
            "max_output": self.max_output_length,
            "network_access": self.allow_network,
            "filesystem_access": self.allow_file_system,
            "allowed_imports": self.allowed_imports or [],
            "blocked_imports": self.blocked_imports or [],
            "working_dir": self.working_directory,
            "env_vars": self.environment_variables or {},
        }


@dataclass
class CodeExecutionResult:
    """Result of code execution with comprehensive information."""
    code_hash: str
    language: CodeLanguage
    execution_status: str  # "success", "error", "timeout", "blocked"
    stdout: str = ""
    stderr: str = ""
    return_value: Any = None
    execution_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    files_created: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    imports_used: List[str] = field(default_factory=list)
    safety_violations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        """Check if execution was successful."""
        return self.execution_status == "success"
    
    @property
    def has_output(self) -> bool:
        """Check if execution produced output."""
        return bool(self.stdout.strip() or self.stderr.strip() or self.return_value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "code_hash": self.code_hash,
            "language": self.language.value,
            "execution_status": self.execution_status,
            "success": self.success,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "return_value": str(self.return_value) if self.return_value is not None else None,
            "execution_time_ms": self.execution_time_ms,
            "memory_usage_mb": self.memory_usage_mb,
            "files_created": self.files_created,
            "files_modified": self.files_modified,
            "imports_used": self.imports_used,
            "safety_violations": self.safety_violations,
            "has_output": self.has_output,
            "metadata": self.metadata,
        }


class CodeExecutionTool(BaseTool, BuiltInToolMixin):
    """
    Code Execution built-in tool integration.
    
    Provides comprehensive code execution capabilities with safety measures,
    result processing, and integration with platform artifact management.
    """
    
    def __init__(self,
                 default_config: Optional[CodeExecutionConfig] = None,
                 logger: Optional[RunLogger] = None,
                 session_service: Optional[SessionService] = None,
                 memory_service: Optional[MemoryService] = None,
                 artifact_service: Optional[ArtifactService] = None,
                 auth_config: Optional[ToolAuthConfig] = None):
        
        super().__init__(
            tool_type=ToolType.CODE_EXECUTION,
            tool_name="code_execution",
            logger=logger,
            session_service=session_service,
            memory_service=memory_service,
            artifact_service=artifact_service,
            auth_config=auth_config,
        )
        
        # Default execution configuration
        self.default_config = default_config or CodeExecutionConfig(
            language=CodeLanguage.PYTHON,
            safety_level=ExecutionSafety.MODERATE,
            timeout_seconds=30,
            blocked_imports=["os", "subprocess", "sys", "shutil", "glob"]
        )
        
        # Safety configurations for different levels
        self.safety_configs = {
            ExecutionSafety.STRICT: {
                "blocked_imports": [
                    "os", "sys", "subprocess", "shutil", "glob", "pathlib",
                    "socket", "urllib", "requests", "http", "ftplib",
                    "smtplib", "telnetlib", "webbrowser", "tempfile"
                ],
                "allowed_builtins": ["print", "len", "range", "enumerate", "zip"],
                "max_loops": 1000,
                "max_recursion": 100,
            },
            ExecutionSafety.MODERATE: {
                "blocked_imports": [
                    "os", "sys", "subprocess", "shutil", "socket",
                    "urllib", "ftplib", "smtplib", "telnetlib"
                ],
                "max_loops": 10000,
                "max_recursion": 500,
            },
            ExecutionSafety.PERMISSIVE: {
                "blocked_imports": ["subprocess"],
                "max_loops": 100000,
                "max_recursion": 1000,
            }
        }
    
    def get_tool_config(self) -> Dict[str, Any]:
        """Get code execution tool configuration."""
        return {
            "tool_type": "code_execution",
            "built_in": True,
            "requires_auth": self.auth_config is not None,
            "supported_languages": [lang.value for lang in CodeLanguage],
            "safety_levels": [level.value for level in ExecutionSafety],
            "default_config": self.default_config.to_dict(),
            "safety_configs": {
                level.value: config 
                for level, config in self.safety_configs.items()
            },
        }
    
    def execute_tool(self, context: ToolContext, **kwargs) -> ToolResult:
        """Execute code with comprehensive safety and monitoring."""
        start_time = time.time()
        
        # Parse execution parameters
        code = kwargs.get("code", "")
        config = self._parse_execution_config(**kwargs)
        
        if not code.strip():
            return ToolResult(
                tool_type=self.tool_type,
                status=ToolExecutionStatus.FAILED,
                error="Empty code provided for execution",
            )
        
        # Generate code hash for tracking
        code_hash = hashlib.sha256(code.encode()).hexdigest()[:16]
        
        try:
            # Get enhanced context for full capabilities
            enhanced_context = self.get_enhanced_context(context)
            enhanced_context.start_execution()
            
            # Pre-execution safety checks
            safety_check = self._perform_safety_checks(code, config)
            if not safety_check["safe"]:
                return ToolResult(
                    tool_type=self.tool_type,
                    status=ToolExecutionStatus.FAILED,
                    error=f"Safety check failed: {', '.join(safety_check['violations'])}",
                    metadata={"safety_violations": safety_check["violations"]},
                )
            
            # Get ADK built-in code executor
            builtin_executor = self.get_builtin_tool_instance(context, "code_executor")
            if not builtin_executor:
                # Fallback to context method
                if not hasattr(context, 'execute_code'):
                    return ToolResult(
                        tool_type=self.tool_type,
                        status=ToolExecutionStatus.FAILED,
                        error="Code execution capability not available",
                    )
                builtin_executor = context
            
            # Log execution start
            self.log_tool_usage("execute_code", {
                "language": config.language.value,
                "safety_level": config.safety_level.value,
                "code_length": len(code),
                "code_hash": code_hash,
            })
            
            # Execute code
            execution_result = self._execute_code_safely(
                builtin_executor, 
                code, 
                config, 
                code_hash
            )
            
            # Store artifacts if any files were created
            if execution_result.files_created and self.artifact_service:
                self._store_execution_artifacts(
                    execution_result, 
                    enhanced_context
                )
            
            # Store execution in memory for future reference
            if self.memory_service and execution_result.success:
                self._store_execution_in_memory(
                    code, 
                    execution_result, 
                    enhanced_context
                )
            
            total_execution_time = (time.time() - start_time) * 1000
            
            result = ToolResult(
                tool_type=self.tool_type,
                status=ToolExecutionStatus.COMPLETED,
                data={
                    "execution_result": execution_result.to_dict(),
                    "code": code,
                    "config": config.to_dict(),
                },
                metadata={
                    "code_hash": code_hash,
                    "language": config.language.value,
                    "safety_level": config.safety_level.value,
                    "total_execution_time_ms": total_execution_time,
                    "code_execution_time_ms": execution_result.execution_time_ms,
                    "has_output": execution_result.has_output,
                    "files_created_count": len(execution_result.files_created),
                },
                execution_time_ms=total_execution_time,
            )
            
            # Complete execution tracking
            enhanced_context.complete_execution(result)
            
            return result
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            error_msg = f"Code execution failed: {str(e)}"
            
            if self.logger:
                self.logger.error(error_msg, 
                                code_hash=code_hash,
                                language=config.language.value,
                                execution_time_ms=execution_time)
            
            return ToolResult(
                tool_type=self.tool_type,
                status=ToolExecutionStatus.FAILED,
                error=error_msg,
                execution_time_ms=execution_time,
                metadata={"code_hash": code_hash},
            )
    
    def execute_python(self, 
                      code: str,
                      context: ToolContext,
                      safety_level: ExecutionSafety = ExecutionSafety.MODERATE,
                      **execution_options) -> CodeExecutionResult:
        """
        Convenient Python code execution method.
        
        Args:
            code: Python code to execute
            context: Tool execution context
            safety_level: Safety level for execution
            **execution_options: Additional execution options
            
        Returns:
            Code execution result with comprehensive information
        """
        result = self.execute_with_context(
            context,
            code=code,
            language=CodeLanguage.PYTHON,
            safety_level=safety_level,
            **execution_options
        )
        
        if result.success and "execution_result" in result.tool_result.data:
            result_data = result.tool_result.data["execution_result"]
            return CodeExecutionResult(**result_data)
        
        # Return failed result
        return CodeExecutionResult(
            code_hash=hashlib.sha256(code.encode()).hexdigest()[:16],
            language=CodeLanguage.PYTHON,
            execution_status="error",
            stderr=result.tool_result.error or "Execution failed",
        )
    
    def execute_with_artifact_management(self,
                                       code: str,
                                       context: ToolContext,
                                       artifact_patterns: Optional[List[str]] = None,
                                       cleanup_after: bool = True,
                                       **execution_options) -> Dict[str, Any]:
        """
        Execute code with comprehensive artifact management.
        
        Args:
            code: Code to execute
            context: Tool execution context
            artifact_patterns: File patterns to track as artifacts
            cleanup_after: Whether to clean up temporary files
            **execution_options: Additional execution options
            
        Returns:
            Dictionary with execution results and artifact information
        """
        # Set up artifact tracking
        required_artifacts = artifact_patterns or []
        
        result = self.execute_with_context(
            context,
            required_artifacts=required_artifacts,
            code=code,
            **execution_options
        )
        
        return {
            "execution_result": result.tool_result.data if result.success else None,
            "artifact_info": {
                "files_created": result.tool_result.data.get("execution_result", {}).get("files_created", []),
                "files_modified": result.tool_result.data.get("execution_result", {}).get("files_modified", []),
                "artifact_patterns": artifact_patterns,
            },
            "cleanup_performed": cleanup_after,
            "overall_success": result.success,
            "execution_info": result.to_dict(),
        }
    
    def _parse_execution_config(self, **kwargs) -> CodeExecutionConfig:
        """Parse and validate execution configuration."""
        language = CodeLanguage(kwargs.get("language", self.default_config.language.value))
        safety_level = ExecutionSafety(kwargs.get("safety_level", self.default_config.safety_level.value))
        
        return CodeExecutionConfig(
            language=language,
            safety_level=safety_level,
            timeout_seconds=kwargs.get("timeout_seconds", self.default_config.timeout_seconds),
            max_output_length=kwargs.get("max_output_length", self.default_config.max_output_length),
            allow_network=kwargs.get("allow_network", self.default_config.allow_network),
            allow_file_system=kwargs.get("allow_file_system", self.default_config.allow_file_system),
            allowed_imports=kwargs.get("allowed_imports", self.default_config.allowed_imports),
            blocked_imports=kwargs.get("blocked_imports", self.default_config.blocked_imports),
            working_directory=kwargs.get("working_directory", self.default_config.working_directory),
            environment_variables=kwargs.get("environment_variables", self.default_config.environment_variables),
        )
    
    def _perform_safety_checks(self, code: str, config: CodeExecutionConfig) -> Dict[str, Any]:
        """Perform comprehensive safety checks on code."""
        violations = []
        safety_config = self.safety_configs.get(config.safety_level, {})
        
        # Check for blocked imports
        blocked_imports = config.blocked_imports or safety_config.get("blocked_imports", [])
        for blocked_import in blocked_imports:
            if f"import {blocked_import}" in code or f"from {blocked_import}" in code:
                violations.append(f"Blocked import: {blocked_import}")
        
        # Check for dangerous patterns
        dangerous_patterns = [
            "exec(", "eval(", "compile(", "open(", "__import__(",
            "getattr(", "setattr(", "delattr(", "globals(", "locals(",
        ]
        
        for pattern in dangerous_patterns:
            if pattern in code:
                violations.append(f"Dangerous pattern: {pattern}")
        
        # Check code length
        if len(code) > 50000:  # 50KB limit
            violations.append("Code too long (>50KB)")
        
        # Language-specific checks
        if config.language == CodeLanguage.PYTHON:
            violations.extend(self._check_python_safety(code, safety_config))
        elif config.language == CodeLanguage.BASH:
            violations.extend(self._check_bash_safety(code))
        
        return {
            "safe": len(violations) == 0,
            "violations": violations,
            "safety_level": config.safety_level.value,
        }
    
    def _check_python_safety(self, code: str, safety_config: Dict[str, Any]) -> List[str]:
        """Python-specific safety checks."""
        violations = []
        
        # Check for infinite loops (basic detection)
        lines = code.split('\n')
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('while True:') or stripped.startswith('while 1:'):
                violations.append(f"Potential infinite loop at line {i+1}")
        
        # Check for file operations
        file_operations = ['open(', 'file(', 'os.remove', 'os.unlink', 'shutil.']
        for op in file_operations:
            if op in code:
                violations.append(f"File operation detected: {op}")
        
        return violations
    
    def _check_bash_safety(self, code: str) -> List[str]:
        """Bash-specific safety checks."""
        violations = []
        
        dangerous_commands = [
            'rm -rf', 'dd if=', 'mkfs', 'fdisk', 'parted',
            'curl', 'wget', 'nc ', 'netcat', 'ssh', 'scp',
            'sudo', 'su ', 'chmod 777', 'chown'
        ]
        
        for cmd in dangerous_commands:
            if cmd in code:
                violations.append(f"Dangerous bash command: {cmd}")
        
        return violations
    
    def _execute_code_safely(self, 
                           executor: Any, 
                           code: str, 
                           config: CodeExecutionConfig,
                           code_hash: str) -> CodeExecutionResult:
        """Execute code using ADK built-in executor with safety measures."""
        execution_start = time.time()
        
        try:
            # Prepare execution parameters
            exec_params = config.to_dict()
            exec_params["code"] = code
            
            # Execute using ADK built-in tool
            if hasattr(executor, 'execute_code'):
                raw_result = executor.execute_code(**exec_params)
            elif hasattr(executor, 'execute'):
                raw_result = executor.execute(code, **exec_params)
            else:
                # Fallback execution method
                raw_result = {"error": "No execution method available"}
            
            execution_time = (time.time() - execution_start) * 1000
            
            # Process result based on ADK format
            if isinstance(raw_result, dict):
                return CodeExecutionResult(
                    code_hash=code_hash,
                    language=config.language,
                    execution_status="success" if not raw_result.get("error") else "error",
                    stdout=raw_result.get("stdout", ""),
                    stderr=raw_result.get("stderr", raw_result.get("error", "")),
                    return_value=raw_result.get("result"),
                    execution_time_ms=execution_time,
                    memory_usage_mb=raw_result.get("memory_usage", 0.0),
                    files_created=raw_result.get("files_created", []),
                    files_modified=raw_result.get("files_modified", []),
                    imports_used=raw_result.get("imports_used", []),
                    metadata=raw_result.get("metadata", {}),
                )
            else:
                # Handle non-dict results
                return CodeExecutionResult(
                    code_hash=code_hash,
                    language=config.language,
                    execution_status="success",
                    stdout=str(raw_result) if raw_result is not None else "",
                    execution_time_ms=execution_time,
                )
                
        except Exception as e:
            execution_time = (time.time() - execution_start) * 1000
            
            return CodeExecutionResult(
                code_hash=code_hash,
                language=config.language,
                execution_status="error",
                stderr=str(e),
                execution_time_ms=execution_time,
            )
    
    def _store_execution_artifacts(self, 
                                 execution_result: CodeExecutionResult,
                                 context) -> None:
        """Store execution artifacts using artifact service."""
        try:
            for file_path in execution_result.files_created:
                if hasattr(context, 'save_artifact'):
                    # Read file content and save as artifact
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()
                        
                        artifact_name = f"execution_{execution_result.code_hash}_{file_path.split('/')[-1]}"
                        context.save_artifact(artifact_name, content)
                        
                        if self.logger:
                            self.logger.debug(
                                f"Stored execution artifact: {artifact_name}",
                                file_path=file_path,
                                code_hash=execution_result.code_hash,
                            )
                            
                    except Exception as e:
                        if self.logger:
                            self.logger.warning(f"Failed to store artifact {file_path}: {e}")
                        
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error storing execution artifacts: {e}")
    
    def _store_execution_in_memory(self,
                                 code: str,
                                 execution_result: CodeExecutionResult,
                                 context) -> None:
        """Store successful execution in memory for future reference."""
        try:
            memory_text = f"""
Code Execution (#{execution_result.code_hash}):
Language: {execution_result.language.value}
Status: {execution_result.execution_status}

Code:
{code}

Output:
{execution_result.stdout}

Execution Time: {execution_result.execution_time_ms:.2f}ms
Files Created: {len(execution_result.files_created)}
""".strip()
            
            if hasattr(context, 'store_memory'):
                context.store_memory(
                    text=memory_text,
                    metadata={
                        "type": "code_execution",
                        "language": execution_result.language.value,
                        "code_hash": execution_result.code_hash,
                        "execution_status": execution_result.execution_status,
                        "execution_time_ms": execution_result.execution_time_ms,
                        "has_output": execution_result.has_output,
                    }
                )
                
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error storing execution in memory: {e}")