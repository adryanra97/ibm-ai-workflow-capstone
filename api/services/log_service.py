"""
Logging service for API operations and model performance

Author: Adryan R A
"""

import logging
import json
import sqlite3
import asyncio
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd

from utils.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class LogLevel(str, Enum):
    """Log levels for API operations"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogCategory(str, Enum):
    """Log categories for different types of operations"""
    API_REQUEST = "api_request"
    MODEL_TRAINING = "model_training"
    MODEL_PREDICTION = "model_prediction"
    MODEL_VALIDATION = "model_validation"
    DATA_ACCESS = "data_access"
    SYSTEM_EVENT = "system_event"
    ERROR_EVENT = "error_event"
    PERFORMANCE = "performance"
    SECURITY = "security"


@dataclass
class LogEntry:
    """Structured log entry"""
    timestamp: datetime
    level: LogLevel
    category: LogCategory
    message: str
    details: Dict[str, Any]
    country: Optional[str] = None
    model_type: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    duration_seconds: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['details'] = json.dumps(self.details)
        return data


class LogService:
    """
    Comprehensive logging service for API operations
    """
    
    def __init__(self):
        """Initialize logging service"""
        self.db_path = settings.log_database_path
        self._init_database()
        
        # Performance tracking
        self.performance_buffer = []
        self.buffer_size = 1000
        
        # Log retention settings
        self.retention_days = getattr(settings, 'log_retention_days', 90)
        self.cleanup_interval_hours = 24
        
        # Start background cleanup task
        asyncio.create_task(self._periodic_cleanup())
    
    def _init_database(self):
        """Initialize the logging database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create logs table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS api_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        level TEXT NOT NULL,
                        category TEXT NOT NULL,
                        message TEXT NOT NULL,
                        details TEXT,
                        country TEXT,
                        model_type TEXT,
                        user_id TEXT,
                        request_id TEXT,
                        duration_seconds REAL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes for better query performance
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_logs_timestamp 
                    ON api_logs(timestamp)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_logs_category 
                    ON api_logs(category)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_logs_country_model 
                    ON api_logs(country, model_type)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_logs_level 
                    ON api_logs(level)
                """)
                
                # Create performance metrics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        endpoint TEXT NOT NULL,
                        method TEXT NOT NULL,
                        response_time_ms REAL NOT NULL,
                        status_code INTEGER NOT NULL,
                        country TEXT,
                        model_type TEXT,
                        request_size_bytes INTEGER,
                        response_size_bytes INTEGER,
                        memory_usage_mb REAL,
                        cpu_usage_percent REAL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_perf_timestamp 
                    ON performance_metrics(timestamp)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_perf_endpoint 
                    ON performance_metrics(endpoint)
                """)
                
                conn.commit()
                logger.info("Logging database initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize logging database: {e}")
            raise
    
    async def log_api_request(self,
                            endpoint: str,
                            method: str,
                            status_code: int,
                            response_time_ms: float,
                            request_data: Optional[Dict] = None,
                            response_data: Optional[Dict] = None,
                            user_id: Optional[str] = None,
                            request_id: Optional[str] = None,
                            error: Optional[str] = None) -> None:
        """
        Log API request details
        
        Args:
            endpoint: API endpoint
            method: HTTP method
            status_code: Response status code
            response_time_ms: Response time in milliseconds
            request_data: Request payload (sanitized)
            response_data: Response data (sanitized)
            user_id: User identifier
            request_id: Request identifier
            error: Error message if any
        """
        try:
            # Determine log level based on status code
            if status_code >= 500:
                level = LogLevel.ERROR
            elif status_code >= 400:
                level = LogLevel.WARNING
            else:
                level = LogLevel.INFO
            
            # Prepare details
            details = {
                'endpoint': endpoint,
                'method': method,
                'status_code': status_code,
                'response_time_ms': response_time_ms
            }
            
            if request_data:
                # Sanitize sensitive data
                sanitized_request = self._sanitize_data(request_data)
                details['request_data'] = sanitized_request
            
            if response_data:
                # Limit response data size in logs
                sanitized_response = self._sanitize_data(response_data)
                details['response_data'] = sanitized_response
            
            if error:
                details['error'] = error
            
            # Extract country and model type from request data if available
            country = None
            model_type = None
            if request_data:
                country = request_data.get('country')
                model_type = request_data.get('model_type')
            
            # Create log entry
            log_entry = LogEntry(
                timestamp=datetime.now(),
                level=level,
                category=LogCategory.API_REQUEST,
                message=f"{method} {endpoint} -> {status_code}",
                details=details,
                country=country,
                model_type=model_type,
                user_id=user_id,
                request_id=request_id,
                duration_seconds=response_time_ms / 1000.0
            )
            
            # Store log entry
            await self._store_log_entry(log_entry)
            
            # Store performance metrics
            await self._store_performance_metric(
                endpoint=endpoint,
                method=method,
                response_time_ms=response_time_ms,
                status_code=status_code,
                country=country,
                model_type=model_type
            )
            
        except Exception as e:
            logger.error(f"Failed to log API request: {e}")
    
    async def log_model_operation(self,
                                operation_type: str,
                                country: str,
                                model_type: str,
                                success: bool,
                                duration_seconds: float,
                                details: Dict[str, Any],
                                user_id: Optional[str] = None,
                                request_id: Optional[str] = None) -> None:
        """
        Log model operations (training, prediction, validation)
        
        Args:
            operation_type: Type of operation ('training', 'prediction', 'validation')
            country: Country name
            model_type: Model type
            success: Whether operation was successful
            duration_seconds: Operation duration
            details: Operation details
            user_id: User identifier
            request_id: Request identifier
        """
        try:
            # Map operation type to category
            category_map = {
                'training': LogCategory.MODEL_TRAINING,
                'prediction': LogCategory.MODEL_PREDICTION,
                'validation': LogCategory.MODEL_VALIDATION
            }
            
            category = category_map.get(operation_type, LogCategory.SYSTEM_EVENT)
            level = LogLevel.INFO if success else LogLevel.ERROR
            
            message = f"Model {operation_type} {'completed' if success else 'failed'} for {model_type} model in {country}"
            
            log_entry = LogEntry(
                timestamp=datetime.now(),
                level=level,
                category=category,
                message=message,
                details=details,
                country=country,
                model_type=model_type,
                user_id=user_id,
                request_id=request_id,
                duration_seconds=duration_seconds
            )
            
            await self._store_log_entry(log_entry)
            
        except Exception as e:
            logger.error(f"Failed to log model operation: {e}")
    
    async def log_error(self,
                       error_message: str,
                       error_details: Dict[str, Any],
                       level: LogLevel = LogLevel.ERROR,
                       category: LogCategory = LogCategory.ERROR_EVENT,
                       country: Optional[str] = None,
                       model_type: Optional[str] = None,
                       user_id: Optional[str] = None,
                       request_id: Optional[str] = None) -> None:
        """
        Log error events
        
        Args:
            error_message: Error message
            error_details: Error details
            level: Log level
            category: Log category
            country: Country name
            model_type: Model type
            user_id: User identifier
            request_id: Request identifier
        """
        try:
            log_entry = LogEntry(
                timestamp=datetime.now(),
                level=level,
                category=category,
                message=error_message,
                details=error_details,
                country=country,
                model_type=model_type,
                user_id=user_id,
                request_id=request_id
            )
            
            await self._store_log_entry(log_entry)
            
        except Exception as e:
            logger.error(f"Failed to log error: {e}")
    
    async def log_data_access(self,
                            operation: str,
                            database: str,
                            query_type: str,
                            records_affected: int,
                            duration_seconds: float,
                            country: Optional[str] = None,
                            success: bool = True,
                            error: Optional[str] = None) -> None:
        """
        Log data access operations
        
        Args:
            operation: Type of operation (SELECT, INSERT, UPDATE, etc.)
            database: Database name
            query_type: Type of query
            records_affected: Number of records affected
            duration_seconds: Query duration
            country: Country filter if applicable
            success: Whether operation was successful
            error: Error message if any
        """
        try:
            details = {
                'operation': operation,
                'database': database,
                'query_type': query_type,
                'records_affected': records_affected,
                'success': success
            }
            
            if error:
                details['error'] = error
            
            level = LogLevel.INFO if success else LogLevel.ERROR
            message = f"Data access: {operation} on {database} ({records_affected} records)"
            
            log_entry = LogEntry(
                timestamp=datetime.now(),
                level=level,
                category=LogCategory.DATA_ACCESS,
                message=message,
                details=details,
                country=country,
                duration_seconds=duration_seconds
            )
            
            await self._store_log_entry(log_entry)
            
        except Exception as e:
            logger.error(f"Failed to log data access: {e}")
    
    async def get_logs(self,
                      start_date: Optional[date] = None,
                      end_date: Optional[date] = None,
                      level: Optional[LogLevel] = None,
                      category: Optional[LogCategory] = None,
                      country: Optional[str] = None,
                      model_type: Optional[str] = None,
                      limit: int = 1000,
                      offset: int = 0) -> Dict[str, Any]:
        """
        Retrieve logs with filtering
        
        Args:
            start_date: Start date filter
            end_date: End date filter
            level: Log level filter
            category: Category filter
            country: Country filter
            model_type: Model type filter
            limit: Maximum number of records
            offset: Offset for pagination
            
        Returns:
            Filtered logs and metadata
        """
        try:
            # Build query
            conditions = []
            params = []
            
            if start_date:
                conditions.append("timestamp >= ?")
                params.append(start_date.isoformat())
            
            if end_date:
                conditions.append("timestamp <= ?")
                params.append((end_date + timedelta(days=1)).isoformat())
            
            if level:
                conditions.append("level = ?")
                params.append(level.value)
            
            if category:
                conditions.append("category = ?")
                params.append(category.value)
            
            if country:
                conditions.append("country = ?")
                params.append(country)
            
            if model_type:
                conditions.append("model_type = ?")
                params.append(model_type)
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            # Count total records
            count_query = f"""
                SELECT COUNT(*) FROM api_logs 
                WHERE {where_clause}
            """
            
            # Fetch logs
            logs_query = f"""
                SELECT * FROM api_logs 
                WHERE {where_clause}
                ORDER BY timestamp DESC 
                LIMIT ? OFFSET ?
            """
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get total count
                cursor.execute(count_query, params)
                total_count = cursor.fetchone()[0]
                
                # Get logs
                cursor.execute(logs_query, params + [limit, offset])
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                
                logs = []
                for row in rows:
                    log_dict = dict(zip(columns, row))
                    # Parse JSON details
                    if log_dict['details']:
                        try:
                            log_dict['details'] = json.loads(log_dict['details'])
                        except json.JSONDecodeError:
                            log_dict['details'] = {}
                    logs.append(log_dict)
                
                return {
                    'logs': logs,
                    'total_count': total_count,
                    'limit': limit,
                    'offset': offset,
                    'has_more': offset + len(logs) < total_count
                }
                
        except Exception as e:
            logger.error(f"Failed to retrieve logs: {e}")
            raise
    
    async def get_performance_metrics(self,
                                    start_date: Optional[date] = None,
                                    end_date: Optional[date] = None,
                                    endpoint: Optional[str] = None,
                                    country: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance metrics
        
        Args:
            start_date: Start date filter
            end_date: End date filter
            endpoint: Endpoint filter
            country: Country filter
            
        Returns:
            Performance metrics and statistics
        """
        try:
            conditions = []
            params = []
            
            if start_date:
                conditions.append("timestamp >= ?")
                params.append(start_date.isoformat())
            
            if end_date:
                conditions.append("timestamp <= ?")
                params.append((end_date + timedelta(days=1)).isoformat())
            
            if endpoint:
                conditions.append("endpoint = ?")
                params.append(endpoint)
            
            if country:
                conditions.append("country = ?")
                params.append(country)
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            query = f"""
                SELECT 
                    endpoint,
                    COUNT(*) as request_count,
                    AVG(response_time_ms) as avg_response_time,
                    MIN(response_time_ms) as min_response_time,
                    MAX(response_time_ms) as max_response_time,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY response_time_ms) as median_response_time,
                    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time_ms) as p95_response_time,
                    AVG(CASE WHEN status_code >= 400 THEN 1.0 ELSE 0.0 END) * 100 as error_rate,
                    AVG(memory_usage_mb) as avg_memory_usage,
                    AVG(cpu_usage_percent) as avg_cpu_usage
                FROM performance_metrics 
                WHERE {where_clause}
                GROUP BY endpoint
                ORDER BY request_count DESC
            """
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                
                metrics = []
                for row in rows:
                    metric_dict = dict(zip(columns, row))
                    metrics.append(metric_dict)
                
                # Overall statistics
                overall_query = f"""
                    SELECT 
                        COUNT(*) as total_requests,
                        AVG(response_time_ms) as overall_avg_response_time,
                        AVG(CASE WHEN status_code >= 400 THEN 1.0 ELSE 0.0 END) * 100 as overall_error_rate,
                        COUNT(DISTINCT DATE(timestamp)) as active_days
                    FROM performance_metrics 
                    WHERE {where_clause}
                """
                
                cursor.execute(overall_query, params)
                overall_row = cursor.fetchone()
                overall_stats = dict(zip([desc[0] for desc in cursor.description], overall_row))
                
                return {
                    'endpoint_metrics': metrics,
                    'overall_statistics': overall_stats,
                    'period': {
                        'start_date': start_date.isoformat() if start_date else None,
                        'end_date': end_date.isoformat() if end_date else None
                    }
                }
                
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            raise
    
    async def get_log_summary(self, days_back: int = 7) -> Dict[str, Any]:
        """
        Get log summary for the specified period
        
        Args:
            days_back: Number of days to look back
            
        Returns:
            Log summary statistics
        """
        try:
            start_date = (datetime.now() - timedelta(days=days_back)).date()
            
            query = """
                SELECT 
                    level,
                    category,
                    COUNT(*) as count,
                    country,
                    model_type
                FROM api_logs 
                WHERE timestamp >= ?
                GROUP BY level, category, country, model_type
                ORDER BY count DESC
            """
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, [start_date.isoformat()])
                
                summary_data = []
                for row in cursor.fetchall():
                    summary_data.append({
                        'level': row[0],
                        'category': row[1],
                        'count': row[2],
                        'country': row[3],
                        'model_type': row[4]
                    })
                
                # Top errors
                error_query = """
                    SELECT message, COUNT(*) as count
                    FROM api_logs 
                    WHERE timestamp >= ? AND level IN ('ERROR', 'CRITICAL')
                    GROUP BY message
                    ORDER BY count DESC
                    LIMIT 10
                """
                
                cursor.execute(error_query, [start_date.isoformat()])
                top_errors = [{'message': row[0], 'count': row[1]} for row in cursor.fetchall()]
                
                # Activity by day
                daily_query = """
                    SELECT 
                        DATE(timestamp) as date,
                        COUNT(*) as total_logs,
                        COUNT(CASE WHEN level = 'ERROR' THEN 1 END) as error_logs
                    FROM api_logs 
                    WHERE timestamp >= ?
                    GROUP BY DATE(timestamp)
                    ORDER BY date
                """
                
                cursor.execute(daily_query, [start_date.isoformat()])
                daily_activity = [
                    {'date': row[0], 'total_logs': row[1], 'error_logs': row[2]} 
                    for row in cursor.fetchall()
                ]
                
                return {
                    'period_days': days_back,
                    'summary_data': summary_data,
                    'top_errors': top_errors,
                    'daily_activity': daily_activity,
                    'generated_at': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Failed to get log summary: {e}")
            raise
    
    async def _store_log_entry(self, log_entry: LogEntry) -> None:
        """Store log entry in database"""
        try:
            log_dict = log_entry.to_dict()
            
            query = """
                INSERT INTO api_logs (
                    timestamp, level, category, message, details,
                    country, model_type, user_id, request_id, duration_seconds
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            values = (
                log_dict['timestamp'],
                log_dict['level'],
                log_dict['category'],
                log_dict['message'],
                log_dict['details'],
                log_dict['country'],
                log_dict['model_type'],
                log_dict['user_id'],
                log_dict['request_id'],
                log_dict['duration_seconds']
            )
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, values)
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to store log entry: {e}")
    
    async def _store_performance_metric(self,
                                      endpoint: str,
                                      method: str,
                                      response_time_ms: float,
                                      status_code: int,
                                      country: Optional[str] = None,
                                      model_type: Optional[str] = None) -> None:
        """Store performance metric"""
        try:
            query = """
                INSERT INTO performance_metrics (
                    timestamp, endpoint, method, response_time_ms, status_code,
                    country, model_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            
            values = (
                datetime.now().isoformat(),
                endpoint,
                method,
                response_time_ms,
                status_code,
                country,
                model_type
            )
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, values)
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to store performance metric: {e}")
    
    def _sanitize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize sensitive data from logs"""
        sensitive_keys = ['password', 'token', 'api_key', 'secret', 'auth']
        
        if isinstance(data, dict):
            sanitized = {}
            for key, value in data.items():
                if any(sensitive in key.lower() for sensitive in sensitive_keys):
                    sanitized[key] = "***REDACTED***"
                elif isinstance(value, dict):
                    sanitized[key] = self._sanitize_data(value)
                elif isinstance(value, list) and len(value) > 100:
                    # Truncate large lists in logs
                    sanitized[key] = f"[{len(value)} items]"
                else:
                    sanitized[key] = value
            return sanitized
        
        return data
    
    async def _periodic_cleanup(self) -> None:
        """Periodic cleanup of old logs"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval_hours * 3600)  # Convert to seconds
                await self._cleanup_old_logs()
            except Exception as e:
                logger.error(f"Error in periodic log cleanup: {e}")
    
    async def _cleanup_old_logs(self) -> None:
        """Clean up logs older than retention period"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete old logs
                cursor.execute(
                    "DELETE FROM api_logs WHERE timestamp < ?",
                    [cutoff_date.isoformat()]
                )
                
                # Delete old performance metrics
                cursor.execute(
                    "DELETE FROM performance_metrics WHERE timestamp < ?",
                    [cutoff_date.isoformat()]
                )
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                if deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} old log entries")
                
        except Exception as e:
            logger.error(f"Failed to cleanup old logs: {e}")


# Global log service instance
log_service = LogService()
