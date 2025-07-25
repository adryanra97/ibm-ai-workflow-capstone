#!/usr/bin/env python3
"""
AI Workflow Capstone - Data Ingestion System
===========================================

This module provides a comprehensive data ingestion system for invoice data from multiple JSON files.
It handles inconsistent field naming, creates a robust database schema, and provides scheduling 
capabilities for automated data ingestion.

Author: Adryan R A

Features:
- Automatic handling of inconsistent JSON field names
- SQLAlchemy-based database operations
- Duplicate detection and handling
- Error handling and logging
- Scheduling for automated runs
- Future-proof design for new data sources

Author: AI Assistant
Date: 2025-07-25
"""

import json
import os
import glob
import logging
import hashlib
from datetime import datetime, date
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import pandas as pd
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, Date, 
    Boolean, Text, UniqueConstraint, Index
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_ingestion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = "sqlite:///file.db"
Base = declarative_base()


class InvoiceRecord(Base):
    """
    SQLAlchemy model for invoice records with comprehensive fields and constraints.
    """
    __tablename__ = 'invoice_records'
    
    # Primary key
    id = Column(Integer, primary_key=True)
    
    # Invoice information
    invoice = Column(String(50), nullable=False, index=True)
    country = Column(String(100), nullable=False, index=True)
    customer_id = Column(Float, nullable=True, index=True)  # Can be null based on data
    
    # Product/Stream information
    stream_id = Column(String(50), nullable=False, index=True)
    price = Column(Float, nullable=False)
    times_viewed = Column(Integer, nullable=False, default=0)
    
    # Date information
    year = Column(Integer, nullable=False, index=True)
    month = Column(Integer, nullable=False, index=True)
    day = Column(Integer, nullable=False, index=True)
    
    # Computed date field for easier querying
    date_field = Column(Date, nullable=False, index=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    source_file = Column(String(255), nullable=False)
    record_hash = Column(String(64), nullable=False, index=True)  # For duplicate detection
    
    # Constraints
    __table_args__ = (
        # Unique constraint to prevent exact duplicates
        UniqueConstraint('record_hash', name='uq_record_hash'),
        # Index for common query patterns
        Index('idx_date_country', 'date_field', 'country'),
        Index('idx_invoice_customer', 'invoice', 'customer_id'),
        Index('idx_stream_date', 'stream_id', 'date_field'),
    )


class FileProcessingLog(Base):
    """
    SQLAlchemy model to track which files have been processed.
    """
    __tablename__ = 'file_processing_log'
    
    id = Column(Integer, primary_key=True)
    file_path = Column(String(500), nullable=False, unique=True)
    file_name = Column(String(255), nullable=False)
    file_size = Column(Integer, nullable=False)
    file_modified_time = Column(DateTime, nullable=False)
    records_processed = Column(Integer, nullable=False, default=0)
    processing_status = Column(String(20), nullable=False, default='pending')  # pending, completed, failed
    error_message = Column(Text, nullable=True)
    processed_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    __table_args__ = (
        Index('idx_file_status', 'file_name', 'processing_status'),
    )


class DataIngestionEngine:
    """
    Main data ingestion engine for processing invoice JSON files.
    """
    
    def __init__(self, database_url: str = DATABASE_URL, data_directory: str = "cs-train"):
        """
        Initialize the data ingestion engine.
        
        Args:
            database_url: SQLAlchemy database URL
            data_directory: Directory containing JSON files to process
        """
        self.database_url = database_url
        self.data_directory = data_directory
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Field name mapping to handle inconsistencies
        self.field_mappings = {
            'stream_id': ['stream_id', 'StreamID'],
            'times_viewed': ['times_viewed', 'TimesViewed'],
            'customer_id': ['customer_id'],
            'invoice': ['invoice'],
            'country': ['country'],
            'price': ['price', 'total_price'],  # Handle both price and total_price
            'year': ['year'],
            'month': ['month'],
            'day': ['day']
        }
        
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created/verified successfully")
        except SQLAlchemyError as e:
            logger.error(f"Error creating database tables: {e}")
            raise
    
    def _normalize_field_names(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize field names in a record to handle inconsistencies across files.
        
        Args:
            record: Raw record from JSON file
            
        Returns:
            Normalized record with consistent field names
        """
        normalized = {}
        
        for standard_field, possible_fields in self.field_mappings.items():
            value = None
            for field_name in possible_fields:
                if field_name in record:
                    value = record[field_name]
                    break
            
            if value is not None:
                normalized[standard_field] = value
            else:
                logger.warning(f"Field {standard_field} not found in record: {record}")
                
        return normalized
    
    def _generate_record_hash(self, record: Dict[str, Any]) -> str:
        """
        Generate a hash for a record to detect duplicates.
        
        Args:
            record: Normalized record
            
        Returns:
            MD5 hash of the record
        """
        # Create a string representation of key fields for hashing
        hash_fields = [
            str(record.get('invoice', '')),
            str(record.get('customer_id', '')),
            str(record.get('stream_id', '')),
            str(record.get('price', '')),
            str(record.get('times_viewed', '')),
            str(record.get('year', '')),
            str(record.get('month', '')),
            str(record.get('day', ''))
        ]
        
        hash_string = '|'.join(hash_fields)
        return hashlib.md5(hash_string.encode('utf-8')).hexdigest()
    
    def _process_json_file(self, file_path: str, session: Session) -> Tuple[int, List[str]]:
        """
        Process a single JSON file and insert records into database.
        
        Args:
            file_path: Path to the JSON file
            session: SQLAlchemy session
            
        Returns:
            Tuple of (records_processed, errors)
        """
        logger.info(f"Processing file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError, UnicodeDecodeError) as e:
            error_msg = f"Error reading file {file_path}: {e}"
            logger.error(error_msg)
            return 0, [error_msg]
        
        if not isinstance(data, list):
            error_msg = f"Expected list in {file_path}, got {type(data)}"
            logger.error(error_msg)
            return 0, [error_msg]
        
        records_processed = 0
        errors = []
        
        for i, raw_record in enumerate(data):
            try:
                # Normalize field names
                normalized_record = self._normalize_field_names(raw_record)
                
                # Validate required fields
                required_fields = ['invoice', 'country', 'stream_id', 'price', 'year', 'month', 'day']
                missing_fields = [field for field in required_fields if field not in normalized_record]
                
                if missing_fields:
                    error_msg = f"Missing required fields in record {i}: {missing_fields}"
                    errors.append(error_msg)
                    continue
                
                # Create date field
                try:
                    record_date = date(
                        int(normalized_record['year']),
                        int(normalized_record['month']),
                        int(normalized_record['day'])
                    )
                except (ValueError, TypeError) as e:
                    error_msg = f"Invalid date in record {i}: {e}"
                    errors.append(error_msg)
                    continue
                
                # Generate record hash
                record_hash = self._generate_record_hash(normalized_record)
                
                # Create InvoiceRecord instance
                invoice_record = InvoiceRecord(
                    invoice=str(normalized_record['invoice']),
                    country=str(normalized_record['country']),
                    customer_id=float(normalized_record['customer_id']) if normalized_record.get('customer_id') is not None else None,
                    stream_id=str(normalized_record['stream_id']),
                    price=float(normalized_record['price']),
                    times_viewed=int(normalized_record.get('times_viewed', 0)),
                    year=int(normalized_record['year']),
                    month=int(normalized_record['month']),
                    day=int(normalized_record['day']),
                    date_field=record_date,
                    source_file=os.path.basename(file_path),
                    record_hash=record_hash
                )
                
                # Add to session
                session.add(invoice_record)
                records_processed += 1
                
                # Commit in batches to avoid memory issues
                if records_processed % 1000 == 0:
                    try:
                        session.commit()
                        logger.info(f"Committed batch: {records_processed} records processed")
                    except IntegrityError:
                        session.rollback()
                        logger.warning(f"Duplicate record skipped in batch at record {records_processed}")
                
            except Exception as e:
                error_msg = f"Error processing record {i} in {file_path}: {e}"
                errors.append(error_msg)
                logger.error(error_msg)
        
        # Final commit
        try:
            session.commit()
        except IntegrityError as e:
            session.rollback()
            logger.warning(f"Some duplicate records were skipped: {e}")
        
        return records_processed, errors
    
    def _log_file_processing(self, session: Session, file_path: str, records_processed: int, 
                           status: str, error_message: str = None):
        """
        Log file processing status to the database.
        
        Args:
            session: SQLAlchemy session
            file_path: Path to the processed file
            records_processed: Number of records processed
            status: Processing status (completed, failed)
            error_message: Error message if processing failed
        """
        file_stat = os.stat(file_path)
        
        # Check if file already logged
        existing_log = session.query(FileProcessingLog).filter_by(file_path=file_path).first()
        
        if existing_log:
            existing_log.records_processed = records_processed
            existing_log.processing_status = status
            existing_log.error_message = error_message
            existing_log.processed_at = datetime.utcnow()
        else:
            file_log = FileProcessingLog(
                file_path=file_path,
                file_name=os.path.basename(file_path),
                file_size=file_stat.st_size,
                file_modified_time=datetime.fromtimestamp(file_stat.st_mtime),
                records_processed=records_processed,
                processing_status=status,
                error_message=error_message
            )
            session.add(file_log)
        
        session.commit()
    
    def get_files_to_process(self) -> List[str]:
        """
        Get list of JSON files that need to be processed.
        
        Returns:
            List of file paths to process
        """
        json_files = glob.glob(os.path.join(self.data_directory, "*.json"))
        
        with self.SessionLocal() as session:
            processed_files = session.query(FileProcessingLog.file_path).filter_by(
                processing_status='completed'
            ).all()
            processed_file_paths = {row[0] for row in processed_files}
        
        # Return files that haven't been successfully processed
        files_to_process = [f for f in json_files if f not in processed_file_paths]
        
        logger.info(f"Found {len(json_files)} total JSON files, {len(files_to_process)} to process")
        return files_to_process
    
    def process_all_files(self, force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Process all JSON files in the data directory.
        
        Args:
            force_reprocess: If True, reprocess all files regardless of previous processing
            
        Returns:
            Dictionary with processing statistics
        """
        start_time = datetime.utcnow()
        
        if force_reprocess:
            json_files = glob.glob(os.path.join(self.data_directory, "*.json"))
        else:
            json_files = self.get_files_to_process()
        
        if not json_files:
            logger.info("No files to process")
            return {
                'files_processed': 0,
                'total_records': 0,
                'total_errors': 0,
                'processing_time': 0,
                'status': 'no_files'
            }
        
        total_records = 0
        total_errors = 0
        files_processed = 0
        processing_errors = []
        
        with self.SessionLocal() as session:
            for file_path in json_files:
                try:
                    records_processed, errors = self._process_json_file(file_path, session)
                    
                    self._log_file_processing(
                        session, file_path, records_processed, 'completed'
                    )
                    
                    total_records += records_processed
                    total_errors += len(errors)
                    files_processed += 1
                    processing_errors.extend(errors)
                    
                    logger.info(f"Successfully processed {file_path}: {records_processed} records")
                    
                except Exception as e:
                    error_msg = f"Failed to process {file_path}: {e}"
                    processing_errors.append(error_msg)
                    logger.error(error_msg)
                    
                    self._log_file_processing(
                        session, file_path, 0, 'failed', error_msg
                    )
        
        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds()
        
        # Log summary
        logger.info(f"Processing completed:")
        logger.info(f"  Files processed: {files_processed}")
        logger.info(f"  Total records: {total_records}")
        logger.info(f"  Total errors: {total_errors}")
        logger.info(f"  Processing time: {processing_time:.2f} seconds")
        
        return {
            'files_processed': files_processed,
            'total_records': total_records,
            'total_errors': total_errors,
            'processing_errors': processing_errors,
            'processing_time': processing_time,
            'status': 'completed'
        }
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the database content.
        
        Returns:
            Dictionary with database statistics
        """
        with self.SessionLocal() as session:
            total_records = session.query(InvoiceRecord).count()
            unique_invoices = session.query(InvoiceRecord.invoice).distinct().count()
            unique_customers = session.query(InvoiceRecord.customer_id).distinct().count()
            unique_countries = session.query(InvoiceRecord.country).distinct().count()
            unique_streams = session.query(InvoiceRecord.stream_id).distinct().count()
            
            date_range = session.query(
                InvoiceRecord.date_field.label('min_date'),
                InvoiceRecord.date_field.label('max_date')
            ).first()
            
            # Get earliest and latest dates
            min_date = session.query(InvoiceRecord.date_field).order_by(InvoiceRecord.date_field.asc()).first()
            max_date = session.query(InvoiceRecord.date_field).order_by(InvoiceRecord.date_field.desc()).first()
            
            return {
                'total_records': total_records,
                'unique_invoices': unique_invoices,
                'unique_customers': unique_customers,
                'unique_countries': unique_countries,
                'unique_streams': unique_streams,
                'date_range': {
                    'min_date': min_date[0] if min_date else None,
                    'max_date': max_date[0] if max_date else None
                }
            }
    
    def export_to_dataframe(self, limit: int = None) -> pd.DataFrame:
        """
        Export database records to a pandas DataFrame.
        
        Args:
            limit: Maximum number of records to export
            
        Returns:
            Pandas DataFrame with invoice records
        """
        with self.SessionLocal() as session:
            query = session.query(InvoiceRecord)
            
            if limit:
                query = query.limit(limit)
            
            # Convert to DataFrame
            records = query.all()
            
            data = []
            for record in records:
                data.append({
                    'id': record.id,
                    'invoice': record.invoice,
                    'country': record.country,
                    'customer_id': record.customer_id,
                    'stream_id': record.stream_id,
                    'price': record.price,
                    'times_viewed': record.times_viewed,
                    'year': record.year,
                    'month': record.month,
                    'day': record.day,
                    'date': record.date_field,
                    'created_at': record.created_at,
                    'source_file': record.source_file
                })
            
            return pd.DataFrame(data)


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Data Ingestion System for Invoice Data')
    parser.add_argument('--data-dir', default='cs-train', help='Directory containing JSON files')
    parser.add_argument('--database-url', default=DATABASE_URL, help='Database URL')
    parser.add_argument('--force-reprocess', action='store_true', help='Reprocess all files')
    parser.add_argument('--stats', action='store_true', help='Show database statistics')
    parser.add_argument('--export-csv', help='Export data to CSV file')
    
    args = parser.parse_args()
    
    # Initialize engine
    engine = DataIngestionEngine(
        database_url=args.database_url,
        data_directory=args.data_dir
    )
    
    if args.stats:
        stats = engine.get_database_stats()
        print("\nDatabase Statistics:")
        print("=" * 50)
        for key, value in stats.items():
            print(f"{key}: {value}")
        return
    
    if args.export_csv:
        df = engine.export_to_dataframe()
        df.to_csv(args.export_csv, index=False)
        print(f"Data exported to {args.export_csv}")
        return
    
    # Process files
    results = engine.process_all_files(force_reprocess=args.force_reprocess)
    
    print("\nProcessing Results:")
    print("=" * 50)
    for key, value in results.items():
        if key != 'processing_errors':
            print(f"{key}: {value}")
    
    if results.get('processing_errors', []):
        print(f"\nErrors ({len(results['processing_errors'])}):")
        for error in results['processing_errors'][:10]:  # Show first 10 errors
            print(f"  - {error}")


if __name__ == "__main__":
    main()
