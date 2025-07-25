#!/usr/bin/env python3
"""
Data Ingestion Scheduler
========================

This module provides scheduling capabilities for automated data ingestion runs.
It supports various scheduling options including cron-like scheduling and
one-time runs at specific times.

Author: Adryan R A

Features:
- Midnight scheduling for daily data ingestion
- Flexible cron-like scheduling
- Email notifications (optional)
- Logging and monitoring
- Error handling and recovery
- Health checks

Author: AI Assistant
Date: 2025-07-25
"""

import os
import sys
import time
import logging
import schedule
import threading
from datetime import datetime, time as dt_time
from typing import Optional, Callable, Dict, Any
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_ingestion import DataIngestionEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataIngestionScheduler:
    """
    Scheduler for automated data ingestion tasks.
    """
    
    def __init__(self, data_directory: str = "cs-train", database_url: str = "sqlite:///file.db",
                 notification_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the scheduler.
        
        Args:
            data_directory: Directory containing JSON files
            database_url: Database URL for SQLAlchemy
            notification_config: Optional configuration for notifications
        """
        self.data_directory = data_directory
        self.database_url = database_url
        self.notification_config = notification_config or {}
        self.ingestion_engine = DataIngestionEngine(database_url, data_directory)
        self.running = False
        self.scheduler_thread = None
        
        logger.info(f"Scheduler initialized for directory: {data_directory}")
    
    def _send_notification(self, subject: str, message: str, is_error: bool = False):
        """
        Send notification (email, webhook, etc.) about job status.
        
        Args:
            subject: Subject line
            message: Message body
            is_error: Whether this is an error notification
        """
        # This is a placeholder for notification functionality
        # In a real implementation, you would integrate with email services,
        # Slack webhooks, or other notification systems
        
        log_func = logger.error if is_error else logger.info
        log_func(f"NOTIFICATION: {subject} - {message}")
        
        # Example: Simple email notification (requires setup)
        if self.notification_config.get('email_enabled', False):
            try:
                # Placeholder for email sending logic
                # import smtplib
                # from email.mime.text import MIMEText
                # ... email sending code ...
                pass
            except Exception as e:
                logger.error(f"Failed to send email notification: {e}")
    
    def run_ingestion_job(self) -> Dict[str, Any]:
        """
        Run a single data ingestion job.
        
        Returns:
            Dictionary with job results
        """
        job_start_time = datetime.now()
        logger.info(f"Starting scheduled data ingestion job at {job_start_time}")
        
        try:
            # Check if data directory exists
            if not os.path.exists(self.data_directory):
                raise FileNotFoundError(f"Data directory not found: {self.data_directory}")
            
            # Run ingestion
            results = self.ingestion_engine.process_all_files(force_reprocess=False)
            
            job_end_time = datetime.now()
            job_duration = (job_end_time - job_start_time).total_seconds()
            
            # Prepare summary
            summary = {
                'job_start_time': job_start_time,
                'job_end_time': job_end_time,
                'job_duration': job_duration,
                'success': True,
                **results
            }
            
            # Send success notification
            if results['files_processed'] > 0:
                self._send_notification(
                    subject="Data Ingestion Job Completed Successfully",
                    message=f"Processed {results['files_processed']} files with {results['total_records']} records in {job_duration:.2f} seconds"
                )
            else:
                logger.info("No new files to process")
            
            return summary
            
        except Exception as e:
            job_end_time = datetime.now()
            job_duration = (job_end_time - job_start_time).total_seconds()
            
            error_summary = {
                'job_start_time': job_start_time,
                'job_end_time': job_end_time,
                'job_duration': job_duration,
                'success': False,
                'error': str(e),
                'files_processed': 0,
                'total_records': 0,
                'total_errors': 1
            }
            
            logger.error(f"Data ingestion job failed: {e}")
            
            # Send error notification
            self._send_notification(
                subject="Data Ingestion Job Failed",
                message=f"Job failed after {job_duration:.2f} seconds. Error: {e}",
                is_error=True
            )
            
            return error_summary
    
    def schedule_midnight_run(self):
        """
        Schedule data ingestion to run every day at midnight.
        """
        schedule.every().day.at("00:00").do(self.run_ingestion_job)
        logger.info("Scheduled daily data ingestion at midnight (00:00)")
    
    def schedule_daily_at(self, time_str: str):
        """
        Schedule data ingestion to run daily at a specific time.
        
        Args:
            time_str: Time in HH:MM format (e.g., "02:30")
        """
        schedule.every().day.at(time_str).do(self.run_ingestion_job)
        logger.info(f"Scheduled daily data ingestion at {time_str}")
    
    def schedule_hourly(self):
        """Schedule data ingestion to run every hour."""
        schedule.every().hour.do(self.run_ingestion_job)
        logger.info("Scheduled hourly data ingestion")
    
    def schedule_custom(self, cron_expression: str):
        """
        Schedule with custom cron-like expression.
        Note: This is a simplified implementation. For full cron support,
        consider using the 'python-crontab' library.
        
        Args:
            cron_expression: Simplified cron expression (e.g., "every 30 minutes")
        """
        # Simple parsing for common expressions
        if "minute" in cron_expression:
            if "30" in cron_expression:
                schedule.every(30).minutes.do(self.run_ingestion_job)
                logger.info("Scheduled data ingestion every 30 minutes")
            elif "15" in cron_expression:
                schedule.every(15).minutes.do(self.run_ingestion_job)
                logger.info("Scheduled data ingestion every 15 minutes")
        else:
            logger.warning(f"Unsupported cron expression: {cron_expression}")
    
    def run_scheduler(self):
        """
        Run the scheduler in a loop.
        """
        self.running = True
        logger.info("Scheduler started")
        
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except KeyboardInterrupt:
                logger.info("Scheduler interrupted by user")
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(60)  # Wait before retrying
        
        logger.info("Scheduler stopped")
    
    def start_background_scheduler(self):
        """
        Start the scheduler in a background thread.
        """
        if self.scheduler_thread is None or not self.scheduler_thread.is_alive():
            self.scheduler_thread = threading.Thread(target=self.run_scheduler, daemon=True)
            self.scheduler_thread.start()
            logger.info("Background scheduler started")
        else:
            logger.warning("Scheduler is already running")
    
    def stop_scheduler(self):
        """
        Stop the scheduler.
        """
        self.running = False
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5)
        logger.info("Scheduler stop requested")
    
    def get_scheduled_jobs(self) -> list:
        """
        Get list of scheduled jobs.
        
        Returns:
            List of scheduled jobs
        """
        jobs = []
        for job in schedule.jobs:
            jobs.append({
                'job': str(job.job_func),
                'next_run': job.next_run,
                'interval': job.interval,
                'unit': job.unit
            })
        return jobs
    
    def clear_all_jobs(self):
        """Clear all scheduled jobs."""
        schedule.clear()
        logger.info("All scheduled jobs cleared")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check of the ingestion system.
        
        Returns:
            Dictionary with health check results
        """
        health_status = {
            'timestamp': datetime.now(),
            'scheduler_running': self.running,
            'data_directory_exists': os.path.exists(self.data_directory),
            'database_accessible': False,
            'scheduled_jobs_count': len(schedule.jobs),
            'errors': []
        }
        
        try:
            # Test database connection
            stats = self.ingestion_engine.get_database_stats()
            health_status['database_accessible'] = True
            health_status['database_stats'] = stats
        except Exception as e:
            health_status['errors'].append(f"Database error: {e}")
        
        # Check data directory
        if not health_status['data_directory_exists']:
            health_status['errors'].append(f"Data directory not found: {self.data_directory}")
        else:
            # Count JSON files
            json_files = len([f for f in os.listdir(self.data_directory) if f.endswith('.json')])
            health_status['json_files_count'] = json_files
        
        # Overall health
        health_status['status'] = 'healthy' if not health_status['errors'] else 'unhealthy'
        
        return health_status


def create_systemd_service(scheduler_script_path: str, service_name: str = "data-ingestion"):
    """
    Create a systemd service file for running the scheduler as a daemon.
    
    Args:
        scheduler_script_path: Full path to the scheduler script
        service_name: Name of the systemd service
    """
    service_content = f"""[Unit]
Description=Data Ingestion Scheduler
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory={os.path.dirname(scheduler_script_path)}
ExecStart=/usr/bin/python3 {scheduler_script_path} --daemon
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
    
    service_file_path = f"/etc/systemd/system/{service_name}.service"
    
    print(f"Systemd service file content:")
    print("=" * 50)
    print(service_content)
    print("=" * 50)
    print(f"\nTo install this service:")
    print(f"1. Save the content above to {service_file_path}")
    print(f"2. Run: sudo systemctl daemon-reload")
    print(f"3. Run: sudo systemctl enable {service_name}")
    print(f"4. Run: sudo systemctl start {service_name}")


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Data Ingestion Scheduler')
    parser.add_argument('--data-dir', default='cs-train', help='Directory containing JSON files')
    parser.add_argument('--database-url', default='sqlite:///file.db', help='Database URL')
    parser.add_argument('--midnight', action='store_true', help='Schedule for midnight runs')
    parser.add_argument('--daily-at', help='Schedule daily at specific time (HH:MM)')
    parser.add_argument('--hourly', action='store_true', help='Schedule hourly runs')
    parser.add_argument('--run-once', action='store_true', help='Run ingestion job once and exit')
    parser.add_argument('--daemon', action='store_true', help='Run as daemon (keeps running)')
    parser.add_argument('--health-check', action='store_true', help='Perform health check')
    parser.add_argument('--create-service', action='store_true', help='Generate systemd service file')
    
    args = parser.parse_args()
    
    # Initialize scheduler
    scheduler = DataIngestionScheduler(
        data_directory=args.data_dir,
        database_url=args.database_url
    )
    
    if args.health_check:
        health = scheduler.health_check()
        print("\nHealth Check Results:")
        print("=" * 50)
        for key, value in health.items():
            if key != 'errors':
                print(f"{key}: {value}")
        
        if health['errors']:
            print(f"\nErrors:")
            for error in health['errors']:
                print(f"  - {error}")
        return
    
    if args.create_service:
        create_systemd_service(os.path.abspath(__file__))
        return
    
    if args.run_once:
        results = scheduler.run_ingestion_job()
        print("\nIngestion Results:")
        print("=" * 50)
        for key, value in results.items():
            print(f"{key}: {value}")
        return
    
    # Set up scheduling
    if args.midnight:
        scheduler.schedule_midnight_run()
    elif args.daily_at:
        scheduler.schedule_daily_at(args.daily_at)
    elif args.hourly:
        scheduler.schedule_hourly()
    else:
        # Default to midnight
        scheduler.schedule_midnight_run()
    
    if args.daemon:
        try:
            scheduler.run_scheduler()
        except KeyboardInterrupt:
            print("\nScheduler stopped by user")
            scheduler.stop_scheduler()
    else:
        print("\nScheduled jobs:")
        for job in scheduler.get_scheduled_jobs():
            print(f"  - {job}")
        
        print(f"\nTo run the scheduler, use --daemon flag")
        print(f"To run once immediately, use --run-once flag")


if __name__ == "__main__":
    main()
