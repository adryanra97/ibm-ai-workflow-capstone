#!/usr/bin/env python3
"""
AI Workflow Capstone - Data Cleaning and Aggregation Pipeline
============================================================

This module processes raw invoice data from the L0 (ingestion) database,
performs comprehensive data cleaning, validation, and aggregation by date,
then saves the results to the L3 (analytics-ready) database.

Author: Adryan R A

Features:
- Data quality validation and cleaning
- Outlier detection and handling
- Date-based aggregation with multiple metrics
- Customer segmentation and analysis
- Product performance metrics
- Geographic insights
- Time series analysis preparation
- Comprehensive error handling and logging

Database Structure:
- L0 Database (file.db): Raw ingested data from JSON files
- L3 Database (analytics.db): Cleaned and aggregated data for analysis

Author: AI Assistant
Date: 2025-07-25
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, Date, 
    Boolean, Text, Index, func, and_, or_
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
import warnings
warnings.filterwarnings('ignore')

# Import from data ingestion module
from data_ingestion import DataIngestionEngine, InvoiceRecord

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_cleaning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Database configuration
L0_DATABASE_URL = "sqlite:///file.db"  # L0 - Raw ingestion database
L3_DATABASE_URL = "sqlite:///analytics.db"  # L3 - Cleaned analytics database

Base = declarative_base()


class DailyAggregatedMetrics(Base):
    """
    L3 Database - Daily aggregated metrics for business analytics.
    """
    __tablename__ = 'daily_aggregated_metrics'
    
    # Primary key
    id = Column(Integer, primary_key=True)
    
    # Date dimension
    date = Column(Date, nullable=False, index=True)
    year = Column(Integer, nullable=False, index=True)
    month = Column(Integer, nullable=False, index=True)
    day = Column(Integer, nullable=False, index=True)
    day_of_week = Column(Integer, nullable=False)  # 0=Monday, 6=Sunday
    week_of_year = Column(Integer, nullable=False)
    
    # Geographic dimension
    country = Column(String(100), nullable=False, index=True)
    
    # Revenue metrics
    total_revenue = Column(Float, nullable=False, default=0.0)
    avg_transaction_value = Column(Float, nullable=False, default=0.0)
    min_transaction_value = Column(Float, nullable=False, default=0.0)
    max_transaction_value = Column(Float, nullable=False, default=0.0)
    median_transaction_value = Column(Float, nullable=False, default=0.0)
    revenue_std_dev = Column(Float, nullable=False, default=0.0)
    
    # Transaction metrics
    total_transactions = Column(Integer, nullable=False, default=0)
    unique_invoices = Column(Integer, nullable=False, default=0)
    unique_customers = Column(Integer, nullable=False, default=0)
    unique_products = Column(Integer, nullable=False, default=0)
    
    # Customer metrics
    new_customers = Column(Integer, nullable=False, default=0)
    returning_customers = Column(Integer, nullable=False, default=0)
    customer_retention_rate = Column(Float, nullable=False, default=0.0)
    avg_customer_value = Column(Float, nullable=False, default=0.0)
    
    # Product metrics
    total_views = Column(Integer, nullable=False, default=0)
    avg_views_per_product = Column(Float, nullable=False, default=0.0)
    view_to_purchase_ratio = Column(Float, nullable=False, default=0.0)
    
    # Quality metrics
    data_quality_score = Column(Float, nullable=False, default=1.0)
    outlier_transactions = Column(Integer, nullable=False, default=0)
    missing_customer_ids = Column(Integer, nullable=False, default=0)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    source_record_count = Column(Integer, nullable=False, default=0)
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_date_country', 'date', 'country'),
        Index('idx_year_month', 'year', 'month'),
        Index('idx_country_revenue', 'country', 'total_revenue'),
    )


class CustomerSegmentMetrics(Base):
    """
    L3 Database - Customer segmentation metrics by date and country.
    """
    __tablename__ = 'customer_segment_metrics'
    
    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False, index=True)
    country = Column(String(100), nullable=False, index=True)
    
    # RFM Analysis components
    avg_recency_days = Column(Float, nullable=False, default=0.0)  # Days since last purchase
    avg_frequency = Column(Float, nullable=False, default=0.0)     # Purchase frequency
    avg_monetary_value = Column(Float, nullable=False, default=0.0) # Average customer value
    
    # Customer segments (based on RFM quintiles)
    champions_count = Column(Integer, nullable=False, default=0)      # High value, frequent, recent
    loyal_customers_count = Column(Integer, nullable=False, default=0) # High frequency, good recency
    potential_loyalists_count = Column(Integer, nullable=False, default=0) # Recent customers, low frequency
    new_customers_count = Column(Integer, nullable=False, default=0)  # Very recent, low frequency
    at_risk_count = Column(Integer, nullable=False, default=0)       # Good value but long time ago
    cannot_lose_count = Column(Integer, nullable=False, default=0)   # High value but long time ago
    hibernating_count = Column(Integer, nullable=False, default=0)   # Low value, long time ago
    
    # Customer lifetime value metrics
    avg_customer_lifetime_value = Column(Float, nullable=False, default=0.0)
    customer_acquisition_rate = Column(Float, nullable=False, default=0.0)
    customer_churn_risk_score = Column(Float, nullable=False, default=0.0)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    __table_args__ = (
        Index('idx_date_country_segment', 'date', 'country'),
    )


class ProductPerformanceMetrics(Base):
    """
    L3 Database - Product performance metrics by date and country.
    """
    __tablename__ = 'product_performance_metrics'
    
    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False, index=True)
    country = Column(String(100), nullable=False, index=True)
    stream_id = Column(String(50), nullable=False, index=True)
    
    # Performance metrics
    revenue = Column(Float, nullable=False, default=0.0)
    transactions = Column(Integer, nullable=False, default=0)
    unique_customers = Column(Integer, nullable=False, default=0)
    total_views = Column(Integer, nullable=False, default=0)
    
    # Derived metrics
    avg_price = Column(Float, nullable=False, default=0.0)
    conversion_rate = Column(Float, nullable=False, default=0.0)  # transactions/views
    customer_penetration = Column(Float, nullable=False, default=0.0)  # unique customers/total customers
    
    # Ranking metrics (within country/date)
    revenue_rank = Column(Integer, nullable=False, default=0)
    popularity_rank = Column(Integer, nullable=False, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    __table_args__ = (
        Index('idx_date_country_product', 'date', 'country', 'stream_id'),
        Index('idx_product_performance', 'stream_id', 'revenue'),
    )


class DataQualityReport(Base):
    """
    L3 Database - Data quality assessment and anomaly detection.
    """
    __tablename__ = 'data_quality_reports'
    
    id = Column(Integer, primary_key=True)
    report_date = Column(Date, nullable=False, index=True)
    country = Column(String(100), nullable=True, index=True)  # Null for global reports
    
    # Data completeness metrics
    total_records = Column(Integer, nullable=False, default=0)
    valid_records = Column(Integer, nullable=False, default=0)
    missing_customer_ids = Column(Integer, nullable=False, default=0)
    invalid_prices = Column(Integer, nullable=False, default=0)
    invalid_dates = Column(Integer, nullable=False, default=0)
    
    # Data quality scores
    completeness_score = Column(Float, nullable=False, default=1.0)
    consistency_score = Column(Float, nullable=False, default=1.0)
    accuracy_score = Column(Float, nullable=False, default=1.0)
    overall_quality_score = Column(Float, nullable=False, default=1.0)
    
    # Anomaly detection
    revenue_anomalies = Column(Integer, nullable=False, default=0)
    volume_anomalies = Column(Integer, nullable=False, default=0)
    price_outliers = Column(Integer, nullable=False, default=0)
    view_count_outliers = Column(Integer, nullable=False, default=0)
    
    # Statistical summaries
    revenue_z_score_max = Column(Float, nullable=False, default=0.0)
    price_iqr_outliers = Column(Integer, nullable=False, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    __table_args__ = (
        Index('idx_report_date_country', 'report_date', 'country'),
    )


class DataCleaningPipeline:
    """
    Main data cleaning and aggregation pipeline.
    """
    
    def __init__(self, l0_database_url: str = L0_DATABASE_URL, l3_database_url: str = L3_DATABASE_URL):
        """
        Initialize the data cleaning pipeline.
        
        Args:
            l0_database_url: L0 (raw ingestion) database URL
            l3_database_url: L3 (analytics) database URL
        """
        self.l0_database_url = l0_database_url
        self.l3_database_url = l3_database_url
        
        # L0 database connection (read-only)
        self.l0_engine = create_engine(l0_database_url, echo=False)
        self.L0Session = sessionmaker(bind=self.l0_engine)
        
        # L3 database connection (write)
        self.l3_engine = create_engine(l3_database_url, echo=False)
        self.L3Session = sessionmaker(bind=self.l3_engine)
        
        # Create L3 tables
        self._create_l3_tables()
        
        # Data quality thresholds
        self.price_outlier_threshold = 3.0  # Z-score threshold
        self.view_outlier_threshold = 3.0   # Z-score threshold
        self.min_price = 0.01              # Minimum valid price
        self.max_price = 10000.0           # Maximum reasonable price
    
    def _create_l3_tables(self):
        """Create L3 database tables if they don't exist."""
        try:
            Base.metadata.create_all(bind=self.l3_engine)
            logger.info("L3 database tables created/verified successfully")
        except SQLAlchemyError as e:
            logger.error(f"Error creating L3 database tables: {e}")
            raise
    
    def _detect_outliers(self, df: pd.DataFrame, column: str, method: str = 'zscore', threshold: float = 3.0) -> pd.Series:
        """
        Detect outliers in a DataFrame column.
        
        Args:
            df: DataFrame
            column: Column name to check for outliers
            method: Method to use ('zscore', 'iqr')
            threshold: Threshold for outlier detection
            
        Returns:
            Boolean Series indicating outliers
        """
        if method == 'zscore':
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            return z_scores > threshold
        
        elif method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return (df[column] < lower_bound) | (df[column] > upper_bound)
        
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
    
    def _clean_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Clean raw data and identify quality issues.
        
        Args:
            df: Raw DataFrame from L0 database
            
        Returns:
            Tuple of (cleaned_df, quality_metrics)
        """
        logger.info(f"Starting data cleaning for {len(df)} records")
        
        quality_metrics = {
            'total_records': len(df),
            'missing_customer_ids': 0,
            'invalid_prices': 0,
            'price_outliers': 0,
            'view_outliers': 0,
            'valid_records': 0
        }
        
        # Make a copy for cleaning
        cleaned_df = df.copy()
        
        # 1. Handle missing customer IDs
        missing_customers = cleaned_df['customer_id'].isnull()
        quality_metrics['missing_customer_ids'] = missing_customers.sum()
        
        # For missing customer IDs, we'll keep them as null but flag them
        cleaned_df['has_customer_id'] = ~missing_customers
        
        # 2. Clean price data
        # Remove negative prices and extreme values
        invalid_prices = (cleaned_df['price'] < self.min_price) | (cleaned_df['price'] > self.max_price)
        quality_metrics['invalid_prices'] = invalid_prices.sum()
        
        # Remove invalid price records
        if invalid_prices.any():
            logger.warning(f"Removing {invalid_prices.sum()} records with invalid prices")
            cleaned_df = cleaned_df[~invalid_prices]
        
        # 3. Detect price outliers
        if len(cleaned_df) > 0:
            price_outliers = self._detect_outliers(cleaned_df, 'price', 'zscore', self.price_outlier_threshold)
            quality_metrics['price_outliers'] = price_outliers.sum()
            cleaned_df['is_price_outlier'] = price_outliers
        
        # 4. Detect view count outliers
        if len(cleaned_df) > 0:
            view_outliers = self._detect_outliers(cleaned_df, 'times_viewed', 'zscore', self.view_outlier_threshold)
            quality_metrics['view_outliers'] = view_outliers.sum()
            cleaned_df['is_view_outlier'] = view_outliers
        
        # 5. Ensure data types
        if len(cleaned_df) > 0:
            cleaned_df['price'] = cleaned_df['price'].astype(float)
            cleaned_df['times_viewed'] = cleaned_df['times_viewed'].astype(int)
            cleaned_df['customer_id'] = cleaned_df['customer_id'].astype('Int64')  # Nullable integer
        
        quality_metrics['valid_records'] = len(cleaned_df)
        
        logger.info(f"Data cleaning completed. Valid records: {quality_metrics['valid_records']}")
        return cleaned_df, quality_metrics
    
    def _aggregate_daily_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate data by date and country.
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            DataFrame with daily aggregated metrics
        """
        logger.info("Aggregating daily metrics by date and country")
        
        if len(df) == 0:
            return pd.DataFrame()
        
        # Group by date and country
        daily_groups = df.groupby(['date', 'country'])
        
        aggregated_data = []
        
        for (date_val, country), group_df in daily_groups:
            # Basic calculations
            total_revenue = group_df['price'].sum()
            total_transactions = len(group_df)
            unique_invoices = group_df['invoice'].nunique()
            unique_customers = group_df['customer_id'].nunique()
            unique_products = group_df['stream_id'].nunique()
            
            # Revenue statistics
            avg_transaction_value = group_df['price'].mean()
            min_transaction_value = group_df['price'].min()
            max_transaction_value = group_df['price'].max()
            median_transaction_value = group_df['price'].median()
            revenue_std_dev = group_df['price'].std()
            
            # Customer metrics
            customers_with_id = group_df[group_df['customer_id'].notna()]
            new_customers = 0  # Would need historical data for proper calculation
            returning_customers = 0  # Would need historical data for proper calculation
            avg_customer_value = customers_with_id.groupby('customer_id')['price'].sum().mean() if len(customers_with_id) > 0 else 0
            
            # Product metrics
            total_views = group_df['times_viewed'].sum()
            avg_views_per_product = group_df.groupby('stream_id')['times_viewed'].sum().mean()
            view_to_purchase_ratio = total_transactions / total_views if total_views > 0 else 0
            
            # Quality metrics
            outlier_transactions = (group_df.get('is_price_outlier', False) | group_df.get('is_view_outlier', False)).sum()
            missing_customer_ids = group_df['customer_id'].isnull().sum()
            data_quality_score = 1.0 - (outlier_transactions + missing_customer_ids) / total_transactions if total_transactions > 0 else 1.0
            
            # Date components
            date_obj = pd.to_datetime(date_val) if isinstance(date_val, str) else date_val
            
            aggregated_record = {
                'date': date_val,
                'year': date_obj.year,
                'month': date_obj.month,
                'day': date_obj.day,
                'day_of_week': date_obj.weekday(),
                'week_of_year': date_obj.isocalendar()[1],
                'country': country,
                'total_revenue': float(total_revenue),
                'avg_transaction_value': float(avg_transaction_value),
                'min_transaction_value': float(min_transaction_value),
                'max_transaction_value': float(max_transaction_value),
                'median_transaction_value': float(median_transaction_value),
                'revenue_std_dev': float(revenue_std_dev) if not pd.isna(revenue_std_dev) else 0.0,
                'total_transactions': int(total_transactions),
                'unique_invoices': int(unique_invoices),
                'unique_customers': int(unique_customers),
                'unique_products': int(unique_products),
                'new_customers': int(new_customers),
                'returning_customers': int(returning_customers),
                'customer_retention_rate': 0.0,  # Would need historical data
                'avg_customer_value': float(avg_customer_value) if not pd.isna(avg_customer_value) else 0.0,
                'total_views': int(total_views),
                'avg_views_per_product': float(avg_views_per_product) if not pd.isna(avg_views_per_product) else 0.0,
                'view_to_purchase_ratio': float(view_to_purchase_ratio),
                'data_quality_score': float(data_quality_score),
                'outlier_transactions': int(outlier_transactions),
                'missing_customer_ids': int(missing_customer_ids),
                'source_record_count': int(total_transactions)
            }
            
            aggregated_data.append(aggregated_record)
        
        result_df = pd.DataFrame(aggregated_data)
        logger.info(f"Generated {len(result_df)} daily aggregated records")
        
        return result_df
    
    def _calculate_customer_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate customer segmentation metrics by date and country.
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            DataFrame with customer segment metrics
        """
        logger.info("Calculating customer segmentation metrics")
        
        if len(df) == 0 or df['customer_id'].isna().all():
            return pd.DataFrame()
        
        # Filter out records without customer IDs for segmentation
        df_with_customers = df[df['customer_id'].notna()].copy()
        
        if len(df_with_customers) == 0:
            return pd.DataFrame()
        
        segment_data = []
        
        # Group by date and country
        for (date_val, country), group_df in df_with_customers.groupby(['date', 'country']):
            # Calculate customer-level metrics
            customer_metrics = group_df.groupby('customer_id').agg({
                'price': ['sum', 'count', 'mean'],
                'times_viewed': 'sum',
                'date': 'max'  # Most recent transaction date for this customer
            }).round(2)
            
            customer_metrics.columns = ['total_spent', 'frequency', 'avg_order_value', 'total_views', 'last_purchase_date']
            
            # Simple segmentation based on spend and frequency
            # In a real scenario, you'd use historical data for proper RFM analysis
            high_value_threshold = customer_metrics['total_spent'].quantile(0.75)
            high_frequency_threshold = customer_metrics['frequency'].quantile(0.75)
            
            # Segment customers
            champions = ((customer_metrics['total_spent'] >= high_value_threshold) & 
                        (customer_metrics['frequency'] >= high_frequency_threshold)).sum()
            
            loyal_customers = (customer_metrics['frequency'] >= high_frequency_threshold).sum() - champions
            
            high_value = (customer_metrics['total_spent'] >= high_value_threshold).sum()
            
            segment_record = {
                'date': date_val,
                'country': country,
                'avg_recency_days': 0.0,  # Would need historical data
                'avg_frequency': float(customer_metrics['frequency'].mean()),
                'avg_monetary_value': float(customer_metrics['total_spent'].mean()),
                'champions_count': int(champions),
                'loyal_customers_count': int(loyal_customers),
                'potential_loyalists_count': 0,  # Would need historical data
                'new_customers_count': len(customer_metrics),  # All customers are "new" in this dataset
                'at_risk_count': 0,  # Would need historical data
                'cannot_lose_count': 0,  # Would need historical data
                'hibernating_count': 0,  # Would need historical data
                'avg_customer_lifetime_value': float(customer_metrics['total_spent'].mean()),
                'customer_acquisition_rate': 0.0,  # Would need historical data
                'customer_churn_risk_score': 0.0,  # Would need historical data
            }
            
            segment_data.append(segment_record)
        
        result_df = pd.DataFrame(segment_data)
        logger.info(f"Generated {len(result_df)} customer segment records")
        
        return result_df
    
    def _calculate_product_performance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate product performance metrics by date, country, and product.
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            DataFrame with product performance metrics
        """
        logger.info("Calculating product performance metrics")
        
        if len(df) == 0:
            return pd.DataFrame()
        
        product_data = []
        
        # Group by date, country, and stream_id
        for (date_val, country, stream_id), group_df in df.groupby(['date', 'country', 'stream_id']):
            revenue = group_df['price'].sum()
            transactions = len(group_df)
            unique_customers = group_df['customer_id'].nunique()
            total_views = group_df['times_viewed'].sum()
            avg_price = group_df['price'].mean()
            
            # Calculate conversion rate (transactions per view)
            conversion_rate = transactions / total_views if total_views > 0 else 0.0
            
            product_record = {
                'date': date_val,
                'country': country,
                'stream_id': stream_id,
                'revenue': float(revenue),
                'transactions': int(transactions),
                'unique_customers': int(unique_customers),
                'total_views': int(total_views),
                'avg_price': float(avg_price),
                'conversion_rate': float(conversion_rate),
                'customer_penetration': 0.0,  # Would need total customers for this date/country
                'revenue_rank': 0,  # Will be calculated later
                'popularity_rank': 0  # Will be calculated later
            }
            
            product_data.append(product_record)
        
        if not product_data:
            return pd.DataFrame()
        
        result_df = pd.DataFrame(product_data)
        
        # Calculate rankings within each date/country combination
        for (date_val, country), group_df in result_df.groupby(['date', 'country']):
            # Revenue rank (1 = highest revenue)
            revenue_ranks = group_df['revenue'].rank(method='dense', ascending=False)
            # Popularity rank (1 = most transactions)
            popularity_ranks = group_df['transactions'].rank(method='dense', ascending=False)
            
            # Update rankings in the main dataframe
            mask = (result_df['date'] == date_val) & (result_df['country'] == country)
            result_df.loc[mask, 'revenue_rank'] = revenue_ranks.astype(int)
            result_df.loc[mask, 'popularity_rank'] = popularity_ranks.astype(int)
        
        logger.info(f"Generated {len(result_df)} product performance records")
        
        return result_df
    
    def _generate_quality_report(self, df: pd.DataFrame, quality_metrics: Dict[str, int]) -> pd.DataFrame:
        """
        Generate data quality report.
        
        Args:
            df: Cleaned DataFrame
            quality_metrics: Quality metrics from cleaning process
            
        Returns:
            DataFrame with quality report data
        """
        logger.info("Generating data quality report")
        
        quality_data = []
        
        if len(df) == 0:
            # Create a minimal report for empty data
            quality_record = {
                'report_date': datetime.now().date(),
                'country': None,
                'total_records': quality_metrics.get('total_records', 0),
                'valid_records': 0,
                'missing_customer_ids': quality_metrics.get('missing_customer_ids', 0),
                'invalid_prices': quality_metrics.get('invalid_prices', 0),
                'invalid_dates': 0,
                'completeness_score': 0.0,
                'consistency_score': 0.0,
                'accuracy_score': 0.0,
                'overall_quality_score': 0.0,
                'revenue_anomalies': 0,
                'volume_anomalies': 0,
                'price_outliers': quality_metrics.get('price_outliers', 0),
                'view_count_outliers': quality_metrics.get('view_outliers', 0),
                'revenue_z_score_max': 0.0,
                'price_iqr_outliers': 0
            }
            quality_data.append(quality_record)
        else:
            # Generate reports by country and overall
            countries = df['country'].unique()
            
            for country in list(countries) + [None]:  # Include overall report (None)
                if country is None:
                    country_df = df
                    country_name = None
                else:
                    country_df = df[df['country'] == country]
                    country_name = country
                
                if len(country_df) == 0:
                    continue
                
                # Calculate quality scores
                total_records = len(country_df)
                missing_customers = country_df['customer_id'].isnull().sum()
                
                completeness_score = 1.0 - (missing_customers / total_records) if total_records > 0 else 1.0
                consistency_score = 1.0  # Would need more complex logic for real consistency checks
                accuracy_score = 1.0 - (quality_metrics.get('invalid_prices', 0) / quality_metrics.get('total_records', 1))
                overall_quality_score = (completeness_score + consistency_score + accuracy_score) / 3
                
                # Detect anomalies
                price_outliers = country_df.get('is_price_outlier', pd.Series([False] * len(country_df))).sum()
                view_outliers = country_df.get('is_view_outlier', pd.Series([False] * len(country_df))).sum()
                
                # Calculate Z-scores for revenue by day
                daily_revenue = country_df.groupby('date')['price'].sum()
                if len(daily_revenue) > 1:
                    revenue_z_scores = np.abs((daily_revenue - daily_revenue.mean()) / daily_revenue.std())
                    revenue_z_score_max = revenue_z_scores.max()
                    revenue_anomalies = (revenue_z_scores > 2.0).sum()
                else:
                    revenue_z_score_max = 0.0
                    revenue_anomalies = 0
                
                quality_record = {
                    'report_date': datetime.now().date(),
                    'country': country_name,
                    'total_records': int(total_records),
                    'valid_records': int(total_records),  # After cleaning
                    'missing_customer_ids': int(missing_customers),
                    'invalid_prices': quality_metrics.get('invalid_prices', 0),
                    'invalid_dates': 0,  # Would need date validation logic
                    'completeness_score': float(completeness_score),
                    'consistency_score': float(consistency_score),
                    'accuracy_score': float(accuracy_score),
                    'overall_quality_score': float(overall_quality_score),
                    'revenue_anomalies': int(revenue_anomalies),
                    'volume_anomalies': 0,  # Would need volume anomaly detection
                    'price_outliers': int(price_outliers),
                    'view_count_outliers': int(view_outliers),
                    'revenue_z_score_max': float(revenue_z_score_max),
                    'price_iqr_outliers': int(self._detect_outliers(country_df, 'price', 'iqr').sum())
                }
                
                quality_data.append(quality_record)
        
        result_df = pd.DataFrame(quality_data)
        logger.info(f"Generated {len(result_df)} quality report records")
        
        return result_df
    
    def _save_to_l3_database(self, daily_metrics_df: pd.DataFrame, 
                            customer_segments_df: pd.DataFrame,
                            product_performance_df: pd.DataFrame,
                            quality_report_df: pd.DataFrame):
        """
        Save all processed data to L3 database.
        
        Args:
            daily_metrics_df: Daily aggregated metrics
            customer_segments_df: Customer segmentation data
            product_performance_df: Product performance data
            quality_report_df: Data quality reports
        """
        logger.info("Saving processed data to L3 database")
        
        # Add timestamp fields that are required by the database models
        from datetime import datetime
        current_time = datetime.now()
        
        with self.L3Session() as session:
            try:
                # Save daily metrics (has both created_at and updated_at)
                if len(daily_metrics_df) > 0:
                    daily_metrics_df = daily_metrics_df.copy()
                    daily_metrics_df['created_at'] = current_time
                    daily_metrics_df['updated_at'] = current_time
                    
                    daily_metrics_df.to_sql('daily_aggregated_metrics', self.l3_engine, 
                                          if_exists='append', index=False, method='multi')
                    logger.info(f"Saved {len(daily_metrics_df)} daily metrics records")
                
                # Save customer segments (only has created_at)
                if len(customer_segments_df) > 0:
                    customer_segments_df = customer_segments_df.copy()
                    customer_segments_df['created_at'] = current_time
                    
                    customer_segments_df.to_sql('customer_segment_metrics', self.l3_engine, 
                                              if_exists='append', index=False, method='multi')
                    logger.info(f"Saved {len(customer_segments_df)} customer segment records")
                
                # Save product performance (only has created_at)
                if len(product_performance_df) > 0:
                    product_performance_df = product_performance_df.copy()
                    product_performance_df['created_at'] = current_time
                    
                    product_performance_df.to_sql('product_performance_metrics', self.l3_engine, 
                                                if_exists='append', index=False, method='multi')
                    logger.info(f"Saved {len(product_performance_df)} product performance records")
                
                # Save quality reports (only has created_at)
                if len(quality_report_df) > 0:
                    quality_report_df = quality_report_df.copy()
                    quality_report_df['created_at'] = current_time
                    
                    quality_report_df.to_sql('data_quality_reports', self.l3_engine, 
                                           if_exists='append', index=False, method='multi')
                    logger.info(f"Saved {len(quality_report_df)} quality report records")
                
                session.commit()
                logger.info("All data successfully saved to L3 database")
                
            except Exception as e:
                session.rollback()
                logger.error(f"Error saving to L3 database: {e}")
                raise
    
    def process_date_range(self, start_date: date = None, end_date: date = None) -> Dict[str, Any]:
        """
        Process data for a specific date range.
        
        Args:
            start_date: Start date (inclusive). If None, processes all data.
            end_date: End date (inclusive). If None, uses start_date or all data.
            
        Returns:
            Dictionary with processing results
        """
        start_time = datetime.utcnow()
        
        logger.info(f"Starting data cleaning pipeline for date range: {start_date} to {end_date}")
        
        try:
            # Load data from L0 database
            with self.L0Session() as l0_session:
                query = l0_session.query(InvoiceRecord)
                
                if start_date:
                    query = query.filter(InvoiceRecord.date_field >= start_date)
                if end_date:
                    query = query.filter(InvoiceRecord.date_field <= end_date)
                
                # Convert to DataFrame
                records = query.all()
                
                if not records:
                    logger.warning("No records found in L0 database for the specified date range")
                    return {
                        'status': 'no_data',
                        'records_processed': 0,
                        'processing_time': 0,
                        'quality_score': 0.0
                    }
                
                # Convert to DataFrame
                data = []
                for record in records:
                    data.append({
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
                        'source_file': record.source_file
                    })
                
                df = pd.DataFrame(data)
                logger.info(f"Loaded {len(df)} records from L0 database")
            
            # Clean the data
            cleaned_df, quality_metrics = self._clean_data(df)
            
            # Generate aggregated datasets
            daily_metrics_df = self._aggregate_daily_metrics(cleaned_df)
            customer_segments_df = self._calculate_customer_segments(cleaned_df)
            product_performance_df = self._calculate_product_performance(cleaned_df)
            quality_report_df = self._generate_quality_report(cleaned_df, quality_metrics)
            
            # Save to L3 database
            self._save_to_l3_database(
                daily_metrics_df,
                customer_segments_df,
                product_performance_df,
                quality_report_df
            )
            
            # Calculate processing statistics
            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds()
            
            overall_quality_score = quality_metrics['valid_records'] / quality_metrics['total_records'] if quality_metrics['total_records'] > 0 else 0.0
            
            results = {
                'status': 'completed',
                'records_processed': quality_metrics['valid_records'],
                'total_input_records': quality_metrics['total_records'],
                'daily_metrics_generated': len(daily_metrics_df),
                'customer_segments_generated': len(customer_segments_df),
                'product_metrics_generated': len(product_performance_df),
                'quality_reports_generated': len(quality_report_df),
                'processing_time': processing_time,
                'quality_score': overall_quality_score,
                'data_quality_issues': {
                    'missing_customer_ids': quality_metrics['missing_customer_ids'],
                    'invalid_prices': quality_metrics['invalid_prices'],
                    'price_outliers': quality_metrics['price_outliers'],
                    'view_outliers': quality_metrics['view_outliers']
                }
            }
            
            logger.info(f"Data cleaning pipeline completed successfully in {processing_time:.2f} seconds")
            logger.info(f"Processed {results['records_processed']} records with {results['quality_score']:.3f} quality score")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in data cleaning pipeline: {e}")
            raise
    
    def get_l3_database_stats(self) -> Dict[str, Any]:
        """
        Get statistics about L3 database content.
        
        Returns:
            Dictionary with L3 database statistics
        """
        with self.L3Session() as session:
            stats = {}
            
            # Daily metrics stats
            daily_count = session.query(DailyAggregatedMetrics).count()
            if daily_count > 0:
                daily_revenue = session.query(func.sum(DailyAggregatedMetrics.total_revenue)).scalar()
                daily_date_range = session.query(
                    func.min(DailyAggregatedMetrics.date),
                    func.max(DailyAggregatedMetrics.date)
                ).first()
                
                stats['daily_metrics'] = {
                    'record_count': daily_count,
                    'total_revenue': float(daily_revenue) if daily_revenue else 0.0,
                    'date_range': {
                        'min_date': daily_date_range[0],
                        'max_date': daily_date_range[1]
                    }
                }
            
            # Customer segment stats
            customer_count = session.query(CustomerSegmentMetrics).count()
            stats['customer_segments'] = {'record_count': customer_count}
            
            # Product performance stats
            product_count = session.query(ProductPerformanceMetrics).count()
            stats['product_performance'] = {'record_count': product_count}
            
            # Quality report stats
            quality_count = session.query(DataQualityReport).count()
            if quality_count > 0:
                avg_quality = session.query(func.avg(DataQualityReport.overall_quality_score)).scalar()
                stats['quality_reports'] = {
                    'record_count': quality_count,
                    'average_quality_score': float(avg_quality) if avg_quality else 0.0
                }
            
            return stats


def main():
    """Main function for command-line usage."""
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description='Data Cleaning and Aggregation Pipeline')
    parser.add_argument('--l0-database', default=L0_DATABASE_URL, help='L0 database URL')
    parser.add_argument('--l3-database', default=L3_DATABASE_URL, help='L3 database URL')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    parser.add_argument('--stats', action='store_true', help='Show L3 database statistics')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = DataCleaningPipeline(
        l0_database_url=args.l0_database,
        l3_database_url=args.l3_database
    )
    
    if args.stats:
        stats = pipeline.get_l3_database_stats()
        print("\nL3 Database Statistics:")
        print("=" * 50)
        for category, metrics in stats.items():
            print(f"\n{category.upper()}:")
            for key, value in metrics.items():
                print(f"  {key}: {value}")
        return
    
    # Parse dates
    start_date = None
    end_date = None
    
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date()
    
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d').date()
    
    # Run pipeline
    results = pipeline.process_date_range(start_date, end_date)
    
    print("\nData Cleaning Pipeline Results:")
    print("=" * 50)
    for key, value in results.items():
        if key != 'data_quality_issues':
            print(f"{key}: {value}")
    
    if 'data_quality_issues' in results:
        print(f"\nData Quality Issues:")
        for issue, count in results['data_quality_issues'].items():
            print(f"  {issue}: {count}")


if __name__ == "__main__":
    main()
