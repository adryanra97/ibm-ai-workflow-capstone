"""
Data service for accessing L0 and L3 databases

Author: Adryan R A
"""

import sqlite3
import logging
import asyncio
from datetime import date, datetime, timedelta
from typing import Optional, Dict, List, Any
import pandas as pd
import numpy as np

from utils.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class DataService:
    """
    Service for accessing and preparing data from L0 and L3 databases
    """
    
    def __init__(self):
        self.l0_db_path = settings.l0_database_path
        self.l3_db_path = settings.l3_database_path
        self._connection_pool = {}
        
    async def initialize(self):
        """Initialize data service"""
        logger.info("Initializing Data Service")
        
        # Test database connections
        l0_status = await self.check_connection(db_type="l0")
        l3_status = await self.check_connection(db_type="l3")
        
        if l0_status != "healthy" or l3_status != "healthy":
            logger.warning("Database connectivity issues detected")
        
        logger.info("Data Service initialized")
    
    async def cleanup(self):
        """Cleanup database connections"""
        for conn in self._connection_pool.values():
            try:
                conn.close()
            except:
                pass
        self._connection_pool.clear()
        logger.info("Data Service cleanup completed")
    
    async def check_connection(self, db_type: str = "both") -> str:
        """
        Check database connectivity
        
        Args:
            db_type: "l0", "l3", or "both"
            
        Returns:
            Connection status
        """
        try:
            if db_type in ["l0", "both"]:
                await self._execute_query(
                    "SELECT COUNT(*) FROM invoice_records LIMIT 1",
                    db_type="l0"
                )
            
            if db_type in ["l3", "both"]:
                await self._execute_query(
                    "SELECT COUNT(*) FROM daily_aggregated_metrics LIMIT 1",
                    db_type="l3"
                )
            
            return "healthy"
            
        except Exception as e:
            logger.error(f"Database connection check failed: {e}")
            return "unhealthy"
    
    async def get_available_countries(self) -> List[str]:
        """
        Get list of countries with available data
        
        Returns:
            List of country names
        """
        try:
            query = """
            SELECT DISTINCT country 
            FROM daily_aggregated_metrics 
            ORDER BY country
            """
            
            result = await self._execute_query(query, db_type="l3")
            countries = [row[0] for row in result]
            
            # Filter to supported countries
            supported_countries = [
                country for country in countries 
                if country in settings.supported_countries
            ]
            
            return supported_countries
            
        except Exception as e:
            logger.error(f"Failed to get available countries: {e}")
            return []
    
    async def get_training_data(self, 
                              country: str,
                              start_date: Optional[date] = None,
                              end_date: Optional[date] = None) -> Optional[pd.DataFrame]:
        """
        Get training data for time series models
        
        Args:
            country: Country name
            start_date: Start date for training data
            end_date: End date for training data
            
        Returns:
            DataFrame with time series data or None
        """
        try:
            # Build query conditions
            conditions = ["country = ?"]
            params = [country]
            
            if start_date:
                conditions.append("date >= ?")
                params.append(start_date.isoformat())
            
            if end_date:
                conditions.append("date <= ?")
                params.append(end_date.isoformat())
            
            where_clause = " AND ".join(conditions)
            
            # Query aggregated daily data from L3 database
            query = f"""
            SELECT 
                date,
                total_revenue as y,
                total_transactions,
                unique_customers,
                avg_transaction_value,
                day_of_week,
                week_of_year,
                month,
                year
            FROM daily_aggregated_metrics
            WHERE {where_clause}
            ORDER BY date
            """
            
            result = await self._execute_query(query, params, db_type="l3")
            
            if not result:
                logger.warning(f"No training data found for {country}")
                return None
            
            # Convert to DataFrame
            columns = [
                'ds', 'y', 'transactions', 'customers', 'avg_order_value',
                'day_of_week', 'week_of_year', 'month', 'year'
            ]
            
            df = pd.DataFrame(result, columns=columns)
            
            # Convert date column to datetime
            df['ds'] = pd.to_datetime(df['ds'])
            
            # Add additional features for modeling
            df = await self._engineer_features(df)
            
            # Validate data quality
            df = await self._validate_training_data(df, country)
            
            logger.info(f"Retrieved {len(df)} days of training data for {country}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to get training data for {country}: {e}")
            return None
    
    async def get_historical_data(self,
                                country: str,
                                end_date: date,
                                lookback_days: int = 90) -> Optional[pd.DataFrame]:
        """
        Get historical data for context in predictions
        
        Args:
            country: Country name
            end_date: End date for historical data
            lookback_days: Number of days to look back
            
        Returns:
            DataFrame with historical data
        """
        try:
            start_date = end_date - timedelta(days=lookback_days)
            
            query = """
            SELECT 
                date,
                total_revenue,
                total_transactions,
                unique_customers,
                avg_transaction_value
            FROM daily_aggregated_metrics
            WHERE country = ? AND date >= ? AND date <= ?
            ORDER BY date
            """
            
            params = [country, start_date.isoformat(), end_date.isoformat()]
            result = await self._execute_query(query, params, db_type="l3")
            
            if not result:
                return None
            
            columns = ['date', 'revenue', 'transactions', 'customers', 'avg_order_value']
            df = pd.DataFrame(result, columns=columns)
            df['date'] = pd.to_datetime(df['date'])
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get historical data: {e}")
            return None
    
    async def get_data_summary(self, country: Optional[str] = None) -> Dict[str, Any]:
        """
        Get data summary statistics
        
        Args:
            country: Optional country filter
            
        Returns:
            Data summary dictionary
        """
        try:
            # Base query for summary stats
            base_query = """
            SELECT 
                COUNT(*) as total_days,
                MIN(date) as start_date,
                MAX(date) as end_date,
                SUM(total_revenue) as total_revenue,
                SUM(total_transactions) as total_transactions,
                COUNT(DISTINCT country) as countries_count
            FROM daily_aggregated_metrics
            """
            
            params = []
            if country:
                base_query += " WHERE country = ?"
                params.append(country)
            
            result = await self._execute_query(base_query, params, db_type="l3")
            
            if result:
                summary = {
                    'total_days': result[0][0],
                    'date_range': {
                        'start': result[0][1],
                        'end': result[0][2]
                    },
                    'total_revenue': float(result[0][3] or 0),
                    'total_transactions': int(result[0][4] or 0),
                    'countries_count': int(result[0][5] or 0)
                }
                
                # Add country-specific stats if no country filter
                if not country:
                    country_stats = await self._get_country_stats()
                    summary['country_breakdown'] = country_stats
                
                return summary
            
            return {}
            
        except Exception as e:
            logger.error(f"Failed to get data summary: {e}")
            return {}
    
    async def get_business_metrics(self, 
                                 country: str,
                                 start_date: date,
                                 end_date: date) -> Dict[str, Any]:
        """
        Get business metrics for performance correlation analysis
        
        Args:
            country: Country name
            start_date: Start date
            end_date: End date
            
        Returns:
            Business metrics dictionary
        """
        try:
            query = """
            SELECT 
                AVG(total_revenue) as avg_daily_revenue,
                AVG(total_transactions) as avg_daily_transactions,
                AVG(unique_customers) as avg_daily_customers,
                AVG(avg_transaction_value) as avg_order_value,
                SUM(total_revenue) as total_revenue,
                COUNT(*) as days_count,
                MAX(total_revenue) as peak_revenue,
                MIN(total_revenue) as min_revenue,
                STDDEV(total_revenue) as revenue_volatility
            FROM daily_aggregated_metrics
            WHERE country = ? AND date >= ? AND date <= ?
            """
            
            params = [country, start_date.isoformat(), end_date.isoformat()]
            result = await self._execute_query(query, params, db_type="l3")
            
            if result and result[0][0] is not None:
                return {
                    'avg_daily_revenue': float(result[0][0]),
                    'avg_daily_transactions': float(result[0][1] or 0),
                    'avg_daily_customers': float(result[0][2] or 0),
                    'avg_order_value': float(result[0][3] or 0),
                    'total_revenue': float(result[0][4]),
                    'days_count': int(result[0][5]),
                    'peak_revenue': float(result[0][6]),
                    'min_revenue': float(result[0][7]),
                    'revenue_volatility': float(result[0][8] or 0)
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Failed to get business metrics: {e}")
            return {}
    
    # Private methods
    
    async def _execute_query(self, 
                           query: str, 
                           params: Optional[List] = None,
                           db_type: str = "l3") -> List[tuple]:
        """
        Execute database query
        
        Args:
            query: SQL query
            params: Query parameters
            db_type: Database type ("l0" or "l3")
            
        Returns:
            Query results
        """
        db_path = self.l0_db_path if db_type == "l0" else self.l3_db_path
        
        # Use asyncio to run database operations in thread pool
        def _sync_execute():
            conn = sqlite3.connect(db_path)
            try:
                cursor = conn.cursor()
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                return cursor.fetchall()
            finally:
                conn.close()
        
        return await asyncio.get_event_loop().run_in_executor(None, _sync_execute)
    
    async def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer additional features for time series modeling
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional features
        """
        try:
            # Sort by date
            df = df.sort_values('ds').reset_index(drop=True)
            
            # Lag features
            for lag in [1, 7, 14, 30]:
                df[f'y_lag_{lag}'] = df['y'].shift(lag)
            
            # Rolling averages
            for window in [7, 14, 30]:
                df[f'y_rolling_mean_{window}'] = df['y'].rolling(window=window, min_periods=1).mean()
                df[f'y_rolling_std_{window}'] = df['y'].rolling(window=window, min_periods=1).std()
            
            # Growth rates
            df['y_growth_1d'] = df['y'].pct_change(1)
            df['y_growth_7d'] = df['y'].pct_change(7)
            
            # Seasonal features
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            df['is_month_start'] = (df['ds'].dt.day <= 3).astype(int)
            df['is_month_end'] = (df['ds'].dt.day >= 28).astype(int)
            
            # Handle any infinite or NaN values
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.fillna(method='bfill').fillna(0)
            
            return df
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            return df
    
    async def _validate_training_data(self, df: pd.DataFrame, country: str) -> pd.DataFrame:
        """
        Validate and clean training data
        
        Args:
            df: Input DataFrame
            country: Country name for logging
            
        Returns:
            Cleaned DataFrame
        """
        try:
            initial_count = len(df)
            
            # Remove rows with missing target values
            df = df.dropna(subset=['y'])
            
            # Remove rows with negative revenue (data quality issue)
            df = df[df['y'] >= 0]
            
            # Remove extreme outliers (beyond 5 standard deviations)
            if len(df) > 10:
                mean_revenue = df['y'].mean()
                std_revenue = df['y'].std()
                outlier_threshold = mean_revenue + 5 * std_revenue
                
                outliers_removed = len(df[df['y'] > outlier_threshold])
                df = df[df['y'] <= outlier_threshold]
                
                if outliers_removed > 0:
                    logger.info(f"Removed {outliers_removed} outliers for {country}")
            
            # Ensure minimum data requirements
            if len(df) < settings.min_training_days:
                logger.warning(f"Insufficient training data for {country}: {len(df)} days")
                return pd.DataFrame()  # Return empty DataFrame
            
            final_count = len(df)
            if final_count < initial_count:
                logger.info(f"Data validation for {country}: {initial_count} -> {final_count} rows")
            
            return df.reset_index(drop=True)
            
        except Exception as e:
            logger.error(f"Data validation failed for {country}: {e}")
            return df
    
    async def _get_country_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics by country"""
        try:
            query = """
            SELECT 
                country,
                COUNT(*) as days,
                SUM(total_revenue) as revenue,
                AVG(total_revenue) as avg_daily_revenue,
                MIN(date) as start_date,
                MAX(date) as end_date
            FROM daily_aggregated_metrics
            GROUP BY country
            ORDER BY revenue DESC
            """
            
            result = await self._execute_query(query, db_type="l3")
            
            country_stats = {}
            for row in result:
                country_stats[row[0]] = {
                    'days': row[1],
                    'total_revenue': float(row[2]),
                    'avg_daily_revenue': float(row[3]),
                    'date_range': f"{row[4]} to {row[5]}"
                }
            
            return country_stats
            
        except Exception as e:
            logger.error(f"Failed to get country stats: {e}")
            return {}
