#!/usr/bin/env python3
"""
Test script to demonstrate the L0 -> L3 data cleaning pipeline.

This script shows how the data cleaning pipeline processes raw invoice data
from the L0 database (file.db) and creates cleaned, aggregated analytics data
in the L3 database (analytics.db).

Author: Adryan R A
"""

import sqlite3
from datetime import date
from data_cleaning import DataCleaningPipeline

def test_databases():
    """Check both L0 and L3 databases"""
    print("=" * 60)
    print("DATA CLEANING PIPELINE VERIFICATION")
    print("=" * 60)
    
    # Check L0 database (raw data)
    print("\nL0 DATABASE (Raw Invoice Data):")
    with sqlite3.connect('file.db') as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM invoice_records")
        l0_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT MIN(date_field), MAX(date_field) FROM invoice_records")
        l0_date_range = cursor.fetchone()
        
        print(f"Total Records: {l0_count:,}")
        print(f"Date Range: {l0_date_range[0]} to {l0_date_range[1]}")
    
    # Check L3 database (analytics data)
    print("\nL3 DATABASE (Analytics Data):")
    with sqlite3.connect('analytics.db') as conn:
        cursor = conn.cursor()
        
        # Daily metrics
        cursor.execute("SELECT COUNT(*) FROM daily_aggregated_metrics")
        daily_count = cursor.fetchone()[0]
        
        # Customer segments  
        cursor.execute("SELECT COUNT(*) FROM customer_segment_metrics")
        customer_count = cursor.fetchone()[0]
        
        # Product performance
        cursor.execute("SELECT COUNT(*) FROM product_performance_metrics")
        product_count = cursor.fetchone()[0]
        
        # Quality reports
        cursor.execute("SELECT COUNT(*) FROM data_quality_reports")
        quality_count = cursor.fetchone()[0]
        
        # Summary stats
        cursor.execute("""
            SELECT 
                MIN(date), MAX(date), 
                COUNT(DISTINCT country),
                printf('%.2f', SUM(total_revenue)),
                SUM(total_transactions)
            FROM daily_aggregated_metrics
        """)
        stats = cursor.fetchone()
        
        print(f"Daily Metrics: {daily_count:,} records")
        print(f"Customer Segments: {customer_count:,} records") 
        print(f"Product Performance: {product_count:,} records")
        print(f"Quality Reports: {quality_count:,} records")
        print(f"Date Coverage: {stats[0]} to {stats[1]}")
        print(f"Countries: {stats[2]}")
        print(f"Total Revenue: ${stats[3]}")
        print(f"Total Transactions: {stats[4]:,}")

def test_sample_query():
    """Run a sample analytics query"""
    print("\nSAMPLE ANALYTICS QUERY:")
    print("Top 5 countries by revenue:")
    
    with sqlite3.connect('analytics.db') as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                country,
                printf('%.2f', SUM(total_revenue)) as total_revenue,
                SUM(total_transactions) as transactions,
                AVG(unique_customers) as avg_customers
            FROM daily_aggregated_metrics 
            GROUP BY country 
            ORDER BY SUM(total_revenue) DESC 
            LIMIT 5
        """)
        
        results = cursor.fetchall()
        print(f"{'Country':<20} {'Revenue':<12} {'Transactions':<12} {'Avg Customers'}")
        print("-" * 60)
        for row in results:
            print(f"{row[0]:<20} ${row[1]:<11} {row[2]:<12} {row[3]:.1f}")

if __name__ == "__main__":
    test_databases()
    test_sample_query()
    
    print("\nDATA CLEANING PIPELINE VERIFICATION COMPLETE!")
    print("\nDatabase Files:")
    print("L0 Database: file.db (Raw invoice data)")
    print("L3 Database: analytics.db (Cleaned analytics data)")
    print("\nPipeline Components:")
    print("data_ingestion.py - Loads JSON files into L0 database") 
    print("data_cleaning.py - Processes L0 data into L3 analytics database")
