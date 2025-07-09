#!/usr/bin/env python3
"""
Database Table Viewer Script
View the first 5 records of all tables in the database
"""

import os
import sqlite3
import sys
from pathlib import Path

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
DATABASE_PATH = PROJECT_ROOT / "database" / "eeg2go.db"

def get_all_tables(cursor):
    """Get all table names from database"""
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' 
        AND name NOT LIKE 'sqlite_%'
        ORDER BY name
    """)
    return [row[0] for row in cursor.fetchall()]

def get_table_info(cursor, table_name):
    """Get table column information"""
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    return [col[1] for col in columns]  # Return column name list

def get_table_count(cursor, table_name):
    """Get table record count"""
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    return cursor.fetchone()[0]

def view_table_data(cursor, table_name, limit=5):
    """View first N records of table"""
    try:
        # Get column information
        columns = get_table_info(cursor, table_name)
        
        # Get record count
        count = get_table_count(cursor, table_name)
        
        # Get first N records
        cursor.execute(f"SELECT * FROM {table_name} LIMIT {limit}")
        rows = cursor.fetchall()
        
        return {
            'columns': columns,
            'count': count,
            'rows': rows
        }
    except Exception as e:
        return {
            'error': str(e),
            'columns': [],
            'count': 0,
            'rows': []
        }

def print_table_data(table_name, table_info, output_file):
    """Print table data to file"""
    output_file.write(f"\n{'='*80}\n")
    output_file.write(f"Table: {table_name}\n")
    output_file.write(f"{'='*80}\n")
    
    if 'error' in table_info:
        output_file.write(f"ERROR: {table_info['error']}\n")
        return
    
    output_file.write(f"Total Records: {table_info['count']}\n")
    output_file.write(f"Columns: {len(table_info['columns'])}\n")
    
    if table_info['count'] == 0:
        output_file.write("Table is empty, no data\n")
        return
    
    # Print column names
    columns = table_info['columns']
    output_file.write(f"\nColumn Names: {', '.join(columns)}\n")
    
    # Print data
    output_file.write(f"\nFirst 5 Records:\n")
    output_file.write("-" * 80 + "\n")
    
    for i, row in enumerate(table_info['rows'], 1):
        output_file.write(f"Record {i}:\n")
        for j, (col_name, value) in enumerate(zip(columns, row)):
            # Limit display length to avoid long output
            if value is None:
                display_value = "NULL"
            elif isinstance(value, str) and len(str(value)) > 50:
                display_value = str(value)[:47] + "..."
            else:
                display_value = str(value)
            
            output_file.write(f"  {col_name}: {display_value}\n")
        output_file.write("\n")

def main():
    """Main function"""
    # Generate output filename
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"database_tables_view_{timestamp}.txt"
    output_path = PROJECT_ROOT / "database" / output_filename
    
    print("Database Table Viewer Tool")
    print(f"Database Path: {DATABASE_PATH}")
    print(f"Output File: {output_path}")
    print(f"{'='*80}")
    
    # Check if database file exists
    if not DATABASE_PATH.exists():
        print(f"ERROR: Database file not found: {DATABASE_PATH}")
        print("Please ensure the database file has been created")
        sys.exit(1)
    
    try:
        # Connect to database
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Get all table names
        tables = get_all_tables(cursor)
        
        if not tables:
            print("ERROR: No tables found in database")
            sys.exit(1)
        
        print(f"SUCCESS: Found {len(tables)} tables:")
        for table in tables:
            print(f"  - {table}")
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("Database Table View Report\n")
            f.write(f"Database Path: {DATABASE_PATH}\n")
            f.write(f"Generated Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Tables: {len(tables)}\n")
            f.write(f"{'='*80}\n")
            
            f.write(f"\nTable List:\n")
            for table in tables:
                f.write(f"  - {table}\n")
            
            # View data for each table
            for table_name in tables:
                table_info = view_table_data(cursor, table_name, limit=5)
                print_table_data(table_name, table_info, f)
            
            f.write(f"\nDatabase view completed!\n")
        
        conn.close()
        print(f"\nSUCCESS: Database view completed! Results saved to: {output_path}")
        
    except sqlite3.Error as e:
        print(f"ERROR: Database error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Program error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 