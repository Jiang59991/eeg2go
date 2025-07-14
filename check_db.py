import sqlite3

# 连接数据库
conn = sqlite3.connect('database/eeg2go.db')
cursor = conn.cursor()

# 检查表
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print("Tables:", [t[0] for t in tables])

# 检查datasets表
try:
    cursor.execute("SELECT COUNT(*) FROM datasets")
    datasets_count = cursor.fetchone()[0]
    print(f"Datasets count: {datasets_count}")
    
    if datasets_count > 0:
        cursor.execute("SELECT id, name FROM datasets LIMIT 5")
        datasets = cursor.fetchall()
        print("Sample datasets:", datasets)
except Exception as e:
    print(f"Error checking datasets: {e}")

# 检查feature_sets表
try:
    cursor.execute("SELECT COUNT(*) FROM feature_sets")
    feature_sets_count = cursor.fetchone()[0]
    print(f"Feature sets count: {feature_sets_count}")
    
    if feature_sets_count > 0:
        cursor.execute("SELECT id, name FROM feature_sets LIMIT 5")
        feature_sets = cursor.fetchall()
        print("Sample feature sets:", feature_sets)
except Exception as e:
    print(f"Error checking feature_sets: {e}")

conn.close() 