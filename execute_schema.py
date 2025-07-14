import sqlite3

# 连接数据库
conn = sqlite3.connect('database/eeg2go.db')
cursor = conn.cursor()

# 读取并执行SQL文件
with open('database/feature_extraction_schema.sql', 'r', encoding='utf-8') as f:
    sql_script = f.read()
    cursor.executescript(sql_script)

conn.commit()
conn.close()

print('Feature extraction schema executed successfully') 