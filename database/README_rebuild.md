# 数据库表格重建工具

这个目录包含了用于重建experiment相关表格和tasks表格的工具。

## 文件说明

- `rebuild_experiment_tables.sql` - SQL脚本，包含重建表格的完整SQL语句
- `rebuild_experiment_tables.py` - Python脚本，用于执行SQL重建脚本
- `rebuild_experiment_tables.sh` - Shell脚本，方便执行Python重建脚本

## 重建的表格

此工具会重建以下表格：

1. **experiment_definitions** - 实验定义表
2. **experiment_results** - 实验结果表
3. **experiment_metadata** - 实验元数据表
4. **experiment_feature_results** - 实验特征结果表
5. **tasks** - 任务队列表

## 使用方法

### 方法1: 使用Shell脚本（推荐）

```bash
# 在项目根目录执行
./rebuild_experiment_tables.sh
```

### 方法2: 直接使用Python脚本

```bash
# 在项目根目录执行
python3 database/rebuild_experiment_tables.py
```

### 方法3: 直接执行SQL脚本

```bash
# 在项目根目录执行
sqlite3 database/eeg2go.db < database/rebuild_experiment_tables.sql
```

## 注意事项

⚠️ **重要警告**: 此操作会删除现有的experiment和tasks数据！

- 重建过程会删除所有现有的experiment相关数据和tasks数据
- 会重新创建表格结构和索引
- 会插入默认的experiment定义
- 会重新创建相关的视图

## 环境变量

可以通过设置环境变量来指定数据库路径：

```bash
export DATABASE_PATH="your/custom/database/path.db"
./rebuild_experiment_tables.sh
```

默认数据库路径为: `database/eeg2go.db`

## 验证重建结果

重建完成后，脚本会自动验证：

1. 检查所有表格是否成功创建
2. 检查所有视图是否成功创建
3. 检查默认数据是否插入成功
4. 显示各表的记录数量

## 故障排除

如果重建失败，请检查：

1. 数据库文件是否有写权限
2. Python环境是否正确
3. 数据库是否被其他进程锁定
4. 磁盘空间是否充足

## 重建后的功能

重建完成后，以下功能将恢复正常：

- 实验管理功能
- 任务队列功能
- 特征提取任务
- 实验分析任务
- 任务状态跟踪
- 结果查询和展示
