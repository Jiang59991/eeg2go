# BIDS (Brain Imaging Data Structure) 数据集导入指南

## 什么是BIDS？

BIDS (Brain Imaging Data Structure) 是一个标准化的神经影像数据组织格式，旨在促进神经影像数据的共享和分析。BIDS标准由神经影像社区开发，目前版本为v1.4.0。

### BIDS的主要特点

- **标准化**: 提供一致的数据组织结构
- **可扩展**: 支持多种神经影像模态
- **元数据丰富**: 包含详细的数据描述信息
- **工具兼容**: 与多种神经影像分析工具兼容
- **社区支持**: 活跃的开发和维护社区

## BIDS目录结构

```
dataset/
├── dataset_description.json          # 数据集描述文件
├── participants.tsv                  # 参与者信息
├── sub-001/                         # 主题1
│   ├── sub-001_sessions.tsv         # 会话信息（可选）
│   ├── ses-001/                     # 会话1
│   │   ├── eeg/                     # EEG数据目录
│   │   │   ├── sub-001_ses-001_task-rest_eeg.edf
│   │   │   ├── sub-001_ses-001_task-rest_eeg.json
│   │   │   ├── sub-001_ses-001_task-rest_events.tsv
│   │   │   ├── sub-001_ses-001_task-rest_channels.tsv
│   │   │   └── sub-001_ses-001_task-rest_electrodes.tsv
│   │   └── anat/                    # 解剖数据（可选）
│   └── ses-002/                     # 会话2
│       └── eeg/
│           └── sub-001_ses-002_task-rest_eeg.edf
├── sub-002/                         # 主题2
│   └── eeg/                         # 直接EEG目录（无会话）
│       ├── sub-002_task-rest_eeg.edf
│       └── sub-002_task-rest_eeg.json
└── code/                            # 分析代码（可选）
    └── analysis_script.py
```

## 核心文件说明

### 1. dataset_description.json
数据集的基本描述信息：
```json
{
    "Name": "My EEG Dataset",
    "BIDSVersion": "1.4.0",
    "DatasetType": "raw",
    "Authors": ["Author 1", "Author 2"],
    "HowToAcknowledge": "Please cite our paper...",
    "Funding": ["Grant 1", "Grant 2"],
    "ReferencesAndLinks": ["Paper 1", "Paper 2"],
    "DatasetDOI": "10.5281/zenodo.1234567"
}
```

### 2. participants.tsv
参与者信息表格（制表符分隔）：
```tsv
participant_id	sex	age	group
sub-001	M	25	control
sub-002	F	30	patient
sub-003	M	28	control
```

### 3. EEG数据文件
支持多种格式：
- `.edf` - European Data Format
- `.bdf` - BioSemi Data Format
- `.set` - EEGLAB格式
- `.cnt` - Neuroscan格式
- `.mff` - EGI格式
- `.nxe` - Nexstim格式

### 4. 事件文件 (events.tsv)
记录实验事件的时间信息：
```tsv
onset	duration	trial_type	value	stim_file
1.2	0.1	stimulus	1	stim1.jpg
5.6	0.1	stimulus	2	stim2.jpg
10.1	0.1	response	1	NaN
```

### 5. 通道信息文件 (channels.tsv)
描述EEG通道的详细信息：
```tsv
name	type	units	sampling_frequency	description	status	status_description
Fp1	EEG	uV	1000	Frontal pole 1	good	Good signal
Fp2	EEG	uV	1000	Frontal pole 2	good	Good signal
F7	EEG	uV	1000	Frontal 7	bad	Bad signal
```

### 6. 电极位置文件 (electrodes.tsv)
记录电极的3D坐标位置：
```tsv
name	x	y	z	size	type	material
Fp1	-0.0307	0.0854	-0.0275	4	EEG	Ag/AgCl
Fp2	0.0307	0.0854	-0.0275	4	EEG	Ag/AgCl
F7	-0.0645	0.0294	-0.0124	4	EEG	Ag/AgCl
```

## 使用BIDS导入器

### 基本用法

```python
from database.import_bids_dataset import import_bids_dataset

# 导入BIDS数据集
dataset_id = import_bids_dataset(
    bids_root="/path/to/bids/dataset",
    dataset_name="My_EEG_Dataset",
    max_import_count=100
)
```

### 高级用法

```python
from database.import_bids_dataset import BIDSImporter

# 创建导入器实例
importer = BIDSImporter("/path/to/bids/dataset")

# 查看数据集信息
print(f"数据集名称: {importer.dataset_description.get('Name')}")
print(f"参与者数量: {len(importer.participants_info)}")

# 导入数据集
dataset_id = importer.import_dataset(
    dataset_name="Custom_Dataset",
    max_import_count=50
)
```

### 命令行使用

```bash
# 基本导入
python database/import_bids_dataset.py /path/to/bids/dataset

# 指定数据集名称
python database/import_bids_dataset.py /path/to/bids/dataset --dataset-name "My_Dataset"

# 限制导入数量
python database/import_bids_dataset.py /path/to/bids/dataset --max-import 100
```

## 导入过程详解

### 1. 验证BIDS结构
- 检查必需文件（dataset_description.json, participants.tsv）
- 验证主题目录结构（sub-*）
- 确认EEG数据目录存在

### 2. 读取元数据
- 解析数据集描述信息
- 加载参与者信息
- 读取会话信息（如果存在）

### 3. 扫描EEG文件
- 查找所有EEG数据文件
- 支持多种文件格式
- 读取相关的BIDS元数据文件

### 4. 数据导入
- 创建数据集记录
- 导入参与者信息
- 导入EEG记录和元数据
- 检测和导入事件信息

### 5. 事件检测
- 自动检测EEG中的事件标记
- 支持多种事件检测方法
- 导入事件到recording_events表

## 支持的BIDS特性

### ✅ 已实现
- 基本BIDS目录结构验证
- 数据集描述文件解析
- 参与者信息导入
- 多种EEG文件格式支持
- 事件文件解析
- 通道信息文件解析
- 电极位置文件解析
- 自动事件检测
- 会话信息支持

### 🔄 计划中
- 扫描信息文件支持
- 更多EEG文件格式
- 解剖数据支持
- 功能数据支持
- 扩散张量成像支持

## 错误处理和日志

导入器提供详细的日志记录和错误处理：

```python
import logging
from logging_config import logger

# 设置日志级别
logger.setLevel(logging.INFO)

# 查看导入过程
logger.info("开始导入BIDS数据集...")
logger.warning("发现无效的EEG文件，跳过...")
logger.error("导入失败：数据库连接错误")
```

## 性能优化建议

### 1. 内存管理
- 设置合适的MAX_MEMORY_GB限制
- 使用内存映射读取大文件
- 及时释放不需要的数据

### 2. 批量处理
- 使用max_import_count限制导入数量
- 分批处理大型数据集
- 利用数据库事务提高性能

### 3. 并行处理
- 考虑多进程导入
- 异步I/O操作
- 数据库连接池

## 常见问题

### Q: 我的BIDS数据集结构不同怎么办？
A: BIDS导入器支持灵活的目录结构，可以处理有会话和无会话的情况。

### Q: 支持哪些EEG文件格式？
A: 支持.edf, .bdf, .set, .cnt, .mff, .nxe等常见格式。

### Q: 如何处理缺失的元数据文件？
A: 导入器会优雅地处理缺失文件，使用默认值或跳过相关功能。

### Q: 可以导入部分数据集吗？
A: 是的，使用max_import_count参数可以限制导入的记录数量。

## 参考资料

- [BIDS官方规范](https://bids-specification.readthedocs.io/)
- [BIDS-EEG扩展](https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/03-electroencephalography.html)
- [BIDS验证工具](https://github.com/bids-standard/bids-validator)
- [BIDS社区](https://bids.neuroimaging.io/)

## 贡献

欢迎提交问题报告和功能请求！请确保：
1. 遵循BIDS标准规范
2. 提供完整的错误信息
3. 包含数据集结构示例
4. 测试新功能后再提交


