import os
import mne
import shutil
from pathlib import Path

# 配置路径
BASE_DIR = Path("data/harvard_EEG")
BIDS_DIR = BASE_DIR / "bids"
MAX_MEMORY_GB = 8  # 最大内存使用限制（GB）
MAX_FILE_SIZE_GB = 2  # 最大文件大小限制（GB）

def get_file_size_gb(file_path):
    """获取文件大小（GB）"""
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024**3)
    except OSError:
        return 0

def estimate_memory_usage_gb(file_path):
    """估算加载文件后的内存使用量（GB）"""
    try:
        # 读取EDF文件头信息
        raw = mne.io.read_raw_edf(file_path, preload=False, verbose=False)
        
        # 估算内存使用：通道数 × 时间点数 × 8字节（float64）
        estimated_memory_gb = (len(raw.ch_names) * len(raw.times) * 8) / (1024**3)
        return estimated_memory_gb
    except Exception as e:
        print(f"无法读取文件 {file_path}: {e}")
        return float('inf')

def clean_large_files():
    """
    删除Harvard EEG数据集中过大的文件
    """
    print("开始清理Harvard EEG数据集中的大文件...")
    print(f"最大文件大小限制: {MAX_FILE_SIZE_GB} GB")
    print(f"最大内存使用限制: {MAX_MEMORY_GB} GB")
    
    if not BIDS_DIR.exists():
        print(f"错误: BIDS目录不存在: {BIDS_DIR}")
        return
    
    # 统计信息
    total_files = 0
    deleted_files = 0
    deleted_size_gb = 0
    large_files_info = []
    
    # 遍历所有EDF文件
    for edf_file in BIDS_DIR.rglob("*.edf"):
        total_files += 1
        file_size_gb = get_file_size_gb(edf_file)
        
        print(f"\n检查文件: {edf_file.name}")
        print(f"  文件大小: {file_size_gb:.2f} GB")
        
        # 检查文件大小
        if file_size_gb > MAX_FILE_SIZE_GB:
            print(f"  ✗ 文件过大，将被删除")
            large_files_info.append({
                'file': edf_file,
                'size_gb': file_size_gb,
                'reason': 'file_size'
            })
            continue
        
        # 估算内存使用
        try:
            memory_usage_gb = estimate_memory_usage_gb(edf_file)
            print(f"  估算内存使用: {memory_usage_gb:.2f} GB")
            
            if memory_usage_gb > MAX_MEMORY_GB:
                print(f"  ✗ 内存使用过大，将被删除")
                large_files_info.append({
                    'file': edf_file,
                    'size_gb': file_size_gb,
                    'memory_gb': memory_usage_gb,
                    'reason': 'memory_usage'
                })
                continue
            else:
                print(f"  ✓ 文件符合要求")
                
        except Exception as e:
            print(f"  ⚠ 无法估算内存使用: {e}")
            # 如果无法估算，但文件大小合理，则保留
            if file_size_gb <= MAX_FILE_SIZE_GB:
                print(f"  ✓ 文件大小合理，保留文件")
            else:
                print(f"  ✗ 文件过大且无法估算内存，将被删除")
                large_files_info.append({
                    'file': edf_file,
                    'size_gb': file_size_gb,
                    'reason': 'file_size_and_estimation_failed'
                })
    
    # 删除大文件
    print(f"\n{'='*60}")
    print("开始删除大文件...")
    
    for file_info in large_files_info:
        file_path = file_info['file']
        try:
            # 删除文件
            file_path.unlink()
            deleted_files += 1
            deleted_size_gb += file_info['size_gb']
            
            print(f"✓ 已删除: {file_path.name} ({file_info['size_gb']:.2f} GB)")
            print(f"  原因: {file_info['reason']}")
            
            # 如果目录为空，也删除目录
            parent_dir = file_path.parent
            if parent_dir.exists() and not any(parent_dir.iterdir()):
                try:
                    parent_dir.rmdir()
                    print(f"  删除空目录: {parent_dir}")
                except OSError:
                    pass  # 目录可能不为空或有其他文件
                    
        except Exception as e:
            print(f"✗ 删除失败: {file_path.name} - {e}")
    
    # 输出统计信息
    print(f"\n{'='*60}")
    print("清理完成！统计信息:")
    print(f"  总文件数: {total_files}")
    print(f"  删除文件数: {deleted_files}")
    print(f"  删除文件总大小: {deleted_size_gb:.2f} GB")
    print(f"  保留文件数: {total_files - deleted_files}")
    
    if deleted_files > 0:
        print(f"\n删除的文件详情:")
        for file_info in large_files_info:
            print(f"  - {file_info['file'].name}: {file_info['size_gb']:.2f} GB ({file_info['reason']})")
    
    # 保存删除记录
    if deleted_files > 0:
        log_file = BASE_DIR / "deleted_large_files.log"
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"Harvard EEG数据集大文件清理记录\n")
            f.write(f"清理时间: {os.popen('date').read().strip()}\n")
            f.write(f"最大文件大小限制: {MAX_FILE_SIZE_GB} GB\n")
            f.write(f"最大内存使用限制: {MAX_MEMORY_GB} GB\n\n")
            f.write(f"删除文件列表:\n")
            for file_info in large_files_info:
                f.write(f"- {file_info['file']}: {file_info['size_gb']:.2f} GB ({file_info['reason']})\n")
        
        print(f"\n删除记录已保存到: {log_file}")

def preview_large_files():
    """
    预览将要删除的大文件（不实际删除）
    """
    print("预览Harvard EEG数据集中的大文件...")
    print(f"最大文件大小限制: {MAX_FILE_SIZE_GB} GB")
    print(f"最大内存使用限制: {MAX_MEMORY_GB} GB")
    
    if not BIDS_DIR.exists():
        print(f"错误: BIDS目录不存在: {BIDS_DIR}")
        return
    
    large_files = []
    total_files = 0
    
    for edf_file in BIDS_DIR.rglob("*.edf"):
        total_files += 1
        file_size_gb = get_file_size_gb(edf_file)
        
        if file_size_gb > MAX_FILE_SIZE_GB:
            large_files.append({
                'file': edf_file,
                'size_gb': file_size_gb,
                'reason': 'file_size'
            })
            continue
        
        try:
            memory_usage_gb = estimate_memory_usage_gb(edf_file)
            if memory_usage_gb > MAX_MEMORY_GB:
                large_files.append({
                    'file': edf_file,
                    'size_gb': file_size_gb,
                    'memory_gb': memory_usage_gb,
                    'reason': 'memory_usage'
                })
        except Exception:
            if file_size_gb > MAX_FILE_SIZE_GB:
                large_files.append({
                    'file': edf_file,
                    'size_gb': file_size_gb,
                    'reason': 'file_size_and_estimation_failed'
                })
    
    print(f"\n发现 {len(large_files)} 个大文件（共 {total_files} 个文件）:")
    total_size = 0
    for file_info in large_files:
        print(f"  - {file_info['file'].name}: {file_info['size_gb']:.2f} GB ({file_info['reason']})")
        total_size += file_info['size_gb']
    
    print(f"\n总计将删除: {len(large_files)} 个文件，{total_size:.2f} GB")
    
    if large_files:
        response = input("\n是否继续删除这些文件？(y/N): ")
        if response.lower() == 'y':
            clean_large_files()
        else:
            print("取消删除操作")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--preview":
        preview_large_files()
    else:
        # 默认先预览，然后询问是否删除
        preview_large_files() 