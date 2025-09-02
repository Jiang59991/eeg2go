#!/usr/bin/env python3
"""
Harvard BIDS数据集导入脚本

这个脚本专门用于导入Harvard EEG数据，虽然数据遵循BIDS目录结构，
但缺少标准的BIDS文件（如dataset_description.json和participants.tsv）。
"""

import os
import sqlite3
import mne
import pandas as pd
from logging_config import logger
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# 数据库路径
DB_PATH = os.path.join(os.path.dirname(__file__), "eeg2go.db")

# 内存使用限制
MAX_MEMORY_GB = 1
mne.utils.set_log_level('WARNING')

class HarvardBIDSImporter:
    """
    Harvard BIDS数据集导入器
    
    专门处理Harvard EEG数据，支持：
    - BIDS目录结构（sub-*, ses-*, eeg/）
    - 多种EEG文件格式
    - 自动事件检测
    - 元数据提取
    """
    
    def __init__(self, bids_root: str, db_path: str = None):
        """
        初始化Harvard BIDS导入器
        
        Args:
            bids_root: BIDS数据集根目录路径
            db_path: 数据库文件路径
        """
        self.bids_root = Path(bids_root)
        self.db_path = db_path or DB_PATH
        
        # 验证目录结构
        if not self._validate_directory_structure():
            raise ValueError(f"Invalid directory structure at {bids_root}")
        
        # 扫描主题信息
        self.subjects_info = self._scan_subjects()
        
        logger.info(f"Harvard BIDS importer initialized with {len(self.subjects_info)} subjects")
    
    def _validate_directory_structure(self) -> bool:
        """验证目录结构"""
        # 检查是否有sub-*目录
        subject_dirs = [d for d in self.bids_root.iterdir() if d.is_dir() and d.name.startswith("sub-")]
        if not subject_dirs:
            logger.error("No subject directories (sub-*) found")
            return False
        
        logger.info(f"Found {len(subject_dirs)} subject directories")
        return True
    
    def _scan_subjects(self) -> Dict[str, Dict[str, Any]]:
        """扫描所有主题信息"""
        subjects_info = {}
        
        for subject_dir in self.bids_root.iterdir():
            if not subject_dir.is_dir() or not subject_dir.name.startswith("sub-"):
                continue
            
            subject_id = subject_dir.name
            subjects_info[subject_id] = {
                'path': subject_dir,
                'sessions': [],
                'eeg_files': []
            }
            
            # 查找会话目录
            for item in subject_dir.iterdir():
                if item.is_dir() and item.name.startswith("ses-"):
                    session_info = {
                        'name': item.name,
                        'path': item,
                        'eeg_path': item / "eeg"
                    }
                    subjects_info[subject_id]['sessions'].append(session_info)
                elif item.is_dir() and item.name == "eeg":
                    # 直接eeg目录（无会话）
                    subjects_info[subject_id]['eeg_files'].extend(
                        self._find_eeg_files_in_directory(item)
                    )
            
            # 查找会话中的EEG文件
            for session in subjects_info[subject_id]['sessions']:
                if session['eeg_path'].exists():
                    subjects_info[subject_id]['eeg_files'].extend(
                        self._find_eeg_files_in_directory(session['eeg_path'])
                    )
        
        return subjects_info
    
    def _find_eeg_files_in_directory(self, eeg_dir: Path) -> List[Dict[str, Any]]:
        """在指定目录中查找EEG文件"""
        eeg_files = []
        
        if not eeg_dir.exists():
            return eeg_files
        
        for file_path in eeg_dir.iterdir():
            if file_path.is_file() and self._is_eeg_data_file(file_path):
                eeg_info = self._extract_eeg_file_info(file_path, eeg_dir)
                if eeg_info:
                    eeg_files.append(eeg_info)
        
        return eeg_files
    
    def _is_eeg_data_file(self, file_path: Path) -> bool:
        """判断文件是否为EEG数据文件"""
        eeg_extensions = ['.eeg', '.edf', '.bdf', '.set', '.cnt', '.mff', '.nxe']
        return file_path.suffix.lower() in eeg_extensions
    
    def _extract_eeg_file_info(self, file_path: Path, eeg_dir: Path) -> Optional[Dict[str, Any]]:
        """提取EEG文件信息"""
        try:
            # 读取EEG文件获取基本信息
            raw = mne.io.read_raw(file_path, preload=False, verbose='ERROR')
            
            # 读取相关的BIDS文件
            json_info = self._read_bids_json(file_path)
            channels_info = self._read_channels_file(file_path)
            events_info = self._read_events_file(file_path)
            
            return {
                'file_path': file_path,
                'file_name': file_path.name,
                'eeg_dir': eeg_dir,
                'raw': raw,
                'json_info': json_info,
                'channels_info': channels_info,
                'events_info': events_info,
                'sfreq': raw.info['sfreq'],
                'channels': len(raw.info['ch_names']),
                'duration': raw.n_times / raw.info['sfreq'],
                'channel_names': raw.info['ch_names']
            }
        except Exception as e:
            logger.error(f"Error extracting info from {file_path}: {e}")
            return None
    
    def _read_bids_json(self, eeg_file: Path) -> Dict[str, Any]:
        """读取BIDS JSON文件"""
        json_file = eeg_file.with_suffix('.json')
        if json_file.exists():
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading JSON file {json_file}: {e}")
        return {}
    
    def _read_channels_file(self, eeg_file: Path) -> Optional[pd.DataFrame]:
        """读取通道信息文件"""
        # 尝试多种可能的通道文件名
        possible_names = [
            eeg_file.with_suffix('.tsv'),
            eeg_file.parent / f"{eeg_file.stem.replace('_eeg', '')}_channels.tsv",
            eeg_file.parent / f"{eeg_file.stem}_channels.tsv"
        ]
        
        for channels_file in possible_names:
            if channels_file.exists():
                try:
                    df = pd.read_csv(channels_file, sep='\t')
                    logger.info(f"Loaded channels info from {channels_file}")
                    return df
                except Exception as e:
                    logger.error(f"Error reading channels file {channels_file}: {e}")
        
        return None
    
    def _read_events_file(self, eeg_file: Path) -> Optional[pd.DataFrame]:
        """读取事件文件"""
        # 尝试多种可能的事件文件名
        possible_names = [
            eeg_file.parent / f"{eeg_file.stem.replace('_eeg', '')}_events.tsv",
            eeg_file.parent / f"{eeg_file.stem}_events.tsv",
            eeg_file.parent / f"{eeg_file.stem.replace('_eeg', '')}_PersystSpikes.csv",
            eeg_file.parent / f"{eeg_file.stem}_PersystSpikes.csv"
        ]
        
        for event_file in possible_names:
            if event_file.exists():
                try:
                    if event_file.suffix == '.tsv':
                        df = pd.read_csv(event_file, sep='\t')
                    else:
                        df = pd.read_csv(event_file)
                    logger.info(f"Loaded events from {event_file}")
                    return df
                except Exception as e:
                    logger.error(f"Error reading events file {event_file}: {e}")
        
        return None
    
    def _detect_events_from_raw(self, raw: mne.io.Raw) -> Tuple[bool, Optional[str], List[Dict[str, Any]]]:
        """从原始EEG数据中检测事件"""
        try:
            events = None
            
            # 方法1: 自动检测
            try:
                events = mne.find_events(raw, verbose='ERROR')
            except:
                pass
            
            # 方法2: 检查STIM通道
            if events is None or len(events) == 0:
                stim_channels = [ch for ch in raw.ch_names if 'STI' in ch.upper() or 'TRIG' in ch.upper()]
                if stim_channels:
                    try:
                        events = mne.find_events(raw, stim_channel=stim_channels[0], verbose='ERROR')
                        logger.info(f"Found events using stim channel: {stim_channels[0]}")
                    except:
                        pass
            
            # 方法3: 检查所有可能的通道
            if events is None or len(events) == 0:
                for ch_name in raw.ch_names:
                    if any(keyword in ch_name.upper() for keyword in ['STIM', 'TRIG', 'EVENT', 'MARKER']):
                        try:
                            events = mne.find_events(raw, stim_channel=ch_name, verbose='ERROR')
                            if len(events) > 0:
                                logger.info(f"Found events using channel: {ch_name}")
                                break
                        except:
                            continue
            
            if events is not None and len(events) > 0:
                event_ids = np.unique(events[:, 2])
                event_types = event_ids.tolist()
                
                # 准备事件数据
                events_data = []
                for event in events:
                    onset = event[0] / raw.info['sfreq']
                    event_type = str(event[2])
                    events_data.append({
                        'event_type': event_type,
                        'onset': onset,
                        'duration': 0.0,
                        'value': str(event[2])
                    })
                
                logger.info(f"Found {len(events)} events with IDs: {event_types}")
                return True, json.dumps(event_types), events_data
            else:
                logger.info("No events found in recording")
                return False, None, []
                
        except Exception as e:
            logger.warning(f"Error detecting events: {e}")
            return False, None, []
    
    def import_dataset(self, dataset_name: str = None, max_import_count: int = None) -> int:
        """
        导入整个Harvard BIDS数据集
        
        Args:
            dataset_name: 数据集名称
            max_import_count: 最大导入数量限制
            
        Returns:
            dataset_id: 导入的数据集ID
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # 创建或获取数据集
        if dataset_name is None:
            dataset_name = f"Harvard_BIDS_{self.bids_root.name}"
        
        c.execute("SELECT id FROM datasets WHERE name = ?", (dataset_name,))
        row = c.fetchone()
        if row is None:
            c.execute("INSERT INTO datasets (name, description, source_type, path) VALUES (?, ?, ?, ?)",
                      (dataset_name, 
                       f"Harvard EEG BIDS dataset from {self.bids_root}",
                       "bids", 
                       str(self.bids_root)))
            dataset_id = c.lastrowid
        else:
            dataset_id = row[0]
        
        logger.info(f"Importing Harvard BIDS dataset '{dataset_name}' (ID: {dataset_id})")
        
        # 导入EEG数据
        imported_count = self._import_eeg_recordings(conn, dataset_id, max_import_count)
        
        conn.commit()
        conn.close()
        
        logger.info(f"Harvard BIDS dataset import complete: {imported_count} recordings imported")
        return dataset_id
    
    def _import_eeg_recordings(self, conn: sqlite3.Connection, dataset_id: int, max_import_count: int = None) -> int:
        """导入EEG记录"""
        c = conn.cursor()
        imported_count = 0
        
        # 遍历所有主题
        for subject_id, subject_info in self.subjects_info.items():
            if max_import_count is not None and imported_count >= max_import_count:
                logger.info(f"Reached import limit of {max_import_count}, stopping import.")
                break
            
            logger.info(f"Processing subject: {subject_id}")
            
            # 查找EEG文件
            eeg_files = subject_info['eeg_files']
            
            if not eeg_files:
                logger.warning(f"No EEG files found for {subject_id}")
                continue
            
            # 为每个主题选择一条记录（优先选择最长的）
            eeg_files.sort(key=lambda x: x['duration'], reverse=True)
            selected_recording = eeg_files[0]
            
            # 插入主题
            c.execute("INSERT OR IGNORE INTO subjects (subject_id, dataset_id) VALUES (?, ?)", 
                     (subject_id, dataset_id))
            
            # 检测事件
            has_events, event_types, events_data = self._detect_events_from_raw(selected_recording['raw'])
            
            # 提取BIDS元数据
            json_info = selected_recording['json_info']
            
            # 插入记录
            c.execute("""INSERT INTO recordings
                (dataset_id, subject_id, filename, path, duration, channels, sampling_rate,
                 original_reference, recording_type, eeg_ground, placement_scheme, manufacturer, 
                 powerline_frequency, software_filters, has_events, event_types)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (dataset_id, subject_id, selected_recording['file_name'], 
                 str(selected_recording['file_path']), selected_recording['duration'], 
                 selected_recording['channels'], selected_recording['sfreq'],
                 json_info.get("EEGReference", "n/a"),
                 json_info.get("RecordingType", "continuous"),
                 json_info.get("EEGGround", "n/a"),
                 json_info.get("EEGPlacementScheme", "n/a"),
                 json_info.get("Manufacturer", "n/a"),
                 json_info.get("PowerLineFrequency", "n/a"),
                 json_info.get("SoftwareFilters", "n/a"),
                 has_events, event_types))
            
            recording_id = c.lastrowid
            
            # 插入事件数据
            if has_events and events_data:
                for event in events_data:
                    c.execute("""INSERT INTO recording_events
                        (recording_id, event_type, onset, duration, value)
                        VALUES (?, ?, ?, ?, ?)""",
                        (recording_id, event['event_type'], event['onset'], 
                         event['duration'], event['value']))
                
                logger.info(f"Inserted {len(events_data)} events for recording {recording_id}")
            
            imported_count += 1
            logger.info(f"Imported: {selected_recording['file_name']} (duration: {selected_recording['duration']:.1f}s)")
        
        return imported_count


def import_harvard_bids_dataset(bids_root: str, dataset_name: str = None, max_import_count: int = None) -> int:
    """
    导入Harvard BIDS数据集的便捷函数
    
    Args:
        bids_root: BIDS数据集根目录路径
        dataset_name: 数据集名称
        max_import_count: 最大导入数量限制
        
    Returns:
        dataset_id: 导入的数据集ID
    """
    try:
        importer = HarvardBIDSImporter(bids_root)
        return importer.import_dataset(dataset_name, max_import_count)
    except Exception as e:
        logger.error(f"Error importing Harvard BIDS dataset: {e}")
        raise


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Import Harvard BIDS dataset')
    parser.add_argument('bids_root', help='Path to Harvard BIDS dataset root directory')
    parser.add_argument('--dataset-name', help='Dataset name (optional)')
    parser.add_argument('--max-import', type=int, help='Maximum number of recordings to import')
    
    args = parser.parse_args()
    
    try:
        dataset_id = import_harvard_bids_dataset(args.bids_root, args.dataset_name, args.max_import)
        print(f"Harvard BIDS dataset imported successfully with ID: {dataset_id}")
    except Exception as e:
        print(f"Import failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    main()


