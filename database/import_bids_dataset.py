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

class BIDSImporter:
    """
    BIDS (Brain Imaging Data Structure) 数据集导入器
    
    支持BIDS v1.4.0标准，包括：
    - 数据集描述文件 (dataset_description.json)
    - 参与者信息 (participants.tsv)
    - 会话信息 (sessions.tsv)
    - 扫描信息 (scans.tsv)
    - EEG数据文件 (.eeg, .edf, .bdf, .set等)
    - 事件文件 (.eve, .tsv)
    - 通道信息文件 (.tsv)
    - 电极位置文件 (.tsv)
    """
    
    def __init__(self, bids_root: str, db_path: str = None):
        """
        初始化BIDS导入器
        
        Args:
            bids_root: BIDS数据集根目录路径
            db_path: 数据库文件路径
        """
        self.bids_root = Path(bids_root)
        self.db_path = db_path or DB_PATH
        
        # 验证BIDS目录结构
        if not self._validate_bids_structure():
            raise ValueError(f"Invalid BIDS structure at {bids_root}")
        
        # 读取数据集描述
        self.dataset_description = self._read_dataset_description()
        
        # 读取参与者信息
        self.participants_info = self._read_participants_info()
        
        # 读取会话信息
        self.sessions_info = self._read_sessions_info()
        
        logger.info(f"BIDS importer initialized for dataset: {self.dataset_description.get('Name', 'Unknown')}")
    
    def _validate_bids_structure(self) -> bool:
        """验证BIDS目录结构"""
        required_files = [
            "dataset_description.json",
            "participants.tsv"
        ]
        
        for file_name in required_files:
            if not (self.bids_root / file_name).exists():
                logger.error(f"Required BIDS file not found: {file_name}")
                return False
        
        # 检查是否有sub-*目录
        subject_dirs = [d for d in self.bids_root.iterdir() if d.is_dir() and d.name.startswith("sub-")]
        if not subject_dirs:
            logger.error("No subject directories (sub-*) found")
            return False
        
        logger.info(f"Found {len(subject_dirs)} subject directories")
        return True
    
    def _read_dataset_description(self) -> Dict[str, Any]:
        """读取数据集描述文件"""
        desc_file = self.bids_root / "dataset_description.json"
        try:
            with open(desc_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading dataset description: {e}")
            return {}
    
    def _read_participants_info(self) -> pd.DataFrame:
        """读取参与者信息文件"""
        participants_file = self.bids_root / "participants.tsv"
        try:
            if participants_file.exists():
                df = pd.read_csv(participants_file, sep='\t', dtype=str)
                logger.info(f"Loaded participants info: {len(df)} participants")
                return df
            else:
                logger.warning("participants.tsv not found")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error reading participants file: {e}")
            return pd.DataFrame()
    
    def _read_sessions_info(self) -> Dict[str, pd.DataFrame]:
        """读取会话信息文件"""
        sessions_info = {}
        
        # 查找所有sessions.tsv文件
        for subject_dir in self.bids_root.iterdir():
            if subject_dir.is_dir() and subject_dir.name.startswith("sub-"):
                sessions_file = subject_dir / f"{subject_dir.name}_sessions.tsv"
                if sessions_file.exists():
                    try:
                        df = pd.read_csv(sessions_file, sep='\t', dtype=str)
                        sessions_info[subject_dir.name] = df
                    except Exception as e:
                        logger.error(f"Error reading sessions file for {subject_dir.name}: {e}")
        
        logger.info(f"Loaded sessions info for {len(sessions_info)} subjects")
        return sessions_info
    
    def _find_eeg_files(self, subject_dir: Path) -> List[Dict[str, Any]]:
        """查找主题目录下的所有EEG文件"""
        eeg_files = []
        
        # 查找eeg子目录
        eeg_dirs = []
        
        # 直接查找eeg目录
        direct_eeg = subject_dir / "eeg"
        if direct_eeg.exists():
            eeg_dirs.append(direct_eeg)
        
        # 查找会话目录下的eeg子目录
        for item in subject_dir.iterdir():
            if item.is_dir() and item.name.startswith("ses-"):
                ses_eeg = item / "eeg"
                if ses_eeg.exists():
                    eeg_dirs.append(ses_eeg)
        
        # 扫描所有eeg目录
        for eeg_dir in eeg_dirs:
            for file_path in eeg_dir.iterdir():
                if file_path.is_file():
                    # 检查是否为EEG数据文件
                    if self._is_eeg_data_file(file_path):
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
            events_info = self._read_events_file(file_path)
            channels_info = self._read_channels_file(file_path)
            electrodes_info = self._read_electrodes_file(file_path)
            
            return {
                'file_path': file_path,
                'file_name': file_path.name,
                'eeg_dir': eeg_dir,
                'raw': raw,
                'json_info': json_info,
                'events_info': events_info,
                'channels_info': channels_info,
                'electrodes_info': electrodes_info,
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
    
    def _read_events_file(self, eeg_file: Path) -> Optional[pd.DataFrame]:
        """读取事件文件"""
        # 尝试多种事件文件格式
        event_extensions = ['.eve', '.tsv']
        base_name = eeg_file.stem.replace('_eeg', '')
        
        for ext in event_extensions:
            event_file = eeg_file.parent / f"{base_name}_events{ext}"
            if event_file.exists():
                try:
                    if ext == '.tsv':
                        df = pd.read_csv(event_file, sep='\t')
                    else:
                        df = pd.read_csv(event_file, sep=' ', header=None)
                        if len(df.columns) >= 3:
                            df.columns = ['onset', 'duration', 'trial_type'] + list(df.columns[3:])
                    logger.info(f"Loaded events from {event_file}")
                    return df
                except Exception as e:
                    logger.error(f"Error reading events file {event_file}: {e}")
        
        return None
    
    def _read_channels_file(self, eeg_file: Path) -> Optional[pd.DataFrame]:
        """读取通道信息文件"""
        channels_file = eeg_file.with_suffix('.tsv')
        if channels_file.exists():
            try:
                df = pd.read_csv(channels_file, sep='\t')
                logger.info(f"Loaded channels info from {channels_file}")
                return df
            except Exception as e:
                logger.error(f"Error reading channels file {channels_file}: {e}")
        return None
    
    def _read_electrodes_file(self, eeg_file: Path) -> Optional[pd.DataFrame]:
        """读取电极位置文件"""
        electrodes_file = eeg_file.parent / f"{eeg_file.stem.replace('_eeg', '')}_electrodes.tsv"
        if electrodes_file.exists():
            try:
                df = pd.read_csv(electrodes_file, sep='\t')
                logger.info(f"Loaded electrodes info from {electrodes_file}")
                return df
            except Exception as e:
                logger.error(f"Error reading electrodes file {electrodes_file}: {e}")
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
        导入整个BIDS数据集
        
        Args:
            dataset_name: 数据集名称，如果为None则使用BIDS描述中的名称
            max_import_count: 最大导入数量限制
            
        Returns:
            dataset_id: 导入的数据集ID
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # 创建或获取数据集
        if dataset_name is None:
            dataset_name = self.dataset_description.get('Name', f"BIDS_{self.bids_root.name}")
        
        c.execute("SELECT id FROM datasets WHERE name = ?", (dataset_name,))
        row = c.fetchone()
        if row is None:
            c.execute("INSERT INTO datasets (name, description, source_type, path) VALUES (?, ?, ?, ?)",
                      (dataset_name, 
                       self.dataset_description.get('Description', f"BIDS dataset from {self.bids_root}"),
                       "bids", 
                       str(self.bids_root)))
            dataset_id = c.lastrowid
        else:
            dataset_id = row[0]
        
        logger.info(f"Importing BIDS dataset '{dataset_name}' (ID: {dataset_id})")
        
        # 导入参与者信息
        self._import_participants(conn, dataset_id)
        
        # 导入EEG数据
        imported_count = self._import_eeg_recordings(conn, dataset_id, max_import_count)
        
        conn.commit()
        conn.close()
        
        logger.info(f"BIDS dataset import complete: {imported_count} recordings imported")
        return dataset_id
    
    def _import_participants(self, conn: sqlite3.Connection, dataset_id: int):
        """导入参与者信息"""
        if self.participants_info.empty:
            logger.warning("No participants info available")
            return
        
        c = conn.cursor()
        inserted = 0
        
        for _, row in self.participants_info.iterrows():
            subject_id = row.get('participant_id', f"sub-{inserted}")
            
            # 插入主题
            c.execute("INSERT OR IGNORE INTO subjects (subject_id, dataset_id) VALUES (?, ?)", 
                     (subject_id, dataset_id))
            
            # 更新主题元数据
            c.execute("""UPDATE subjects SET
                sex = ?, age = ?, race = ?, ethnicity = ?
                WHERE subject_id = ? AND dataset_id = ?""", (
                row.get('sex'),
                self._parse_age(row.get('age')),
                row.get('race'),
                row.get('ethnicity'),
                subject_id, dataset_id
            ))
            
            inserted += 1
        
        logger.info(f"Imported {inserted} participants")
    
    def _parse_age(self, age_str: str) -> Optional[float]:
        """解析年龄字符串"""
        if pd.isna(age_str) or age_str == '':
            return None
        
        try:
            # 尝试直接转换为数字
            return float(age_str)
        except ValueError:
            # 处理特殊格式，如"25Y"表示25岁
            if isinstance(age_str, str):
                if age_str.upper().endswith('Y'):
                    return float(age_str[:-1])
                elif age_str.upper().endswith('M'):
                    return float(age_str[:-1]) / 12.0
                elif age_str.upper().endswith('D'):
                    return float(age_str[:-1]) / 365.25
        
        return None
    
    def _import_eeg_recordings(self, conn: sqlite3.Connection, dataset_id: int, max_import_count: int = None) -> int:
        """导入EEG记录"""
        c = conn.cursor()
        imported_count = 0
        
        # 遍历所有主题目录
        for subject_dir in self.bids_root.iterdir():
            if not subject_dir.is_dir() or not subject_dir.name.startswith("sub-"):
                continue
            
            if max_import_count is not None and imported_count >= max_import_count:
                logger.info(f"Reached import limit of {max_import_count}, stopping import.")
                break
            
            subject_id = subject_dir.name
            logger.info(f"Processing subject: {subject_id}")
            
            # 查找EEG文件
            eeg_files = self._find_eeg_files(subject_dir)
            
            if not eeg_files:
                logger.warning(f"No EEG files found for {subject_id}")
                continue
            
            # 为每个主题选择一条记录（优先选择最长的）
            eeg_files.sort(key=lambda x: x['duration'], reverse=True)
            selected_recording = eeg_files[0]
            
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


def import_bids_dataset(bids_root: str, dataset_name: str = None, max_import_count: int = None) -> int:
    """
    导入BIDS数据集的便捷函数
    
    Args:
        bids_root: BIDS数据集根目录路径
        dataset_name: 数据集名称
        max_import_count: 最大导入数量限制
        
    Returns:
        dataset_id: 导入的数据集ID
    """
    try:
        importer = BIDSImporter(bids_root)
        return importer.import_dataset(dataset_name, max_import_count)
    except Exception as e:
        logger.error(f"Error importing BIDS dataset: {e}")
        raise


def main():
    """主函数示例"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Import BIDS dataset')
    parser.add_argument('bids_root', help='Path to BIDS dataset root directory')
    parser.add_argument('--dataset-name', help='Dataset name (optional)')
    parser.add_argument('--max-import', type=int, help='Maximum number of recordings to import')
    
    args = parser.parse_args()
    
    try:
        dataset_id = import_bids_dataset(args.bids_root, args.dataset_name, args.max_import)
        print(f"Dataset imported successfully with ID: {dataset_id}")
    except Exception as e:
        print(f"Import failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    main()


