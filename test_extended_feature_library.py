#!/usr/bin/env python3
"""
测试扩展的EEG特征库
验证新添加的高级特征函数的功能和格式兼容性
"""

import sys
import os
import numpy as np
import mne
from datetime import datetime
import traceback

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from eeg2fx.function_registry import FEATURE_FUNCS, FEATURE_METADATA
from logging_config import logger

def create_test_epochs():
    """创建测试用的EEG epochs数据"""
    # 创建模拟的EEG数据
    n_channels = 4
    n_epochs = 10
    n_times = 1000
    sfreq = 250  # 采样频率
    
    # 生成模拟数据
    data = np.random.randn(n_epochs, n_channels, n_times) * 10
    
    # 添加一些频率成分
    t = np.linspace(0, n_times/sfreq, n_times)
    for i in range(n_epochs):
        for j in range(n_channels):
            # 添加alpha波 (8-13 Hz)
            alpha_freq = 10 + np.random.randn() * 2
            data[i, j, :] += 5 * np.sin(2 * np.pi * alpha_freq * t)
            
            # 添加theta波 (4-8 Hz)
            theta_freq = 6 + np.random.randn() * 2
            data[i, j, :] += 3 * np.sin(2 * np.pi * theta_freq * t)
    
    # 创建通道信息
    ch_names = ['Fp1', 'Fp2', 'C3', 'C4']
    ch_types = ['eeg'] * n_channels
    
    # 创建info对象
    info = mne.create_info(ch_names, sfreq, ch_types)
    
    # 创建epochs对象
    epochs = mne.EpochsArray(data, info)
    
    return epochs

def validate_feature_result(result, feature_name, n_epochs):
    """验证特征函数返回结果的格式"""
    if not isinstance(result, dict):
        return False, "返回值不是字典格式"
    
    # 检查是否有通道数据
    if len(result) == 0:
        return False, "返回的字典为空"
    
    # 检查每个通道的数据格式
    for channel_name, channel_data in result.items():
        if not isinstance(channel_data, list):
            return False, f"通道 {channel_name} 的数据不是列表格式"
        
        if len(channel_data) != n_epochs:
            return False, f"通道 {channel_name} 的数据长度 ({len(channel_data)}) 与epoch数量 ({n_epochs}) 不匹配"
        
        # 检查每个epoch的数据格式
        for i, epoch_data in enumerate(channel_data):
            if not isinstance(epoch_data, dict):
                return False, f"通道 {channel_name} 第 {i} 个epoch的数据不是字典格式"
            
            required_keys = ['epoch', 'start', 'end', 'value']
            for key in required_keys:
                if key not in epoch_data:
                    return False, f"通道 {channel_name} 第 {i} 个epoch缺少必需的键: {key}"
    
    return True, "格式正确"

def test_basic_features(epochs, features):
    """测试基本特征函数"""
    logger.info("测试基本特征函数...")
    
    basic_features = [
        'bandpower',
        'mean_amplitude', 
        'rms',
        'spectral_entropy'
    ]
    
    results = {}
    for feature_name in basic_features:
        if feature_name in features:
            try:
                logger.info(f"测试特征: {feature_name}")
                feature_func = features[feature_name]
                
                # 测试函数
                result = feature_func(epochs)
                
                # 验证结果格式
                is_valid, message = validate_feature_result(result, feature_name, len(epochs))
                if is_valid:
                    # 计算统计信息
                    all_values = []
                    for channel_data in result.values():
                        for epoch_data in channel_data:
                            all_values.append(epoch_data['value'])
                    
                    results[feature_name] = {
                        'status': 'success',
                        'n_channels': len(result),
                        'mean': np.mean(all_values),
                        'std': np.std(all_values)
                    }
                    logger.info(f"  ✓ {feature_name}: {len(result)}个通道, 均值={np.mean(all_values):.4f}")
                else:
                    results[feature_name] = {'status': 'error', 'message': message}
                    logger.error(f"  ✗ {feature_name}: {message}")
                    
            except Exception as e:
                results[feature_name] = {'status': 'error', 'message': str(e)}
                logger.error(f"  ✗ {feature_name}: {e}")
        else:
            results[feature_name] = {'status': 'missing', 'message': '特征函数不存在'}
            logger.warning(f"  ? {feature_name}: 特征函数不存在")
    
    return results

def test_advanced_freq_features(epochs, features):
    """测试高级频域特征"""
    logger.info("测试高级频域特征...")
    
    advanced_freq_features = [
        'spectral_centroid',
        'spectral_bandwidth', 
        'spectral_rolloff',
        'spectral_flatness',
        'spectral_skewness',
        'spectral_kurtosis',
        'band_energy_ratio',
        'spectral_complexity',
        'frequency_modulation_index'
    ]
    
    results = {}
    for feature_name in advanced_freq_features:
        if feature_name in features:
            try:
                logger.info(f"测试特征: {feature_name}")
                feature_func = features[feature_name]
                
                # 测试函数
                result = feature_func(epochs)
                
                # 验证结果格式
                is_valid, message = validate_feature_result(result, feature_name, len(epochs))
                if is_valid:
                    # 计算统计信息
                    all_values = []
                    for channel_data in result.values():
                        for epoch_data in channel_data:
                            all_values.append(epoch_data['value'])
                    
                    results[feature_name] = {
                        'status': 'success',
                        'n_channels': len(result),
                        'mean': np.mean(all_values),
                        'std': np.std(all_values)
                    }
                    logger.info(f"  ✓ {feature_name}: {len(result)}个通道, 均值={np.mean(all_values):.4f}")
                else:
                    results[feature_name] = {'status': 'error', 'message': message}
                    logger.error(f"  ✗ {feature_name}: {message}")
                    
            except Exception as e:
                results[feature_name] = {'status': 'error', 'message': str(e)}
                logger.error(f"  ✗ {feature_name}: {e}")
        else:
            results[feature_name] = {'status': 'missing', 'message': '特征函数不存在'}
            logger.warning(f"  ? {feature_name}: 特征函数不存在")
    
    return results

def test_advanced_time_features(epochs, features):
    """测试高级时域特征"""
    logger.info("测试高级时域特征...")
    
    advanced_time_features = [
        'signal_variance',
        'signal_skewness',
        'signal_kurtosis',
        'peak_to_peak_amplitude',
        'crest_factor',
        'shape_factor',
        'impulse_factor',
        'margin_factor',
        'signal_entropy',
        'signal_complexity',
        'signal_regularity',
        'signal_stability'
    ]
    
    results = {}
    for feature_name in advanced_time_features:
        if feature_name in features:
            try:
                logger.info(f"测试特征: {feature_name}")
                feature_func = features[feature_name]
                
                # 测试函数
                result = feature_func(epochs)
                
                # 验证结果格式
                is_valid, message = validate_feature_result(result, feature_name, len(epochs))
                if is_valid:
                    # 计算统计信息
                    all_values = []
                    for channel_data in result.values():
                        for epoch_data in channel_data:
                            all_values.append(epoch_data['value'])
                    
                    results[feature_name] = {
                        'status': 'success',
                        'n_channels': len(result),
                        'mean': np.mean(all_values),
                        'std': np.std(all_values)
                    }
                    logger.info(f"  ✓ {feature_name}: {len(result)}个通道, 均值={np.mean(all_values):.4f}")
                else:
                    results[feature_name] = {'status': 'error', 'message': message}
                    logger.error(f"  ✗ {feature_name}: {message}")
                    
            except Exception as e:
                results[feature_name] = {'status': 'error', 'message': str(e)}
                logger.error(f"  ✗ {feature_name}: {e}")
        else:
            results[feature_name] = {'status': 'missing', 'message': '特征函数不存在'}
            logger.warning(f"  ? {feature_name}: 特征函数不存在")
    
    return results

def test_advanced_connect_features(epochs, features):
    """测试高级连接性特征"""
    logger.info("测试高级连接性特征...")
    
    advanced_connect_features = [
        'mutual_information',
        'cross_correlation',
        'phase_synchronization',
        'amplitude_correlation',
        'granger_causality',
        'directed_transfer_function',
        'synchronization_likelihood'
    ]
    
    results = {}
    for feature_name in advanced_connect_features:
        if feature_name in features:
            try:
                logger.info(f"测试特征: {feature_name}")
                feature_func = features[feature_name]
                
                # 连接性特征需要指定通道对
                result = feature_func(epochs, chans="C3-C4")
                
                # 验证结果格式
                is_valid, message = validate_feature_result(result, feature_name, len(epochs))
                if is_valid:
                    # 计算统计信息
                    all_values = []
                    for channel_data in result.values():
                        for epoch_data in channel_data:
                            all_values.append(epoch_data['value'])
                    
                    results[feature_name] = {
                        'status': 'success',
                        'n_channels': len(result),
                        'mean': np.mean(all_values),
                        'std': np.std(all_values)
                    }
                    logger.info(f"  ✓ {feature_name}: {len(result)}个通道, 均值={np.mean(all_values):.4f}")
                else:
                    results[feature_name] = {'status': 'error', 'message': message}
                    logger.error(f"  ✗ {feature_name}: {message}")
                    
            except Exception as e:
                results[feature_name] = {'status': 'error', 'message': str(e)}
                logger.error(f"  ✗ {feature_name}: {e}")
        else:
            results[feature_name] = {'status': 'missing', 'message': '特征函数不存在'}
            logger.warning(f"  ? {feature_name}: 特征函数不存在")
    
    return results

def test_feature_parameters(epochs, features):
    """测试特征函数的参数处理"""
    logger.info("测试特征函数参数处理...")
    
    test_cases = [
        ('bandpower', {'band': 'beta'}),
        ('spectral_rolloff', {'percentile': 90}),
        ('signal_entropy', {'bins': 30}),
        ('mutual_information', {'bins': 15}),
        ('granger_causality', {'order': 3})
    ]
    
    results = {}
    for feature_name, params in test_cases:
        if feature_name in features:
            try:
                logger.info(f"测试特征参数: {feature_name} {params}")
                feature_func = features[feature_name]
                
                # 测试带参数的函数调用
                if feature_name in ['mutual_information', 'granger_causality']:
                    # 连接性特征需要通道参数
                    result = feature_func(epochs, chans="C3-C4", **params)
                else:
                    result = feature_func(epochs, **params)
                
                # 验证结果
                is_valid, message = validate_feature_result(result, feature_name, len(epochs))
                if is_valid:
                    results[feature_name] = {
                        'status': 'success',
                        'params': params,
                        'n_channels': len(result)
                    }
                    logger.info(f"  ✓ {feature_name}: 参数={params}, {len(result)}个通道")
                else:
                    results[feature_name] = {'status': 'error', 'message': message}
                    logger.error(f"  ✗ {feature_name}: {message}")
                    
            except Exception as e:
                results[feature_name] = {'status': 'error', 'message': str(e)}
                logger.error(f"  ✗ {feature_name}: {e}")
        else:
            results[feature_name] = {'status': 'missing', 'message': '特征函数不存在'}
            logger.warning(f"  ? {feature_name}: 特征函数不存在")
    
    return results

def test_feature_metadata():
    """测试特征元数据"""
    logger.info("测试特征元数据...")
    
    results = {}
    
    # 检查所有特征函数都有对应的元数据
    for feature_name in FEATURE_FUNCS.keys():
        if feature_name in FEATURE_METADATA:
            metadata = FEATURE_METADATA[feature_name]
            if 'type' in metadata and 'description' in metadata:
                results[feature_name] = {
                    'status': 'success',
                    'type': metadata['type'],
                    'description': metadata['description']
                }
                logger.info(f"  ✓ {feature_name}: {metadata['type']} - {metadata['description']}")
            else:
                results[feature_name] = {'status': 'error', 'message': '元数据格式不完整'}
                logger.error(f"  ✗ {feature_name}: 元数据格式不完整")
        else:
            results[feature_name] = {'status': 'missing', 'message': '缺少元数据'}
            logger.warning(f"  ? {feature_name}: 缺少元数据")
    
    return results

def generate_test_report(all_results):
    """生成测试报告"""
    logger.info("生成测试报告...")
    
    # 统计结果
    total_tests = 0
    successful_tests = 0
    failed_tests = 0
    missing_tests = 0
    
    for category, results in all_results.items():
        for feature_name, result in results.items():
            total_tests += 1
            if result['status'] == 'success':
                successful_tests += 1
            elif result['status'] == 'error':
                failed_tests += 1
            elif result['status'] == 'missing':
                missing_tests += 1
    
    # 生成报告
    report = f"""
=== EEG特征库扩展测试报告 ===
测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

总体统计:
- 总测试数: {total_tests}
- 成功: {successful_tests}
- 失败: {failed_tests}
- 缺失: {missing_tests}
- 成功率: {successful_tests/total_tests*100:.1f}%

详细结果:
"""
    
    for category, results in all_results.items():
        report += f"\n{category.upper()}:\n"
        report += "-" * 50 + "\n"
        
        for feature_name, result in results.items():
            status = result['status']
            if status == 'success':
                report += f"✓ {feature_name}: 成功"
                if 'n_channels' in result:
                    report += f" ({result['n_channels']}个通道)"
                if 'mean' in result:
                    report += f" [均值={result['mean']:.4f}]"
                report += "\n"
            elif status == 'error':
                report += f"✗ {feature_name}: 错误 - {result['message']}\n"
            elif status == 'missing':
                report += f"? {feature_name}: 缺失 - {result['message']}\n"
    
    return report

def main():
    """主测试函数"""
    logger.info("开始EEG特征库扩展测试...")
    
    try:
        # 获取特征函数
        logger.info("获取特征函数...")
        features = FEATURE_FUNCS
        logger.info(f"找到 {len(features)} 个特征函数")
        
        # 创建测试数据
        logger.info("创建测试数据...")
        epochs = create_test_epochs()
        logger.info(f"创建测试数据: {len(epochs)} epochs, {len(epochs.ch_names)} channels")
        
        # 运行各种测试
        all_results = {}
        
        # 测试基本特征
        all_results['basic_features'] = test_basic_features(epochs, features)
        
        # 测试高级频域特征
        all_results['advanced_freq_features'] = test_advanced_freq_features(epochs, features)
        
        # 测试高级时域特征
        all_results['advanced_time_features'] = test_advanced_time_features(epochs, features)
        
        # 测试高级连接性特征
        all_results['advanced_connect_features'] = test_advanced_connect_features(epochs, features)
        
        # 测试参数处理
        all_results['feature_parameters'] = test_feature_parameters(epochs, features)
        
        # 测试元数据
        all_results['feature_metadata'] = test_feature_metadata()
        
        # 生成报告
        report = generate_test_report(all_results)
        
        # 保存报告
        report_file = f"feature_library_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"测试报告已保存到: {report_file}")
        print(report)
        
        # 显示特征库统计信息
        logger.info("特征库统计信息:")
        logger.info(f"  总特征数量: {len(features)}")
        logger.info(f"  元数据数量: {len(FEATURE_METADATA)}")
        
        # 按类型统计特征
        type_counts = {}
        for metadata in FEATURE_METADATA.values():
            feature_type = metadata.get('type', 'unknown')
            type_counts[feature_type] = type_counts.get(feature_type, 0) + 1
        
        logger.info("按类型分类的特征:")
        for feature_type, count in type_counts.items():
            logger.info(f"  {feature_type}: {count} 个特征")
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        logger.error(traceback.format_exc())
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 