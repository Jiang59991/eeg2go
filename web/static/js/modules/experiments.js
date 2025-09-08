// modules/experiments.js
import { setActiveNavButton, hideAllViews, updateBreadcrumb } from './navigation.js';
import { showStatus } from './ui-utils.js';

export function initializeExperiments() {
    console.log('Initializing experiments module...');
}

export function showExperiments() {
    setActiveNavButton(document.getElementById('navExperimentsBtn'));
    hideAllViews();
    document.getElementById('experimentsView').style.display = 'block';
    updateBreadcrumb('experiments');
    
    loadExperiments();
}

export function showExperimentsView() {
    setActiveNavButton(document.getElementById('navExperimentsBtn'));
    hideAllViews();
    document.getElementById('experimentsView').style.display = 'block';
    updateBreadcrumb('experiments');
    
    loadExperiments();
}

export async function loadExperiments() {
    try {
        console.log('Loading experiments...');
        const response = await fetch('/api/experiments');
        
        if (!response.ok) {
            console.error(`HTTP error! status: ${response.status}`);
            const tbody = document.getElementById('experimentsTableBody');
            if (tbody) {
                tbody.innerHTML = `
                    <tr>
                        <td colspan="9" class="text-center py-4">
                            <div style="padding: 1.5rem 0; background-color: #f8d7da; border-radius: 8px; margin: 1rem 0;">
                                <i class="bi bi-exclamation-triangle text-danger" style="font-size: 2rem; opacity: 0.8; display: block; margin-bottom: 1rem;"></i>
                                <h6 class="text-danger" style="font-weight: 500; margin-bottom: 1rem;">Failed to Load Experiments</h6>
                                <p class="text-muted small mb-3">HTTP Error: ${response.status}</p>
                                <button class="btn btn-outline-primary btn-sm" onclick="loadExperiments()">
                                    <i class="bi bi-arrow-clockwise"></i> Try Again
                                </button>
                            </div>
                        </td>
                    </tr>
                `;
            }
            return;
        }
        
        const experiments = await response.json();
        console.log('Experiments loaded:', experiments);
        
        const tbody = document.getElementById('experimentsTableBody');
        if (!tbody) {
            console.error('experimentsTableBody element not found');
            return;
        }
        
        if (experiments.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="9" class="text-center py-5">
                        <div style="padding: 2rem 0; background-color: #f8f9fa; border-radius: 8px; margin: 1rem 0;">
                            <i class="bi bi-flask text-muted" style="font-size: 3rem; opacity: 0.5; display: block; margin-bottom: 1rem;"></i>
                            <h5 class="text-muted" style="font-weight: 500; margin-bottom: 1rem;">No Experiments Found</h5>
                            <p class="text-muted mb-3">There are currently no experiment results.</p>
                            <div class="text-muted small">
                                <p class="mb-1">To run an experiment:</p>
                                <ul class="list-unstyled">
                                    <li style="margin-bottom: 0.5rem;">
                                        <i class="bi bi-arrow-right" style="margin-right: 0.5rem; font-size: 0.875rem;"></i>
                                        Click "Add Experiment" to create a new experiment
                                    </li>
                                    <li style="margin-bottom: 0.5rem;">
                                        <i class="bi bi-arrow-right" style="margin-right: 0.5rem; font-size: 0.875rem;"></i>
                                        Select experiment type, dataset, and feature set
                                    </li>
                                    <li style="margin-bottom: 0.5rem;">
                                        <i class="bi bi-arrow-right" style="margin-right: 0.5rem; font-size: 0.875rem;"></i>
                                        Configure parameters and submit
                                    </li>
                                </ul>
                            </div>
                        </div>
                    </td>
                </tr>
            `;
            console.log('Displayed empty state for experiments');
        } else {
            tbody.innerHTML = experiments.map(exp => `
                <tr>
                    <td>${exp.id}</td>
                    <td>${exp.experiment_name || 'N/A'}</td>
                    <td>${exp.experiment_type}</td>
                    <td>${exp.dataset_name || 'N/A'}</td>
                    <td>${exp.feature_set_name || 'N/A'}</td>
                    <td>
                        <span class="badge bg-${getStatusBadgeColor(exp.status)}">${exp.status}</span>
                    </td>
                    <td>${exp.run_time ? new Date(exp.run_time).toLocaleString() : 'N/A'}</td>
                    <td>${exp.duration_seconds ? `${exp.duration_seconds}s` : 'N/A'}</td>
                    <td>
                        <button class="btn btn-sm btn-outline-primary" onclick="showExperimentDetails(${exp.id})">
                            <i class="bi bi-eye"></i> Details
                        </button>
                    </td>
                </tr>
            `).join('');
            console.log(`Displayed ${experiments.length} experiments`);
        }
        
    } catch (error) {
        console.error('Failed to load experiments:', error);
        const tbody = document.getElementById('experimentsTableBody');
        if (tbody) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="9" class="text-center py-4">
                        <div style="padding: 1.5rem 0; background-color: #f8d7da; border-radius: 8px; margin: 1rem 0;">
                            <i class="bi bi-exclamation-triangle text-danger" style="font-size: 2rem; opacity: 0.8; display: block; margin-bottom: 1rem;"></i>
                            <h6 class="text-danger" style="font-weight: 500; margin-bottom: 1rem;">Failed to Load Experiments</h6>
                            <p class="text-muted small mb-3">Network error or server issue</p>
                            <button class="btn btn-outline-primary btn-sm" onclick="loadExperiments()">
                                <i class="bi bi-arrow-clockwise"></i> Try Again
                            </button>
                        </div>
                    </td>
                </tr>
            `;
        }
    }
}

function getStatusBadgeColor(status) {
    switch (status) {
        case 'completed': return 'success';
        case 'running': return 'warning';
        case 'failed': return 'danger';
        case 'pending': return 'secondary';
        default: return 'secondary';
    }
}

export function showExperimentDetails(experimentId) {
    console.log('Showing experiment details for ID:', experimentId);
    
    fetch(`/api/experiment_details/${experimentId}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showStatus(`Error: ${data.error}`, 'error');
                return;
            }
            
            // 填充实验信息
            document.getElementById('experimentInfo').innerHTML = `
                <table class="table table-sm">
                    <tr><td><strong>ID:</strong></td><td>${data.experiment.id}</td></tr>
                    <tr><td><strong>Name:</strong></td><td>${data.experiment.experiment_name || 'N/A'}</td></tr>
                    <tr><td><strong>Type:</strong></td><td>${data.experiment.experiment_type}</td></tr>
                    <tr><td><strong>Dataset:</strong></td><td>${data.experiment.dataset_name || 'N/A'}</td></tr>
                    <tr><td><strong>Feature Set:</strong></td><td>${data.experiment.feature_set_name || 'N/A'}</td></tr>
                    <tr><td><strong>Status:</strong></td><td><span class="badge bg-${getStatusBadgeColor(data.experiment.status)}">${data.experiment.status}</span></td></tr>
                    <tr><td><strong>Run Time:</strong></td><td>${data.experiment.run_time ? new Date(data.experiment.run_time).toLocaleString() : 'N/A'}</td></tr>
                    <tr><td><strong>Duration:</strong></td><td>${data.experiment.duration_seconds ? `${data.experiment.duration_seconds}s` : 'N/A'}</td></tr>
                </table>
            `;
            
            // 填充实验摘要统计 - 优先从数据库调取，然后从任务结果调取
            const statsContainer = document.getElementById('experimentStats');
            let summaryHtml = '';
            
            if (data.experiment.result) {
                // 优先从任务结果中提取结构化数据
                try {
                    const resultData = JSON.parse(data.experiment.result);
                    
                    if (resultData.frontend_summary) {
                        summaryHtml = generateSummaryFromTaskResult(data.experiment.experiment_type, resultData.frontend_summary);
                        console.log('Using task result frontend_summary for summary');
                    } else if (data.feature_results && data.feature_results.length > 0) {
                        // 如果没有frontend_summary，回退到feature_results
                        const experimentType = data.experiment.experiment_type;
                        summaryHtml = generateExperimentSummary(experimentType, data.feature_results);
                        console.log('Using database feature results for summary (fallback)');
                    } else if (resultData.summary && resultData.summary.frontend_summary) {
                        summaryHtml = generateSummaryFromTaskResult(data.experiment.experiment_type, resultData.summary.frontend_summary);
                        console.log('Using nested task result frontend_summary for summary');
                } else if (resultData.summary) {
                        // 兼容旧格式的文本摘要
                        if (typeof resultData.summary === 'string') {
                            summaryHtml = generateLegacySummary(data.experiment.experiment_type, resultData.summary);
                            console.log('Using legacy text summary');
                        } else if (typeof resultData.summary === 'object') {
                            // 如果summary是对象，尝试提取有用的信息
                            summaryHtml = generateObjectSummary(data.experiment.experiment_type, resultData.summary);
                            console.log('Using object summary');
                        }
                    }
                } catch (e) {
                    console.warn('Failed to parse task result:', e);
                }
            }
            
            if (summaryHtml) {
                statsContainer.innerHTML = summaryHtml;
            } else if (data.results_index && data.results_index.summary) {
                // 兼容旧格式
                const summary = data.results_index.summary;
                statsContainer.innerHTML = `
                    <div class="row">
                        <div class="col-md-6">
                            <h6>Analysis Summary</h6>
                            <ul class="list-unstyled">
                                <li><strong>Total Features:</strong> ${summary.total_features || 'N/A'}</li>
                                <li><strong>Top Features Analyzed:</strong> ${summary.top_features_count || 'N/A'}</li>
                                <li><strong>Generated:</strong> ${summary.generated_at ? new Date(summary.generated_at).toLocaleString() : 'N/A'}</li>
                            </ul>
                        </div>
                        <div class="col-md-6">
                            <h6>Output Files</h6>
                            <ul class="list-unstyled">
                                <li><strong>Data Files:</strong> ${Object.keys(data.results_index.files || {}).length}</li>
                                <li><strong>Plots:</strong> ${Object.keys(data.results_index.plots || {}).length}</li>
                                <li><strong>Total Files:</strong> ${data.output_files.length}</li>
                            </ul>
                        </div>
                    </div>
                `;
            } else {
                statsContainer.innerHTML = '<p class="text-muted">No summary statistics available</p>';
            }
            
            // 填充特征结果表格 - 只显示具体的特征结果，过滤掉汇总数据
            const resultsBody = document.getElementById('experimentResultsBody');
            if (data.feature_results && data.feature_results.length > 0) {
                console.log('原始特征结果数据:', data.feature_results);
                
                // 过滤掉汇总数据，只保留具体的特征结果
                const filteredResults = data.feature_results.filter(result => {
                    // 跳过汇总数据 - 修复过滤逻辑
                    if (result.metric_name === 'significant_features_count' ||
                        result.metric_name === 'significant_associations' ||
                        result.metric_name === 'csv_content' ||
                        result.feature_name === 'overall_significant_associations' ||
                        (result.feature_name && result.feature_name.startsWith('associations_')) ||
                        (result.feature_name && result.feature_name.startsWith('csv_data_'))) {
                        console.log('过滤掉汇总数据:', result);
                        return false;
                    }
                    
                    // 对于 correlation 实验，保留具体的特征结果
                    if (data.experiment.experiment_type === 'correlation') {
                        const isValid = result.feature_name && 
                                      result.metric_name === 'correlation_coefficient' &&
                                      result.rank_position > 0 &&
                                      !result.feature_name.startsWith('summary_') &&
                                      !result.feature_name.startsWith('overall_');
                        
                        if (isValid) {
                            console.log('保留 correlation 特征结果:', result);
                        } else {
                            console.log('过滤掉 correlation 特征结果:', result);
                        }
                        
                        return isValid;
                    }
                    
                    // 对于其他实验类型，保持原有逻辑
                    return result.feature_name && 
                           result.metric_name !== 'significant_features_count' &&
                           result.metric_name !== 'significant_associations' &&
                           result.rank_position > 0;
                });
                
                console.log('过滤后的特征结果:', filteredResults);
                
                if (filteredResults.length > 0) {
                    resultsBody.innerHTML = filteredResults.map(result => `
                        <tr>
                            <td>${result.rank_position || 'N/A'}</td>
                            <td>${result.feature_name || result.feature_shortname || 'N/A'}</td>
                            <td>${result.feature_channels || 'N/A'}</td>
                            <td>${result.target_variable || 'N/A'}</td>
                            <td>${result.metric_name || 'N/A'}</td>
                            <td>${result.metric_value ? result.metric_value.toFixed(4) : 'N/A'}</td>
                            <td>${result.significance_level || 'N/A'}</td>
                        </tr>
                    `).join('');
                } else {
                    resultsBody.innerHTML = '<tr><td colspan="7" class="text-center text-muted">No detailed feature results available. Check Summary Statistics below for overview.</td></tr>';
                }
            } else {
                resultsBody.innerHTML = '<tr><td colspan="7" class="text-center text-muted">No feature results available</td></tr>';
            }
            
            // 显示输出文件
            displayExperimentFiles(data.output_files, data.results_index, experimentId);
            
            // 显示模态框
            const modal = new bootstrap.Modal(document.getElementById('experimentModal'));
            modal.show();
            
        })
        .catch(error => {
            console.error('Failed to load experiment details:', error);
            showStatus('Failed to load experiment details', 'error');
        });
}

// 根据实验类型生成摘要HTML
function generateExperimentSummary(experimentType, featureResults) {
    let summaryHtml = '<div class="row">';
    
    switch (experimentType) {
        case 'feature_statistics':
            summaryHtml += generateStatisticsSummary(featureResults);
            break;
        case 'correlation':
            summaryHtml += generateCorrelationSummary(featureResults);
            break;
        case 'classification':
            summaryHtml += generateClassificationSummary(featureResults);
            break;
        default:
            summaryHtml += generateGenericSummary(featureResults);
    }
    
    summaryHtml += '</div>';
    return summaryHtml;
}

// 生成特征统计摘要
function generateStatisticsSummary(featureResults) {
    const overallHealth = featureResults.find(r => r.feature_name === 'overall_dataset_health');
    const qualityDist = featureResults.filter(r => r.target_variable === 'quality_distribution');
    const qualityIssues = featureResults.filter(r => r.target_variable === 'quality_issues');
    
    let html = `
        <div class="col-md-6">
            <h6><i class="bi bi-clipboard-data"></i> Data Health Summary</h6>
            <div class="card bg-light">
                <div class="card-body p-3">
                    ${overallHealth ? `
                        <div class="d-flex justify-content-between align-items-center mb-2">
                            <span><strong>Overall Health Score:</strong></span>
                            <span class="badge bg-${getHealthScoreColor(overallHealth.metric_value)} fs-6">
                                ${(overallHealth.metric_value * 100).toFixed(1)}%
                            </span>
                        </div>
                    ` : ''}
                    ${qualityDist.length > 0 ? `
                        <div class="mb-2">
                            <strong>Quality Distribution:</strong>
                            <div class="mt-1">
                                ${qualityDist.map(q => `
                                    <span class="badge bg-secondary me-1">
                                        ${q.feature_name.replace('quality_grade_', '')}: ${q.metric_value}
                                    </span>
                                `).join('')}
                            </div>
                        </div>
                    ` : ''}
                </div>
            </div>
        </div>
        <div class="col-md-6">
            <h6><i class="bi bi-exclamation-triangle"></i> Quality Issues</h6>
            <div class="card bg-light">
                <div class="card-body p-3">
                    ${qualityIssues.length > 0 ? qualityIssues.map(issue => `
                        <div class="d-flex justify-content-between align-items-center mb-1">
                            <span>${formatIssueName(issue.feature_name)}:</span>
                            <span class="badge bg-warning">${issue.metric_value}</span>
                        </div>
                    `).join('') : '<p class="text-muted mb-0">No major issues detected</p>'}
                </div>
            </div>
        </div>
    `;
    
    return html;
}

// 生成关联性分析摘要
function generateCorrelationSummary(featureResults) {
    // 处理新的数据格式：从featureResults中提取correlation数据
    const correlationResults = featureResults.filter(r => r.result_type === 'correlation' && r.metric_name === 'correlation_coefficient');
    const significantResults = correlationResults.filter(r => r.significance_level && r.significance_level !== 'ns');
    
    // 按target_variable分组
    const targetGroups = {};
    correlationResults.forEach(result => {
        const target = result.target_variable;
        if (!targetGroups[target]) {
            targetGroups[target] = [];
        }
        targetGroups[target].push(result);
    });
    
    // 计算每个target的显著特征数量
    const targetStats = {};
    Object.keys(targetGroups).forEach(target => {
        const targetResults = targetGroups[target];
        const significantCount = targetResults.filter(r => r.significance_level && r.significance_level !== 'ns').length;
        const totalCount = targetResults.length;
        targetStats[target] = {
            significant_count: significantCount,
            total_count: totalCount,
            significant_ratio: totalCount > 0 ? significantCount / totalCount : 0
        };
    });
    
    let html = `
        <div class="col-md-6">
            <h6><i class="bi bi-graph-up"></i> Overall Associations</h6>
            <div class="card bg-light">
                <div class="card-body p-3">
                    <div class="d-flex justify-content-between align-items-center mb-2">
                        <span><strong>Significant Features:</strong></span>
                        <span class="badge bg-success fs-6">${significantResults.length}</span>
                    </div>
                    <div class="text-muted small">
                        Features with q < 0.05 after FDR correction
                    </div>
                </div>
            </div>
        </div>
        <div class="col-md-6">
            <h6><i class="bi bi-target"></i> Per-Target Results</h6>
            <div class="card bg-light">
                <div class="card-body p-3">
                    ${Object.keys(targetStats).length > 0 ? Object.entries(targetStats).map(([target_var, stats]) => `
                        <div class="mb-2">
                            <strong>${target_var}:</strong>
                            <div class="d-flex justify-content-between">
                                <span>Significant: ${stats.significant_count}</span>
                                <span class="text-muted">(${(stats.significant_ratio * 100).toFixed(1)}%)</span>
                            </div>
                        </div>
                    `).join('') : '<p class="text-muted mb-0">No target-specific results</p>'}
                </div>
            </div>
        </div>
    `;
    
    // 添加具体的特征结果表格
    const featureTable = generateCorrelationFeatureTable(featureResults);
    if (featureTable && !featureTable.includes('No detailed feature results available')) {
        html += `
            <div class="col-12 mt-3">
                <h6><i class="bi bi-table"></i> Top Feature Results</h6>
                ${featureTable}
            </div>
        `;
    }
    
    return html;
}

// 生成相关性分析的具体特征结果表格
function generateCorrelationFeatureTable(featureResults) {
    // 查找包含具体特征数据的记录
    const csvDataRecords = featureResults.filter(r => r.feature_name.startsWith('csv_data_'));
    
    if (csvDataRecords.length === 0) {
        return '<p class="text-muted">No detailed feature results available</p>';
    }
    
    let tableHtml = '<div class="table-responsive"><table class="table table-sm table-striped">';
    tableHtml += '<thead><tr><th>Rank</th><th>Feature</th><th>Correlation</th><th>P-value</th><th>Q-value</th><th>Significance</th></tr></thead><tbody>';
    
    csvDataRecords.forEach(record => {
        try {
            const additionalData = record.additional_data ? 
                (typeof record.additional_data === 'string' ? JSON.parse(record.additional_data) : record.additional_data) : {};
            
            if (additionalData.csv_content) {
                // 解析CSV内容
                const lines = additionalData.csv_content.split('\n');
                const headers = lines[0].split(',');
                const dataLines = lines.slice(1).filter(line => line.trim());
                
                // 找到相关列的位置
                const featureIndex = headers.findIndex(h => h.includes('feature'));
                const corrIndex = headers.findIndex(h => h.includes('correlation'));
                const pIndex = headers.findIndex(h => h.includes('p_value'));
                const qIndex = headers.findIndex(h => h.includes('q_value'));
                
                // 显示前20个特征
                dataLines.slice(0, 20).forEach((line, index) => {
                    const values = line.split(',');
                    const feature = values[featureIndex] || 'N/A';
                    const correlation = parseFloat(values[corrIndex]) || 0;
                    const pValue = parseFloat(values[pIndex]) || 1;
                    const qValue = parseFloat(values[qIndex]) || 1;
                    
                    // 确定显著性
                    let significance = 'ns';
                    if (qValue < 0.001) significance = '***';
                    else if (qValue < 0.01) significance = '**';
                    else if (qValue < 0.05) significance = '*';
                    
                    tableHtml += `<tr>
                        <td>${index + 1}</td>
                        <td>${feature}</td>
                        <td>${correlation.toFixed(3)}</td>
                        <td>${pValue.toFixed(6)}</td>
                        <td>${qValue.toFixed(6)}</td>
                        <td><span class="badge bg-${significance === 'ns' ? 'secondary' : 'success'}">${significance}</span></td>
                    </tr>`;
                });
            }
        } catch (e) {
            console.error('Error parsing CSV data:', e);
        }
    });
    
    tableHtml += '</tbody></table></div>';
    return tableHtml;
}

// 生成特征选择摘要

// 生成分类分析摘要
function generateClassificationSummary(featureResults) {
    const overallPerformance = featureResults.find(r => r.feature_name === 'overall_classification_performance');
    const targetPerformance = featureResults.filter(r => r.feature_name.startsWith('classification_'));
    
    let html = `
        <div class="col-md-6">
            <h6><i class="bi bi-robot"></i> Overall Performance</h6>
            <div class="card bg-light">
                <div class="card-body p-3">
                    ${overallPerformance ? `
                        <div class="d-flex justify-content-between align-items-center mb-2">
                            <span><strong>Average F1 Score:</strong></span>
                            <span class="badge bg-${getPerformanceColor(overallPerformance.metric_value)} fs-6">
                                ${(overallPerformance.metric_value * 100).toFixed(1)}%
                            </span>
                        </div>
                    ` : ''}
                    <div class="text-muted small">
                        Cross-validation performance across targets
                    </div>
                </div>
            </div>
        </div>
        <div class="col-md-6">
            <h6><i class="bi bi-target"></i> Per-Target Performance</h6>
            <div class="card bg-light">
                <div class="card-body p-3">
                    ${targetPerformance.length > 0 ? targetPerformance.map(target => {
                        const targetVar = target.feature_name.replace('classification_', '');
                        const additionalData = target.additional_data ? 
                            (typeof target.additional_data === 'string' ? JSON.parse(target.additional_data) : target.additional_data) : {};
                        return `
                            <div class="mb-2">
                                <strong>${targetVar}:</strong>
                                <div class="d-flex justify-content-between">
                                    <span>F1: ${(target.metric_value * 100).toFixed(1)}%</span>
                                    ${additionalData.roc_auc ? `
                                        <span class="text-muted">AUC: ${(additionalData.roc_auc * 100).toFixed(1)}%</span>
                                    ` : ''}
                                </div>
                            </div>
                        `;
                    }).join('') : '<p class="text-muted mb-0">No target-specific results</p>'}
                </div>
            </div>
        </div>
    `;
    
    return html;
}

// 生成通用摘要
function generateGenericSummary(featureResults) {
    const resultTypes = [...new Set(featureResults.map(r => r.result_type))];
    const totalFeatures = featureResults.length;
    
    let html = `
        <div class="col-md-6">
            <h6><i class="bi bi-info-circle"></i> Results Overview</h6>
            <div class="card bg-light">
                <div class="card-body p-3">
                    <div class="mb-2">
                        <strong>Total Results:</strong> ${totalFeatures}
                    </div>
                    <div class="mb-2">
                        <strong>Result Types:</strong>
                        <div class="mt-1">
                            ${resultTypes.map(type => `
                                <span class="badge bg-secondary me-1">${type}</span>
                            `).join('')}
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <div class="col-md-6">
            <h6><i class="bi bi-bar-chart"></i> Top Results</h6>
            <div class="card bg-light">
                <div class="card-body p-3">
                    ${featureResults.slice(0, 5).map(result => `
                        <div class="d-flex justify-content-between align-items-center mb-1">
                            <span class="text-truncate" style="max-width: 150px;" title="${result.feature_name}">
                                ${result.feature_name}
                            </span>
                            <span class="badge bg-info">${result.metric_value ? result.metric_value.toFixed(3) : 'N/A'}</span>
                        </div>
                    `).join('')}
                </div>
            </div>
        </div>
    `;
    
    return html;
}

// 从任务结果生成摘要
function generateSummaryFromTaskResult(experimentType, frontendSummary) {
    let summaryHtml = '<div class="row">';
    
    switch (experimentType) {
        case 'feature_statistics':
            summaryHtml += generateStatisticsSummaryFromTask(frontendSummary);
            break;
        case 'correlation':
            summaryHtml += generateCorrelationSummaryFromTask(frontendSummary);
            break;
        case 'classification':
            summaryHtml += generateClassificationSummaryFromTask(frontendSummary);
            break;
        default:
            summaryHtml += generateGenericSummaryFromTask(frontendSummary);
    }
    
    summaryHtml += '</div>';
    return summaryHtml;
}

// 从任务结果生成特征统计摘要
function generateStatisticsSummaryFromTask(frontendSummary) {
    const { overall_health_score, overall_health_grade, quality_distribution, quality_issues, total_features, total_samples, top_worst_features } = frontendSummary;
    
    let html = `
        <div class="col-md-6">
            <h6><i class="bi bi-clipboard-data"></i> Data Health Summary</h6>
            <div class="card bg-light">
                <div class="card-body p-3">
                    <div class="d-flex justify-content-between align-items-center mb-2">
                        <span><strong>Overall Health Score:</strong></span>
                        <span class="badge bg-${getHealthScoreColor(overall_health_score / 100)} fs-6">
                            ${overall_health_score.toFixed(1)}%
                        </span>
                    </div>
                    <div class="mb-2">
                        <strong>Health Grade:</strong> <span class="badge bg-success">${overall_health_grade}</span>
                    </div>
                    <div class="mb-2">
                        <strong>Total Features:</strong> ${total_features}
                    </div>
                    <div class="mb-2">
                        <strong>Total Samples:</strong> ${total_samples}
                    </div>
                </div>
            </div>
        </div>
        <div class="col-md-6">
            <h6><i class="bi bi-exclamation-triangle"></i> Quality Issues</h6>
            <div class="card bg-light">
                <div class="card-body p-3">
                    ${quality_issues.high_missing_features > 0 ? `
                        <div class="d-flex justify-content-between align-items-center mb-1">
                            <span>High Missing Features:</span>
                            <span class="badge bg-warning">${quality_issues.high_missing_features}</span>
                        </div>
                    ` : ''}
                    ${quality_issues.zero_variance_features > 0 ? `
                        <div class="d-flex justify-content-between align-items-center mb-1">
                            <span>Zero Variance Features:</span>
                            <span class="badge bg-warning">${quality_issues.zero_variance_features}</span>
                        </div>
                    ` : ''}
                    ${quality_issues.high_extreme_features > 0 ? `
                        <div class="d-flex justify-content-between align-items-center mb-1">
                            <span>High Extreme Features:</span>
                            <span class="badge bg-warning">${quality_issues.high_extreme_features}</span>
                        </div>
                    ` : ''}
                    ${quality_issues.high_missing_features === 0 && quality_issues.zero_variance_features === 0 && quality_issues.high_extreme_features === 0 ? 
                        '<p class="text-muted mb-0">No major issues detected</p>' : ''}
                </div>
            </div>
        </div>
    `;
    
    // 添加最差特征卡片
    if (top_worst_features && top_worst_features.length > 0) {
        html += `
        <div class="col-12 mt-3">
            <h6><i class="bi bi-exclamation-triangle-fill text-warning"></i> Top-5 Worst Quality Features</h6>
            <div class="card bg-light">
                <div class="card-body p-3">
                    <div class="table-responsive">
                        <table class="table table-sm table-hover mb-0">
                            <thead>
                                <tr>
                                    <th>Feature</th>
                                    <th>Quality Score</th>
                                    <th>Grade</th>
                                    <th>Missing Rate</th>
                                    <th>Zero Variance</th>
                                    <th>Extreme Rate</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${top_worst_features.map(feature => `
                                    <tr>
                                        <td><code class="small">${feature.feature}</code></td>
                                        <td>
                                            <span class="badge bg-${getHealthScoreColor(feature.quality_score / 100)}">
                                                ${feature.quality_score.toFixed(1)}
                                            </span>
                                        </td>
                                        <td>
                                            <span class="badge bg-${getQualityGradeColor(feature.quality_grade)}">
                                                ${feature.quality_grade}
                                            </span>
                                        </td>
                                        <td>${(feature.missing_rate * 100).toFixed(1)}%</td>
                                        <td>${feature.zero_variance ? '<i class="bi bi-check-circle-fill text-danger"></i>' : '<i class="bi bi-x-circle text-success"></i>'}</td>
                                        <td>${(feature.extreme_rate * 100).toFixed(1)}%</td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
        `;
    }
    
    return html;
}

// 从任务结果生成关联性分析摘要
function generateCorrelationSummaryFromTask(frontendSummary) {
    const { overall_significant_features, target_variables, total_features, total_samples, correlation_method, fdr_alpha } = frontendSummary;
    
    let html = `
        <div class="col-md-4">
            <h6><i class="bi bi-graph-up"></i> Overall Summary</h6>
            <div class="card bg-light">
                <div class="card-body p-3">
                    <div class="mb-2">
                        <small class="text-muted">Total Features:</small>
                        <div class="fw-bold">${total_features || 0}</div>
                    </div>
                    <div class="mb-2">
                        <small class="text-muted">Total Samples:</small>
                        <div class="fw-bold">${total_samples || 0}</div>
                    </div>
                    <div class="mb-2">
                        <small class="text-muted">Significant Features:</small>
                        <div class="fw-bold text-success">${overall_significant_features || 0}</div>
                    </div>
                    <div class="text-muted small">
                        Method: ${correlation_method || 'pearson'}, FDR α = ${fdr_alpha || 0.05}
                    </div>
                </div>
            </div>
        </div>
        <div class="col-md-8">
            <h6><i class="bi bi-target"></i> Target Variable Analysis</h6>
            <div class="card bg-light">
                <div class="card-body p-3">
                    ${target_variables && Object.keys(target_variables).length > 0 ? Object.entries(target_variables).map(([target_var, target_data]) => `
                        <div class="mb-3">
                            <div class="d-flex justify-content-between align-items-center mb-2">
                                <strong>${target_var} (${target_data.type})</strong>
                                <div>
                                    <span class="badge bg-success me-1">${target_data.significant_count || 0} significant</span>
                                    <span class="text-muted small">(${(target_data.significant_ratio * 100).toFixed(1)}%)</span>
                                </div>
                            </div>
                            ${target_data.top_associations && target_data.top_associations.length > 0 ? `
                                <div class="small">
                                    <div class="text-muted mb-1">Top 5 Associations:</div>
                                    <div class="table-responsive">
                                        <table class="table table-sm table-hover mb-0">
                                            <thead>
                                                <tr>
                                                    <th>Feature</th>
                                                    <th>Correlation</th>
                                                    <th>Q-value</th>
                                                    <th>Significance</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                ${target_data.top_associations.map(assoc => `
                                                    <tr>
                                                        <td><code class="small">${assoc.feature}</code></td>
                                                        <td>
                                                            <span class="badge bg-${assoc.correlation > 0 ? 'primary' : 'info'}">
                                                                ${assoc.correlation}
                                                            </span>
                                                        </td>
                                                        <td>${assoc.q_value}</td>
                                                        <td>
                                                            <span class="badge bg-${assoc.significance === '***' ? 'danger' : assoc.significance === '**' ? 'warning' : assoc.significance === '*' ? 'info' : 'secondary'}">
                                                                ${assoc.significance}
                                                            </span>
                                                        </td>
                                                    </tr>
                                                `).join('')}
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                            ` : '<p class="text-muted small mb-0">No significant associations found</p>'}
                        </div>
                    `).join('') : '<p class="text-muted mb-0">No target-specific results</p>'}
                </div>
            </div>
        </div>
    `;
    
    return html;
}


// 从任务结果生成分类分析摘要
function generateClassificationSummaryFromTask(frontendSummary) {
    const { overall_performance, target_performance, total_targets, total_features_used } = frontendSummary;
    
    let html = `
        <div class="col-md-4">
            <h6><i class="bi bi-robot"></i> Overall Summary</h6>
            <div class="card bg-light">
                <div class="card-body p-3">
                    <div class="mb-2">
                        <small class="text-muted">Total Targets:</small>
                        <div class="fw-bold">${total_targets || 0}</div>
                    </div>
                    <div class="mb-2">
                        <small class="text-muted">Features Used:</small>
                        <div class="fw-bold">${total_features_used || 0}</div>
                    </div>
                    ${overall_performance ? `
                        <div class="mb-2">
                            <small class="text-muted">Average F1 Score:</small>
                            <div class="fw-bold text-success">${(overall_performance.average_f1 * 100).toFixed(1)}%</div>
                        </div>
                        <div class="mb-2">
                            <small class="text-muted">Average Accuracy:</small>
                            <div class="fw-bold text-info">${(overall_performance.average_accuracy * 100).toFixed(1)}%</div>
                        </div>
                        ${overall_performance.average_roc_auc ? `
                            <div class="mb-2">
                                <small class="text-muted">Average ROC AUC:</small>
                                <div class="fw-bold text-warning">${(overall_performance.average_roc_auc * 100).toFixed(1)}%</div>
                            </div>
                        ` : ''}
                    ` : ''}
                    <div class="text-muted small">
                        Baseline predictive check results
                    </div>
                </div>
            </div>
        </div>
        <div class="col-md-8">
            <h6><i class="bi bi-target"></i> Target Performance Details</h6>
            <div class="card bg-light">
                <div class="card-body p-3">
                    ${target_performance ? Object.entries(target_performance).map(([target_var, target_data]) => `
                        <div class="mb-3">
                            <div class="d-flex justify-content-between align-items-center mb-2">
                                <strong>${target_var}</strong>
                                <span class="badge bg-primary">${target_data.model || 'Unknown Model'}</span>
                            </div>
                            <div class="row">
                                <div class="col-md-6">
                                    <div class="small">
                                        <div class="d-flex justify-content-between mb-1">
                                            <span>F1 Score:</span>
                                            <span class="badge bg-${getPerformanceColor(target_data.f1_score)}">
                                                ${(target_data.f1_score * 100).toFixed(1)}%
                                            </span>
                                        </div>
                                        <div class="d-flex justify-content-between mb-1">
                                            <span>Accuracy:</span>
                                            <span class="badge bg-${getPerformanceColor(target_data.accuracy)}">
                                                ${(target_data.accuracy * 100).toFixed(1)}%
                                            </span>
                                        </div>
                                        ${target_data.roc_auc ? `
                                            <div class="d-flex justify-content-between mb-1">
                                                <span>ROC AUC:</span>
                                                <span class="badge bg-${getPerformanceColor(target_data.roc_auc)}">
                                                    ${(target_data.roc_auc * 100).toFixed(1)}%
                                                </span>
                                            </div>
                                        ` : ''}
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="small">
                                        <div class="text-muted mb-1">Cross-Validation:</div>
                                        <div class="d-flex justify-content-between mb-1">
                                            <span>CV Mean:</span>
                                            <span class="text-muted">${(target_data.cv_mean * 100).toFixed(1)}%</span>
                                        </div>
                                        <div class="d-flex justify-content-between mb-1">
                                            <span>CV Std:</span>
                                            <span class="text-muted">±${(target_data.cv_std * 100).toFixed(1)}%</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    `).join('') : '<p class="text-muted mb-0">No target-specific results</p>'}
                </div>
            </div>
        </div>
    `;
    
    return html;
}

// 从任务结果生成通用摘要
function generateGenericSummaryFromTask(frontendSummary) {
    const keys = Object.keys(frontendSummary);
    
    let html = '<div class="row">';
    html += '<div class="col-md-6"><h6>Results Overview</h6>';
    html += '<div class="card bg-light"><div class="card-body p-3">';
    
    for (const key of keys.slice(0, 3)) {
        const value = frontendSummary[key];
        if (typeof value === 'number') {
            html += `<p><strong>${formatKeyName(key)}:</strong> ${value}</p>`;
        } else if (typeof value === 'string') {
            html += `<p><strong>${formatKeyName(key)}:</strong> ${value}</p>`;
        }
    }
    
    html += '</div></div></div>';
    html += '<div class="col-md-6"><h6>Additional Data</h6>';
    html += '<div class="card bg-light"><div class="card-body p-3">';
    
    for (const key of keys.slice(3, 6)) {
        const value = frontendSummary[key];
        if (typeof value === 'number') {
            html += `<p><strong>${formatKeyName(key)}:</strong> ${value}</p>`;
        } else if (typeof value === 'string') {
            html += `<p><strong>${formatKeyName(key)}:</strong> ${value}</p>`;
        }
    }
    
    html += '</div></div></div></div>';
    
    return html;
}

// 生成旧格式摘要
function generateLegacySummary(experimentType, summaryText) {
    return `
        <div class="row">
            <div class="col-12">
                <h6><i class="bi bi-file-text"></i> Text Summary</h6>
                <div class="card bg-light">
                    <div class="card-body p-3">
                        <pre class="mb-0" style="white-space: pre-wrap; font-size: 0.875rem;">${summaryText}</pre>
                    </div>
                </div>
            </div>
        </div>
    `;
}

// 生成对象摘要
function generateObjectSummary(experimentType, summaryObj) {
    let html = '<div class="row">';
    
    // 尝试提取有用的信息
    if (summaryObj.status) {
        html += `
            <div class="col-md-6">
                <h6><i class="bi bi-info-circle"></i> Experiment Status</h6>
                <div class="card bg-light">
                    <div class="card-body p-3">
                        <p><strong>Status:</strong> <span class="badge bg-success">${summaryObj.status}</span></p>
                        ${summaryObj.output_dir ? `<p><strong>Output Directory:</strong> ${summaryObj.output_dir}</p>` : ''}
                        ${summaryObj.duration ? `<p><strong>Duration:</strong> ${summaryObj.duration.toFixed(2)}s</p>` : ''}
                    </div>
                </div>
            </div>
        `;
    }
    
    // 如果有其他有用的字段
    const usefulFields = ['total_features', 'total_samples', 'significant_features', 'selected_features'];
    const availableFields = usefulFields.filter(field => summaryObj[field] !== undefined);
    
    if (availableFields.length > 0) {
        html += `
            <div class="col-md-6">
                <h6><i class="bi bi-bar-chart"></i> Key Metrics</h6>
                <div class="card bg-light">
                    <div class="card-body p-3">
                        ${availableFields.map(field => {
                            const value = summaryObj[field];
                            if (typeof value === 'number') {
                                return `<p><strong>${formatKeyName(field)}:</strong> ${value}</p>`;
                            } else if (typeof value === 'string') {
                                return `<p><strong>${formatKeyName(field)}:</strong> ${value}</p>`;
                            }
                            return '';
                        }).join('')}
                    </div>
                </div>
            </div>
        `;
    }
    
    // 如果没有有用的字段，显示原始对象
    if (!summaryObj.status && availableFields.length === 0) {
        html += `
            <div class="col-12">
                <h6><i class="bi bi-code"></i> Raw Summary Data</h6>
                <div class="card bg-light">
                    <div class="card-body p-3">
                        <pre class="mb-0" style="white-space: pre-wrap; font-size: 0.875rem;">${JSON.stringify(summaryObj, null, 2)}</pre>
                    </div>
                </div>
            </div>
        `;
    }
    
    html += '</div>';
    return html;
}

// 辅助函数
function getHealthScoreColor(score) {
    if (score >= 0.8) return 'success';
    if (score >= 0.6) return 'warning';
    return 'danger';
}

function getQualityGradeColor(grade) {
    switch (grade) {
        case 'A': return 'success';
        case 'B': return 'info';
        case 'C': return 'warning';
        case 'D': return 'warning';
        case 'F': return 'danger';
        default: return 'secondary';
    }
}

function getPerformanceColor(score) {
    if (score >= 0.8) return 'success';
    if (score >= 0.6) return 'warning';
    return 'danger';
}

function formatIssueName(issueName) {
    return issueName
        .replace('_', ' ')
        .replace(/\b\w/g, l => l.toUpperCase());
}

function displayExperimentFiles(outputFiles, resultsIndex, experimentId) {
    const filesContainer = document.getElementById('experimentFiles');
    if (!filesContainer) return;
    
    if (!outputFiles || outputFiles.length === 0) {
        filesContainer.innerHTML = '<p class="text-muted">No output files available</p>';
        return;
    }
    
    // 按类型分组文件
    const imageFiles = outputFiles.filter(f => f.type === 'image');
    const dataFiles = outputFiles.filter(f => f.type === 'csv');
    const otherFiles = outputFiles.filter(f => !['image', 'csv'].includes(f.type));
    
    let filesHtml = '';
    
    // 根据实验类型显示不同的标题和描述
    const experimentType = resultsIndex?.experiment_type || 'unknown';
    const experimentTitles = {
        'feature_statistics': {
            title: 'Feature Statistics Analysis',
            description: 'Statistical analysis of EEG features including distributions, outliers, and quality assessment'
        },
        'correlation': {
            title: 'Correlation Analysis',
            description: 'Analysis of correlations between EEG features and target variables'
        },
        'classification': {
            title: 'Classification Analysis',
            description: 'Classification tasks using EEG features with model comparison'
        }
    };
    
    const expInfo = experimentTitles[experimentType] || {
        title: 'Experiment Results',
        description: 'Analysis results and visualizations'
    };
    
    filesHtml += `
        <div class="alert alert-info mb-3">
            <h6 class="alert-heading"><i class="bi bi-info-circle"></i> ${expInfo.title}</h6>
            <p class="mb-0 small">${expInfo.description}</p>
        </div>
    `;
    
    // 显示图片文件
    if (imageFiles.length > 0) {
        filesHtml += `
            <div class="mb-4">
                <h6><i class="bi bi-images"></i> Visualizations</h6>
                <div class="row">
                    ${imageFiles.map(file => `
                        <div class="col-md-6 mb-3">
                            <div class="card">
                                <div class="card-header">
                                    <small class="text-muted">${file.name}</small>
                                </div>
                                <div class="card-body text-center">
                                    <img src="/api/experiment_image/${experimentId}/${file.path}" 
                                         class="img-fluid" style="max-height: 300px;" 
                                         alt="${file.name}"
                                         onclick="showImageModal(this.src, '${file.name}')">
                                </div>
                                <div class="card-footer">
                                    <small class="text-muted">
                                        Size: ${(file.size / 1024).toFixed(1)} KB | 
                                        Modified: ${new Date(file.modified).toLocaleString()}
                                    </small>
                                </div>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }
    
    // 显示数据文件
    if (dataFiles.length > 0) {
        filesHtml += `
            <div class="mb-4">
                <h6><i class="bi bi-table"></i> Data Files</h6>
                <div class="table-responsive">
                    <table class="table table-sm">
                        <thead>
                            <tr>
                                <th>File</th>
                                <th>Size</th>
                                <th>Modified</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${dataFiles.map(file => `
                                <tr>
                                    <td>${file.name}</td>
                                    <td>${(file.size / 1024).toFixed(1)} KB</td>
                                    <td>${new Date(file.modified).toLocaleString()}</td>
                                    <td>
                                        <button class="btn btn-sm btn-outline-primary" 
                                                onclick="viewDataFile(${experimentId}, '${file.path}')">
                                            <i class="bi bi-eye"></i> View
                                        </button>
                                        <button class="btn btn-sm btn-outline-secondary" 
                                                onclick="downloadFile(${experimentId}, '${file.path}')">
                                            <i class="bi bi-download"></i> Download
                                        </button>
                                    </td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            </div>
        `;
    }
    
    // 显示其他文件
    if (otherFiles.length > 0) {
        filesHtml += `
            <div class="mb-4">
                <h6><i class="bi bi-file-earmark"></i> Other Files</h6>
                <div class="table-responsive">
                    <table class="table table-sm">
                        <thead>
                            <tr>
                                <th>File</th>
                                <th>Size</th>
                                <th>Modified</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${otherFiles.map(file => `
                                <tr>
                                    <td>${file.name}</td>
                                    <td>${(file.size / 1024).toFixed(1)} KB</td>
                                    <td>${new Date(file.modified).toLocaleString()}</td>
                                    <td>
                                        <button class="btn btn-sm btn-outline-secondary" 
                                                onclick="downloadFile(${experimentId}, '${file.path}')">
                                            <i class="bi bi-download"></i> Download
                                        </button>
                                    </td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            </div>
        `;
    }
    
    filesContainer.innerHTML = filesHtml;
}

// 全局函数，用于图片模态框
window.showImageModal = function(imageSrc, imageName) {
    const modalHtml = `
        <div class="modal fade" id="imageModal" tabindex="-1">
            <div class="modal-dialog modal-xl">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">${imageName}</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body text-center">
                        <img src="${imageSrc}" class="img-fluid" alt="${imageName}">
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // 移除已存在的模态框
    const existingModal = document.getElementById('imageModal');
    if (existingModal) {
        existingModal.remove();
    }
    
    // 添加新模态框
    document.body.insertAdjacentHTML('beforeend', modalHtml);
    
    // 显示模态框
    const modal = new bootstrap.Modal(document.getElementById('imageModal'));
    modal.show();
};

// 全局函数，用于查看数据文件
window.viewDataFile = function(experimentId, filePath) {
    fetch(`/api/experiment_data/${experimentId}/${filePath}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showStatus(`Error: ${data.error}`, 'error');
                return;
            }
            
            let content = '';
            if (data.type === 'csv') {
                content = `
                    <h6>${data.filename} (${data.shape[0]} rows × ${data.shape[1]} columns)</h6>
                    <div class="table-responsive" style="max-height: 400px; overflow-y: auto;">
                        <table class="table table-sm table-striped">
                            <thead>
                                <tr>${data.columns.map(col => `<th>${col}</th>`).join('')}</tr>
                            </thead>
                            <tbody>
                                ${data.data.slice(0, 50).map(row => 
                                    `<tr>${data.columns.map(col => `<td>${row[col] || ''}</td>`).join('')}</tr>`
                                ).join('')}
                            </tbody>
                        </table>
                    </div>
                    ${data.data.length > 50 ? `<p class="text-muted">Showing first 50 rows of ${data.data.length} total rows</p>` : ''}
                `;
            } else {
                content = `<pre class="bg-light p-3 rounded">${data.content}</pre>`;
            }
            
            const modalHtml = `
                <div class="modal fade" id="dataModal" tabindex="-1">
                    <div class="modal-dialog modal-xl">
                        <div class="modal-content">
                            <div class="modal-header">
                                <h5 class="modal-title">${data.filename}</h5>
                                <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                            </div>
                            <div class="modal-body">
                                ${content}
                            </div>
                        </div>
                    </div>
                </div>
            `;
            
            // 移除已存在的模态框
            const existingModal = document.getElementById('dataModal');
            if (existingModal) {
                existingModal.remove();
            }
            
            // 添加新模态框
            document.body.insertAdjacentHTML('beforeend', modalHtml);
            
            // 显示模态框
            const modal = new bootstrap.Modal(document.getElementById('dataModal'));
            modal.show();
        })
        .catch(error => {
            console.error('Failed to load data file:', error);
            showStatus('Failed to load data file', 'error');
        });
};

// 全局函数，用于下载文件
window.downloadFile = function(experimentId, filePath) {
    window.open(`/api/experiment_file/${experimentId}/${filePath}`, '_blank');
};

export function showAddExperimentModal() {
    // 加载实验类型
    fetch('/api/experiment_types')
        .then(response => response.json())
        .then(types => {
            const select = document.querySelector('select[name="experiment_type"]');
            select.innerHTML = '<option value="">Select experiment type...</option>';
            types.forEach(type => {
                select.innerHTML += `<option value="${type.type}">${type.name}</option>`;
            });
        })
        .catch(error => {
            console.error('Failed to load experiment types:', error);
        });
    
    // 加载数据集
    fetch('/api/datasets')
        .then(response => response.json())
        .then(datasets => {
            const select = document.querySelector('select[name="dataset_id"]');
            select.innerHTML = '<option value="">Select dataset...</option>';
            datasets.forEach(dataset => {
                select.innerHTML += `<option value="${dataset.id}">${dataset.name}</option>`;
            });
        })
        .catch(error => {
            console.error('Failed to load datasets:', error);
        });
    
    // 加载特征集
    fetch('/api/feature_sets')
        .then(response => response.json())
        .then(featureSets => {
            const select = document.querySelector('select[name="feature_set_id"]');
            select.innerHTML = '<option value="">Select feature set...</option>';
            featureSets.forEach(fs => {
                select.innerHTML += `<option value="${fs.id}">${fs.name}</option>`;
            });
        })
        .catch(error => {
            console.error('Failed to load feature sets:', error);
        });
    
    // 显示模态框
    const modal = new bootstrap.Modal(document.getElementById('addExperimentModal'));
    modal.show();
}

// 动态生成参数表单
export function generateParameterForm(experimentType) {
    if (!experimentType) {
        document.getElementById('parameterForm').innerHTML = '<p class="text-muted">Select an experiment type to configure parameters</p>';
        return;
    }
    
    fetch(`/api/experiment_parameters/${experimentType}`)
        .then(response => response.json())
        .then(parameters => {
            let formHTML = '<div class="row">';
            
            for (const [paramName, paramConfig] of Object.entries(parameters)) {
                const required = paramConfig.required ? 'required' : '';
                const requiredMark = paramConfig.required ? '<span class="text-danger">*</span>' : '';
                
                // 为特定参数添加条件显示类
                let conditionalClass = '';
                if (paramName === 'age_threshold') {
                    conditionalClass = 'conditional-param age-dependent';
                }
                
                formHTML += `<div class="col-md-6 mb-3 ${conditionalClass}">`;
                formHTML += `<label for="${paramName}" class="form-label">${paramConfig.label} ${requiredMark}</label>`;
                
                switch (paramConfig.type) {
                    case 'select':
                        formHTML += `<select class="form-select" id="${paramName}" name="${paramName}" ${required}>`;
                        formHTML += '<option value="">Select...</option>';
                        paramConfig.options.forEach(option => {
                            const selected = option === paramConfig.default ? 'selected' : '';
                            formHTML += `<option value="${option}" ${selected}>${option}</option>`;
                        });
                        formHTML += '</select>';
                        break;
                        
                    case 'multi_select':
                        formHTML += `<select class="form-select" id="${paramName}" name="${paramName}" multiple ${required}>`;
                        paramConfig.options.forEach(option => {
                            const selected = paramConfig.default.includes(option) ? 'selected' : '';
                            formHTML += `<option value="${option}" ${selected}>${option}</option>`;
                        });
                        formHTML += '</select>';
                        break;
                        
                    case 'number':
                        formHTML += `<input type="number" class="form-control" id="${paramName}" name="${paramName}" 
                            min="${paramConfig.min}" max="${paramConfig.max}" step="${paramConfig.step}" 
                            value="${paramConfig.default}" ${required}>`;
                        break;
                        
                    case 'checkbox':
                        const checked = paramConfig.default ? 'checked' : '';
                        formHTML += `<div class="form-check">`;
                        formHTML += `<input class="form-check-input" type="checkbox" id="${paramName}" name="${paramName}" ${checked}>`;
                        formHTML += `<label class="form-check-label" for="${paramName}">${paramConfig.label}</label>`;
                        formHTML += '</div>';
                        break;
                        
                    default:
                        formHTML += `<input type="text" class="form-control" id="${paramName}" name="${paramName}" 
                            value="${paramConfig.default || ''}" ${required}>`;
                }
                
                if (paramConfig.description) {
                    formHTML += `<div class="form-text">${paramConfig.description}</div>`;
                }
                
                formHTML += '</div>';
            }
            
            formHTML += '</div>';
            document.getElementById('parameterForm').innerHTML = formHTML;
            
            // 添加动态参数显示逻辑
            setupDynamicParameterVisibility(experimentType);
        })
        .catch(error => {
            console.error('Failed to load experiment parameters:', error);
            document.getElementById('parameterForm').innerHTML = '<p class="text-danger">Failed to load parameters</p>';
        });
}

function setupDynamicParameterVisibility(experimentType) {
    // 为classification实验添加target_var变化监听
    if (experimentType === 'classification') {
        const targetVarSelect = document.getElementById('target_var');
        if (targetVarSelect) {
            targetVarSelect.addEventListener('change', function() {
                updateClassificationParameters(this.value);
            });
            // 初始化时也调用一次
            updateClassificationParameters(targetVarSelect.value);
        }
    }
    
    // 为correlation实验添加target_vars变化监听
    if (experimentType === 'correlation') {
        const targetVarsSelect = document.getElementById('target_vars');
        if (targetVarsSelect) {
            targetVarsSelect.addEventListener('change', function() {
                updateCorrelationParameters(this.value);
            });
            // 初始化时也调用一次
            updateCorrelationParameters(targetVarsSelect.value);
        }
    }
}

function updateClassificationParameters(targetVar) {
    const ageThresholdContainer = document.querySelector('.age-dependent');
    if (ageThresholdContainer) {
        if (targetVar === 'age_group' || targetVar === 'age_class') {
            // 显示年龄阈值参数
            ageThresholdContainer.classList.remove('hidden');
            const ageThresholdInput = ageThresholdContainer.querySelector('input');
            if (ageThresholdInput) {
                ageThresholdInput.required = true;
            }
        } else {
            // 隐藏年龄阈值参数
            ageThresholdContainer.classList.add('hidden');
            const ageThresholdInput = ageThresholdContainer.querySelector('input');
            if (ageThresholdInput) {
                ageThresholdInput.required = false;
                ageThresholdInput.value = ''; // 清空值
            }
        }
    }
}

function updateCorrelationParameters(targetVars) {
    // 这里可以添加correlation实验的动态参数逻辑
    // 例如根据选择的目标变量来调整其他参数
    console.log('Correlation target variables changed:', targetVars);
}

// 提交实验
export function submitExperiment() {
    const experimentType = document.querySelector('select[name="experiment_type"]').value;
    const datasetId = document.querySelector('select[name="dataset_id"]').value;
    const featureSetId = document.querySelector('select[name="feature_set_id"]').value;
    
    if (!experimentType || !datasetId || !featureSetId) {
        showStatus('Please fill in all required fields', 'error');
        return;
    }
    
    // 收集参数
    const parameters = {};
    const parameterForm = document.getElementById('parameterForm');
    const inputs = parameterForm.querySelectorAll('input, select');
    
    inputs.forEach(input => {
        // 检查参数是否被隐藏
        const paramContainer = input.closest('.conditional-param');
        if (paramContainer && paramContainer.classList.contains('hidden')) {
            // 跳过隐藏的参数
            return;
        }
        
        if (input.type === 'checkbox') {
            parameters[input.name] = input.checked;
        } else if (input.type === 'select-multiple') {
            const selectedOptions = Array.from(input.selectedOptions).map(option => option.value);
            parameters[input.name] = selectedOptions;
        } else {
            parameters[input.name] = input.value;
        }
    });
    
    // 提交实验
    fetch('/api/run_experiment', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            experiment_type: experimentType,
            dataset_id: parseInt(datasetId),
            feature_set_id: parseInt(featureSetId),
            parameters: parameters
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showStatus(`Experiment task created: ${data.message}`, 'success');
            // 关闭模态框
            const modal = bootstrap.Modal.getInstance(document.getElementById('addExperimentModal'));
            modal.hide();
            // 延迟刷新实验列表，给任务创建一些时间
            setTimeout(() => {
                loadExperiments();
            }, 1000);
        } else {
            showStatus(`Error: ${data.error}`, 'error');
        }
    })
    .catch(error => {
        console.error('Failed to submit experiment:', error);
        showStatus('Failed to submit experiment', 'error');
    });
}

// 将函数暴露到全局作用域
window.showExperimentDetails = showExperimentDetails;
window.showAddExperimentModal = showAddExperimentModal;
window.showExperimentsView = showExperimentsView;
window.showExperiments = showExperiments;
window.generateParameterForm = generateParameterForm;
window.submitExperiment = submitExperiment;
window.loadExperiments = loadExperiments;