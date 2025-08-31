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
            
            // 填充实验摘要统计
            const statsContainer = document.getElementById('experimentStats');
            if (data.results_index && data.results_index.summary) {
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
            
            // 填充特征结果表格
            const resultsBody = document.getElementById('experimentResultsBody');
            if (data.feature_results && data.feature_results.length > 0) {
                resultsBody.innerHTML = data.feature_results.map(result => `
                    <tr>
                        <td>${result.rank_position || 'N/A'}</td>
                        <td>${result.feature_shortname || 'N/A'}</td>
                        <td>${result.feature_channels || 'N/A'}</td>
                        <td>${result.target_variable || 'N/A'}</td>
                        <td>${result.metric_name || 'N/A'}</td>
                        <td>${result.metric_value ? result.metric_value.toFixed(4) : 'N/A'}</td>
                        <td>${result.significance_level || 'N/A'}</td>
                    </tr>
                `).join('');
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
        'feature_selection': {
            title: 'Feature Selection',
            description: 'Selection of most important features using multiple methods'
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