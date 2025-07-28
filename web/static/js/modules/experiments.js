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
                        <button class="btn btn-sm btn-outline-primary" onclick="showExperimentDetails('${exp.id}')">
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
        .then(response => {
            console.log('Response status:', response.status);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('Received data:', data);
            
            if (data.error) {
                console.error('API returned error:', data.error);
                showStatus(`Error: ${data.error}`, 'error');
                return;
            }
            
            if (!data.experiment) {
                console.error('No experiment data in response');
                showStatus('No experiment data received', 'error');
                return;
            }
            
            const experiment = data.experiment;
            console.log('Experiment data:', experiment);
            
            // 填充实验信息
            let infoHTML = `
                <table class="table table-sm">
                    <tr><td><strong>ID:</strong></td><td>${experiment.id || 'N/A'}</td></tr>
                    <tr><td><strong>Name:</strong></td><td>${experiment.experiment_name || 'N/A'}</td></tr>
                    <tr><td><strong>Type:</strong></td><td>${experiment.experiment_type || 'N/A'}</td></tr>
                    <tr><td><strong>Dataset:</strong></td><td>${experiment.dataset_name || 'N/A'}</td></tr>
                    <tr><td><strong>Feature Set:</strong></td><td>${experiment.feature_set_name || 'N/A'}</td></tr>
                    <tr><td><strong>Status:</strong></td><td><span class="badge bg-${getStatusBadgeColor(experiment.status)}">${experiment.status || 'N/A'}</span></td></tr>
                    <tr><td><strong>Run Time:</strong></td><td>${experiment.run_time ? new Date(experiment.run_time).toLocaleString() : 'N/A'}</td></tr>
                    <tr><td><strong>Duration:</strong></td><td>${experiment.duration_seconds ? `${experiment.duration_seconds}s` : 'N/A'}</td></tr>
            `;
            
            // 如果是任务，显示额外信息
            if (experiment.is_task) {
                console.log('This is a task, showing additional info');
                infoHTML += `
                    <tr><td><strong>Progress:</strong></td><td>${experiment.progress || 0}%</td></tr>
                    <tr><td><strong>Processed:</strong></td><td>${experiment.processed_count || 0} / ${experiment.total_count || 0}</td></tr>
                `;
                
                if (experiment.notes) {
                    infoHTML += `<tr><td><strong>Notes:</strong></td><td>${experiment.notes}</td></tr>`;
                }
                
                if (experiment.error_message) {
                    infoHTML += `<tr><td><strong>Error:</strong></td><td class="text-danger">${experiment.error_message}</td></tr>`;
                }
            }
            
            infoHTML += '</table>';
            console.log('Generated info HTML:', infoHTML);
            
            const experimentInfo = document.getElementById('experimentInfo');
            if (experimentInfo) {
                console.log('Found experimentInfo element, setting innerHTML');
                experimentInfo.innerHTML = infoHTML;
            } else {
                console.error('experimentInfo element not found');
            }
            
            // 填充特征结果（只有实验结果才有）
            const resultsBody = document.getElementById('experimentResultsBody');
            const resultsSection = document.getElementById('experimentResultsSection');
            if (data.feature_results && data.feature_results.length > 0 && resultsBody && resultsSection) {
                console.log('Showing feature results');
                resultsBody.innerHTML = data.feature_results.map(result => `
                    <tr>
                        <td>${result.rank_position || 'N/A'}</td>
                        <td>${result.feature_shortname || 'N/A'}</td>
                        <td>${result.feature_channels || 'N/A'}</td>
                        <td>${result.metric_name || 'N/A'}</td>
                        <td>${result.metric_value || 'N/A'}</td>
                        <td>${result.p_value || 'N/A'}</td>
                    </tr>
                `).join('');
                resultsSection.style.display = 'block';
            } else if (resultsSection) {
                console.log('Hiding feature results section');
                resultsSection.style.display = 'none';
            }
            
            // 填充元数据
            const metadataBody = document.getElementById('experimentMetadataBody');
            const metadataSection = document.getElementById('experimentMetadataSection');
            if (data.metadata && data.metadata.length > 0 && metadataBody && metadataSection) {
                console.log('Showing metadata');
                metadataBody.innerHTML = data.metadata.map(md => `
                    <tr>
                        <td>${md.key}</td>
                        <td>${md.value}</td>
                        <td>${md.value_type}</td>
                    </tr>
                `).join('');
                metadataSection.style.display = 'block';
            } else if (metadataSection) {
                console.log('Hiding metadata section');
                metadataSection.style.display = 'none';
            }
            
            // 填充输出文件
            const filesBody = document.getElementById('experimentFilesBody');
            const filesSection = document.getElementById('experimentFilesSection');
            if (data.output_files && data.output_files.length > 0 && filesBody && filesSection) {
                console.log('Showing output files');
                filesBody.innerHTML = data.output_files.map(file => `
                    <tr>
                        <td>${file.name}</td>
                        <td>${(file.size / 1024).toFixed(2)} KB</td>
                        <td>${file.type}</td>
                        <td>${new Date(file.modified).toLocaleString()}</td>
                        <td>
                            <a href="/api/experiment_file/${experiment.id}/${file.name}" 
                               class="btn btn-sm btn-outline-primary" target="_blank">
                                <i class="bi bi-download"></i> Download
                            </a>
                        </td>
                    </tr>
                `).join('');
                filesSection.style.display = 'block';
            } else if (filesSection) {
                console.log('Hiding output files section');
                filesSection.style.display = 'none';
            }
            
            // 显示模态框
            const modalElement = document.getElementById('experimentDetailsModal');
            if (modalElement) {
                console.log('Found modal element, showing modal');
                const modal = new bootstrap.Modal(modalElement);
                modal.show();
            } else {
                console.error('experimentDetailsModal element not found');
                showStatus('Modal element not found', 'error');
            }
        })
        .catch(error => {
            console.error('Failed to load experiment details:', error);
            showStatus('Failed to load experiment details', 'error');
        });
}

// 添加刷新函数
export function refreshExperimentDetails() {
    // 这里可以重新加载当前实验的详情
    // 暂时只是关闭模态框
    const modalElement = document.getElementById('experimentDetailsModal');
    if (modalElement) {
        const modal = bootstrap.Modal.getInstance(modalElement);
        if (modal) {
            modal.hide();
        }
    }
}

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
                
                formHTML += '<div class="col-md-6 mb-3">';
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
        })
        .catch(error => {
            console.error('Failed to load experiment parameters:', error);
            document.getElementById('parameterForm').innerHTML = '<p class="text-danger">Failed to load parameters</p>';
        });
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
window.refreshExperimentDetails = refreshExperimentDetails;