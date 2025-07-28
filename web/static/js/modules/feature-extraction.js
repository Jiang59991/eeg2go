// modules/feature-extraction.js
import { setActiveNavButton, hideAllViews, updateBreadcrumb } from './navigation.js';
import { showStatus } from './ui-utils.js';

export function initializeFeatureExtraction() {
    console.log('Initializing feature extraction module...');
}

export function showFeatureExtraction() {
    setActiveNavButton(document.getElementById('navFeatureExtractionBtn'));
    hideAllViews();
    document.getElementById('featureExtractionView').style.display = 'block';
    updateBreadcrumb('feature-extraction');
    
    loadFeatureExtractionData();
}

async function loadFeatureExtractionData() {
    try {
        // 加载数据集
        const datasetsResponse = await fetch('/api/datasets');
        const datasets = await datasetsResponse.json();
        
        const datasetSelect = document.getElementById('extractionDatasetSelect');
        datasetSelect.innerHTML = '<option value="">Choose dataset...</option>';
        datasets.forEach(dataset => {
            datasetSelect.innerHTML += `<option value="${dataset.id}">${dataset.name}</option>`;
        });
        
        // 加载特征集
        const featureSetsResponse = await fetch('/api/feature_sets');
        const featureSets = await featureSetsResponse.json();
        
        const featureSetSelect = document.getElementById('extractionFeatureSetSelect');
        featureSetSelect.innerHTML = '<option value="">Choose feature set...</option>';
        featureSets.forEach(fs => {
            featureSetSelect.innerHTML += `<option value="${fs.id}">${fs.name}</option>`;
        });
        
        // 加载提取任务
        await loadExtractionTasks();
        
    } catch (error) {
        console.error('Failed to load feature extraction data:', error);
        showStatus('Failed to load feature extraction data', 'error');
    }
}

export async function loadExtractionTasks() {
    try {
        console.log('Loading extraction tasks...');
        const response = await fetch('/api/feature_extraction_tasks');
        
        if (!response.ok) {
            console.error(`HTTP error! status: ${response.status}`);
            const tbody = document.getElementById('extractionTableBody');
            if (tbody) {
                tbody.innerHTML = `
                    <tr>
                        <td colspan="7" class="text-center py-4">
                            <div style="padding: 1.5rem 0; background-color: #f8d7da; border-radius: 8px; margin: 1rem 0;">
                                <i class="bi bi-exclamation-triangle text-danger" style="font-size: 2rem; opacity: 0.8; display: block; margin-bottom: 1rem;"></i>
                                <h6 class="text-danger" style="font-weight: 500; margin-bottom: 1rem;">Failed to Load Tasks</h6>
                                <p class="text-muted small mb-3">HTTP Error: ${response.status}</p>
                                <button class="btn btn-outline-primary btn-sm" onclick="loadExtractionTasks()">
                                    <i class="bi bi-arrow-clockwise"></i> Try Again
                                </button>
                            </div>
                        </td>
                    </tr>
                `;
            }
            return;
        }
        
        const tasks = await response.json();
        console.log('Extraction tasks loaded:', tasks);
        
        const tbody = document.getElementById('extractionTableBody');
        if (!tbody) {
            console.error('extractionTableBody element not found');
            return;
        }
        
        if (tasks.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="7" class="text-center py-5">
                        <div style="padding: 2rem 0; background-color: #f8f9fa; border-radius: 8px; margin: 1rem 0;">
                            <i class="bi bi-list-task text-muted" style="font-size: 3rem; opacity: 0.5; display: block; margin-bottom: 1rem;"></i>
                            <h5 class="text-muted" style="font-weight: 500; margin-bottom: 1rem;">No Extraction Tasks Found</h5>
                            <p class="text-muted mb-3">There are currently no feature extraction tasks.</p>
                            <div class="text-muted small">
                                <p class="mb-1">To start a feature extraction task:</p>
                                <ul class="list-unstyled">
                                    <li style="margin-bottom: 0.5rem;">
                                        <i class="bi bi-arrow-right" style="margin-right: 0.5rem; font-size: 0.875rem;"></i>
                                        Select a dataset and feature set above
                                    </li>
                                    <li style="margin-bottom: 0.5rem;">
                                        <i class="bi bi-arrow-right" style="margin-right: 0.5rem; font-size: 0.875rem;"></i>
                                        Click "Start Feature Extraction"
                                    </li>
                                </ul>
                            </div>
                        </div>
                    </td>
                </tr>
            `;
            console.log('Displayed empty state for extraction tasks');
        } else {
            tbody.innerHTML = tasks.map(task => `
                <tr>
                    <td>${task.id}</td>
                    <td>${task.dataset_name || 'N/A'}</td>
                    <td>${task.feature_set_name || 'N/A'}</td>
                    <td>
                        <span class="badge bg-${getTaskStatusBadgeColor(task.status)}">${task.status}</span>
                    </td>
                    <td>${task.created_at ? new Date(task.created_at).toLocaleString() : 'N/A'}</td>
                    <td>${task.duration_seconds ? `${task.duration_seconds}s` : 'N/A'}</td>
                    <td>
                        <button class="btn btn-outline-primary btn-sm" onclick="viewExtractionTask(${task.id})">
                            <i class="bi bi-eye"></i> View
                        </button>
                    </td>
                </tr>
            `).join('');
            console.log(`Displayed ${tasks.length} extraction tasks`);
        }
        
    } catch (error) {
        console.error('Failed to load extraction tasks:', error);
        const tbody = document.getElementById('extractionTableBody');
        if (tbody) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="7" class="text-center py-4">
                        <div style="padding: 1.5rem 0; background-color: #f8d7da; border-radius: 8px; margin: 1rem 0;">
                            <i class="bi bi-exclamation-triangle text-danger" style="font-size: 2rem; opacity: 0.8; display: block; margin-bottom: 1rem;"></i>
                            <h6 class="text-danger" style="font-weight: 500; margin-bottom: 1rem;">Failed to Load Tasks</h6>
                            <p class="text-muted small mb-3">Network error or server issue</p>
                            <button class="btn btn-outline-primary btn-sm" onclick="loadExtractionTasks()">
                                <i class="bi bi-arrow-clockwise"></i> Try Again
                            </button>
                        </div>
                    </td>
                </tr>
            `;
        }
    }
}

function getTaskStatusBadgeColor(status) {
    switch (status) {
        case 'completed': return 'success';
        case 'running': return 'warning';
        case 'failed': return 'danger';
        case 'pending': return 'secondary';
        default: return 'secondary';
    }
}

export function startFeatureExtraction() {
    const datasetId = document.getElementById('extractionDatasetSelect').value;
    const featureSetId = document.getElementById('extractionFeatureSetSelect').value;
    
    if (!datasetId || !featureSetId) {
        showStatus('Please select both dataset and feature set', 'error');
        return;
    }
    
    fetch('/api/start_feature_extraction', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            dataset_id: parseInt(datasetId),
            feature_set_id: parseInt(featureSetId)
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.task_id) {
            showStatus(`Feature extraction task started (ID: ${data.task_id})`, 'success');
            // 延迟刷新任务列表，给任务创建一些时间
            setTimeout(() => {
                loadExtractionTasks();
            }, 1000);
        } else {
            showStatus(`Error: ${data.error}`, 'error');
        }
    })
    .catch(error => {
        console.error('Failed to start feature extraction:', error);
        showStatus('Failed to start feature extraction', 'error');
    });
}

// 全局变量存储当前查看的任务ID
let currentExtractionTaskId = null;

export function viewExtractionTask(taskId) {
    console.log('Viewing extraction task:', taskId);
    currentExtractionTaskId = taskId;
    
    // 先显示加载状态
    showExtractionTaskLoadingState();
    
    // 显示模态框
    const modal = new bootstrap.Modal(document.getElementById('extractionTaskModal'));
    modal.show();
    
    // 加载任务详情
    loadExtractionTaskDetails(taskId);
}

function showExtractionTaskLoadingState() {
    // 显示加载状态
    document.getElementById('extractionTaskId').innerHTML = '<div class="spinner-border spinner-border-sm" role="status"></div> Loading...';
    document.getElementById('extractionTaskType').innerHTML = '<div class="spinner-border spinner-border-sm" role="status"></div> Loading...';
    document.getElementById('extractionTaskStatus').innerHTML = '<div class="spinner-border spinner-border-sm" role="status"></div> Loading...';
    document.getElementById('extractionTaskDataset').innerHTML = '<div class="spinner-border spinner-border-sm" role="status"></div> Loading...';
    document.getElementById('extractionTaskFeatureSet').innerHTML = '<div class="spinner-border spinner-border-sm" role="status"></div> Loading...';
    document.getElementById('extractionTaskCreated').innerHTML = '<div class="spinner-border spinner-border-sm" role="status"></div> Loading...';
    document.getElementById('extractionTaskStarted').innerHTML = '<div class="spinner-border spinner-border-sm" role="status"></div> Loading...';
    document.getElementById('extractionTaskCompleted').innerHTML = '<div class="spinner-border spinner-border-sm" role="status"></div> Loading...';
    document.getElementById('extractionTaskDuration').innerHTML = '<div class="spinner-border spinner-border-sm" role="status"></div> Loading...';
    
    // 进度部分
    document.getElementById('extractionTaskProgressText').innerHTML = '<div class="spinner-border spinner-border-sm" role="status"></div> Loading...';
    document.getElementById('extractionTaskProgressBar').innerHTML = '<div class="spinner-border spinner-border-sm" role="status"></div>';
    document.getElementById('extractionTaskProcessed').innerHTML = '<div class="spinner-border spinner-border-sm" role="status"></div> Loading...';
    document.getElementById('extractionTaskTotal').innerHTML = '<div class="spinner-border spinner-border-sm" role="status"></div> Loading...';
    document.getElementById('extractionTaskFailed').innerHTML = '<div class="spinner-border spinner-border-sm" role="status"></div> Loading...';
    
    // 参数部分
    document.getElementById('extractionTaskParams').innerHTML = '<div class="spinner-border spinner-border-sm" role="status"></div> Loading task parameters...';
    
    // 隐藏其他部分
    document.getElementById('extractionTaskResultSection').style.display = 'none';
    document.getElementById('extractionTaskErrorSection').style.display = 'none';
    document.getElementById('extractionTaskNotesSection').style.display = 'none';
    document.getElementById('extractionTaskDownloadBtn').style.display = 'none';
}

async function loadExtractionTaskDetails(taskId) {
    try {
        console.log('Loading extraction task details for ID:', taskId);
        const response = await fetch(`/api/feature_extraction_status/${taskId}`);
        
        if (!response.ok) {
            console.error(`HTTP error! status: ${response.status}`);
            showStatus(`Failed to load task details: HTTP ${response.status}`, 'error');
            showExtractionTaskErrorState(`HTTP Error: ${response.status}`);
            return;
        }
        
        const task = await response.json();
        console.log('Task details loaded:', task);
        
        // 检查任务数据是否为空
        if (!task || Object.keys(task).length === 0) {
            console.error('Task data is empty or null');
            showStatus('Task data is empty', 'error');
            showExtractionTaskErrorState('Task data is empty');
            return;
        }
        
        // 填充任务信息
        document.getElementById('extractionTaskId').textContent = task.id || 'N/A';
        document.getElementById('extractionTaskType').textContent = task.task_type || 'N/A';
        document.getElementById('extractionTaskStatus').innerHTML = `<span class="badge bg-${getTaskStatusBadgeColor(task.status)}">${task.status || 'N/A'}</span>`;
        document.getElementById('extractionTaskDataset').textContent = task.dataset_name || 'N/A';
        document.getElementById('extractionTaskFeatureSet').textContent = task.feature_set_name || 'N/A';
        document.getElementById('extractionTaskCreated').textContent = task.created_at ? new Date(task.created_at).toLocaleString() : 'N/A';
        document.getElementById('extractionTaskStarted').textContent = task.started_at ? new Date(task.started_at).toLocaleString() : 'N/A';
        document.getElementById('extractionTaskCompleted').textContent = task.completed_at ? new Date(task.completed_at).toLocaleString() : 'N/A';
        document.getElementById('extractionTaskDuration').textContent = task.duration ? `${task.duration.toFixed(1)}s` : 'N/A';
        
        // 填充进度信息
        const progress = task.progress || 0;
        document.getElementById('extractionTaskProgressText').textContent = `${progress.toFixed(1)}%`;
        document.getElementById('extractionTaskProgressBar').style.width = `${progress}%`;
        document.getElementById('extractionTaskProgressBar').textContent = `${progress.toFixed(1)}%`;
        document.getElementById('extractionTaskProcessed').textContent = task.processed_count || 0;
        document.getElementById('extractionTaskTotal').textContent = task.total_count || 0;
        document.getElementById('extractionTaskFailed').textContent = task.failed_count || 0;
        
        // 填充参数
        const params = task.parameters ? JSON.stringify(task.parameters, null, 2) : 'No parameters';
        document.getElementById('extractionTaskParams').textContent = params;
        
        // 处理结果
        if (task.result && task.status === 'completed') {
            document.getElementById('extractionTaskResultSection').style.display = 'block';
            const result = JSON.stringify(task.result, null, 2);
            document.getElementById('extractionTaskResult').textContent = result;
            
            // 显示下载按钮
            document.getElementById('extractionTaskDownloadBtn').style.display = 'inline-block';
        } else {
            document.getElementById('extractionTaskResultSection').style.display = 'none';
            document.getElementById('extractionTaskDownloadBtn').style.display = 'none';
        }
        
        // 处理错误信息
        if (task.error_message) {
            document.getElementById('extractionTaskErrorSection').style.display = 'block';
            document.getElementById('extractionTaskError').textContent = task.error_message;
        } else {
            document.getElementById('extractionTaskErrorSection').style.display = 'none';
        }
        
        // 处理备注
        if (task.notes) {
            document.getElementById('extractionTaskNotesSection').style.display = 'block';
            document.getElementById('extractionTaskNotes').textContent = task.notes;
        } else {
            document.getElementById('extractionTaskNotesSection').style.display = 'none';
        }
        
        console.log('Task details populated successfully');
        
    } catch (error) {
        console.error('Failed to load extraction task details:', error);
        showStatus('Failed to load task details', 'error');
        showExtractionTaskErrorState('Network error or server issue');
    }
}

function showExtractionTaskErrorState(errorMessage) {
    // 显示错误状态
    document.getElementById('extractionTaskId').textContent = 'Error';
    document.getElementById('extractionTaskType').textContent = 'Error';
    document.getElementById('extractionTaskStatus').innerHTML = '<span class="badge bg-danger">Error</span>';
    document.getElementById('extractionTaskDataset').textContent = 'Error';
    document.getElementById('extractionTaskFeatureSet').textContent = 'Error';
    document.getElementById('extractionTaskCreated').textContent = 'Error';
    document.getElementById('extractionTaskStarted').textContent = 'Error';
    document.getElementById('extractionTaskCompleted').textContent = 'Error';
    document.getElementById('extractionTaskDuration').textContent = 'Error';
    
    // 进度部分
    document.getElementById('extractionTaskProgressText').textContent = 'Error';
    document.getElementById('extractionTaskProgressBar').style.width = '0%';
    document.getElementById('extractionTaskProgressBar').textContent = 'Error';
    document.getElementById('extractionTaskProcessed').textContent = 'Error';
    document.getElementById('extractionTaskTotal').textContent = 'Error';
    document.getElementById('extractionTaskFailed').textContent = 'Error';
    
    // 参数部分
    document.getElementById('extractionTaskParams').textContent = `Error loading task details: ${errorMessage}`;
    
    // 隐藏其他部分
    document.getElementById('extractionTaskResultSection').style.display = 'none';
    document.getElementById('extractionTaskErrorSection').style.display = 'none';
    document.getElementById('extractionTaskNotesSection').style.display = 'none';
    document.getElementById('extractionTaskDownloadBtn').style.display = 'none';
}

export function refreshExtractionTaskDetails() {
    if (currentExtractionTaskId) {
        showExtractionTaskLoadingState();
        loadExtractionTaskDetails(currentExtractionTaskId);
    }
}

export function downloadExtractionResult() {
    if (currentExtractionTaskId) {
        window.open(`/api/download_extraction_result/${currentExtractionTaskId}`, '_blank');
    }
}

// 导出到全局作用域
window.showFeatureExtraction = showFeatureExtraction;
window.startFeatureExtraction = startFeatureExtraction;
window.viewExtractionTask = viewExtractionTask;
window.loadExtractionTasks = loadExtractionTasks;
window.refreshExtractionTaskDetails = refreshExtractionTaskDetails;
window.downloadExtractionResult = downloadExtractionResult;