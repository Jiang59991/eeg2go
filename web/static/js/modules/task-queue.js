// modules/task-queue.js
import { setActiveNavButton, hideAllViews, updateBreadcrumb } from './navigation.js';
import { showStatus } from './ui-utils.js';

// 导入其他模块的函数
import { showExperimentDetails } from './experiments.js';
import { viewExtractionTask } from './feature-extraction.js';

export function initializeTaskQueue() {
    console.log('Initializing task queue module...');
}

export function showTasks() {
    console.log('showTasks function called');
    
    // 确保元素存在
    const navBtn = document.getElementById('navTasksBtn');
    const tasksView = document.getElementById('tasksView');
    const tasksTableBody = document.getElementById('tasksTableBody');
    
    if (!navBtn) {
        console.error('navTasksBtn element not found');
        return;
    }
    
    if (!tasksView) {
        console.error('tasksView element not found');
        return;
    }
    
    if (!tasksTableBody) {
        console.error('tasksTableBody element not found');
        return;
    }
    
    console.log('All elements found, proceeding with showTasks');
    
    setActiveNavButton(navBtn);
    hideAllViews();
    
    // 直接设置显示样式
    tasksView.style.display = 'block';
    tasksView.style.visibility = 'visible';
    tasksView.style.opacity = '1';
    
    updateBreadcrumb('tasks');
    
    // 立即显示加载状态
    tasksTableBody.innerHTML = `
        <tr>
            <td colspan="6" class="text-center py-3">
                <div class="spinner-border spinner-border-sm" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <span class="ms-2">Loading tasks...</span>
            </td>
        </tr>
    `;
    
    // 强制重新渲染
    setTimeout(() => {
        refreshTasks();
    }, 100);
}

export async function refreshTasks() {
    try {
        console.log('Refreshing tasks...');
        const response = await fetch('/api/tasks');
        
        if (!response.ok) {
            console.error(`HTTP error! status: ${response.status}`);
            const tbody = document.getElementById('tasksTableBody');
            if (tbody) {
                tbody.innerHTML = `
                    <tr>
                        <td colspan="6" class="text-center py-4">
                            <div class="error-state">
                                <i class="bi bi-exclamation-triangle text-danger" style="font-size: 2rem;"></i>
                                <h6 class="mt-2 text-danger">Failed to Load Tasks</h6>
                                <p class="text-muted small">HTTP Error: ${response.status}</p>
                                <button class="btn btn-outline-primary btn-sm" onclick="refreshTasks()">
                                    <i class="bi bi-arrow-clockwise"></i> Try Again
                                </button>
                            </div>
                        </td>
                    </tr>
                `;
            }
            return;
        }
        
        const data = await response.json();
        console.log('Tasks data:', data);
        
        const tbody = document.getElementById('tasksTableBody');
        if (!tbody) {
            console.error('tasksTableBody element not found in refreshTasks');
            return;
        }
        
        if (data.success) {
            if (data.tasks.length === 0) {
                // 显示友好的空状态提示 - 使用更简单的HTML结构
                tbody.innerHTML = `
                    <tr>
                        <td colspan="6" class="text-center py-5">
                            <div style="padding: 2rem 0; background-color: #f8f9fa; border-radius: 8px; margin: 1rem 0;">
                                <i class="bi bi-list-task text-muted" style="font-size: 3rem; opacity: 0.5; display: block; margin-bottom: 1rem;"></i>
                                <h5 class="text-muted" style="font-weight: 500; margin-bottom: 1rem;">No Tasks Found</h5>
                                <p class="text-muted mb-3">There are currently no tasks in the queue.</p>
                                <div class="text-muted small">
                                    <p class="mb-1">To create a task, you can:</p>
                                    <ul class="list-unstyled">
                                        <li style="margin-bottom: 0.5rem;">
                                            <i class="bi bi-arrow-right" style="margin-right: 0.5rem; font-size: 0.875rem;"></i>
                                            Go to <strong>Feature Extraction</strong> to start a feature extraction task
                                        </li>
                                        <li style="margin-bottom: 0.5rem;">
                                            <i class="bi bi-arrow-right" style="margin-right: 0.5rem; font-size: 0.875rem;"></i>
                                            Go to <strong>Experiments</strong> to run an experiment
                                        </li>
                                    </ul>
                                </div>
                            </div>
                        </td>
                    </tr>
                `;
                console.log('Displayed empty state');
                
                // 强制重新渲染
                tbody.style.display = 'table-row-group';
                tbody.style.visibility = 'visible';
                tbody.style.opacity = '1';
            } else {
                tbody.innerHTML = data.tasks.map(task => `
                    <tr>
                        <td>${task.id}</td>
                        <td>${task.task_type}</td>
                        <td>
                            <span class="badge bg-${getTaskStatusBadgeColor(task.status)}">${task.status}</span>
                        </td>
                        <td>
                            <div class="progress" style="height: 20px;">
                                <div class="progress-bar" role="progressbar" style="width: ${task.progress || 0}%">
                                    ${task.progress || 0}%
                                </div>
                            </div>
                        </td>
                        <td>${task.created_at ? new Date(task.created_at).toLocaleString() : 'N/A'}</td>
                        <td>
                            <button class="btn btn-outline-primary btn-sm" onclick="showTaskDetails(${task.id})">
                                <i class="bi bi-eye"></i> Details
                            </button>
                        </td>
                    </tr>
                `).join('');
                console.log(`Displayed ${data.tasks.length} tasks`);
            }
        } else {
            console.error('Failed to load tasks:', data.error);
            tbody.innerHTML = `
                <tr>
                    <td colspan="6" class="text-center py-4">
                        <div style="padding: 1.5rem 0; background-color: #fff3cd; border-radius: 8px; margin: 1rem 0;">
                            <i class="bi bi-exclamation-triangle text-danger" style="font-size: 2rem; opacity: 0.8; display: block; margin-bottom: 1rem;"></i>
                            <h6 class="text-danger" style="font-weight: 500; margin-bottom: 1rem;">Failed to Load Tasks</h6>
                            <p class="text-muted small mb-3">${data.error || 'Unknown error occurred'}</p>
                            <button class="btn btn-outline-primary btn-sm" onclick="refreshTasks()">
                                <i class="bi bi-arrow-clockwise"></i> Try Again
                            </button>
                        </div>
                    </td>
                </tr>
            `;
        }
    } catch (error) {
        console.error('Failed to refresh tasks:', error);
        const tbody = document.getElementById('tasksTableBody');
        if (tbody) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="6" class="text-center py-4">
                        <div style="padding: 1.5rem 0; background-color: #f8d7da; border-radius: 8px; margin: 1rem 0;">
                            <i class="bi bi-exclamation-triangle text-danger" style="font-size: 2rem; opacity: 0.8; display: block; margin-bottom: 1rem;"></i>
                            <h6 class="text-danger" style="font-weight: 500; margin-bottom: 1rem;">Failed to Load Tasks</h6>
                            <p class="text-muted small mb-3">Network error or server issue</p>
                            <button class="btn btn-outline-primary btn-sm" onclick="refreshTasks()">
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

export function showTaskDetails(taskId) {
    console.log('Showing task details for ID:', taskId);
    
    fetch(`/api/task_details/${taskId}`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                const task = data.task;
                const taskType = task.task_type;
                
                console.log(`Task type: ${taskType}, routing to appropriate detail view`);
                
                // 根据任务类型路由到对应的详情页面
                switch (taskType) {
                    case 'experiment':
                        // 调用实验详情函数
                        showExperimentDetails(taskId);
                        break;
                        
                    case 'feature_extraction':
                        // 调用特征提取详情函数
                        viewExtractionTask(taskId);
                        break;
                        
                    default:
                        // 对于其他类型的任务，显示通用的任务详情模态框
                        showGenericTaskDetails(task);
                        break;
                }
            } else {
                showStatus(`Error: ${data.error}`, 'error');
            }
        })
        .catch(error => {
            console.error('Failed to load task details:', error);
            showStatus('Failed to load task details', 'error');
        });
}

// 通用任务详情显示函数（用于非experiment/feature_extraction任务）
function showGenericTaskDetails(task) {
    // 填充任务详情
    document.getElementById('taskDetailId').textContent = task.id;
    document.getElementById('taskDetailType').textContent = task.task_type;
    document.getElementById('taskDetailStatus').textContent = task.status;
    document.getElementById('taskDetailCreated').textContent = task.created_at ? new Date(task.created_at).toLocaleString() : 'N/A';
    document.getElementById('taskDetailStarted').textContent = task.started_at ? new Date(task.started_at).toLocaleString() : 'N/A';
    document.getElementById('taskDetailCompleted').textContent = task.completed_at ? new Date(task.completed_at).toLocaleString() : 'N/A';
    
    // 参数和结果
    document.getElementById('taskDetailParams').textContent = task.parameters ? JSON.stringify(task.parameters, null, 2) : 'N/A';
    document.getElementById('taskDetailResult').textContent = task.result ? JSON.stringify(task.result, null, 2) : 'N/A';
    
    // 错误信息
    if (task.error_message) {
        document.getElementById('taskDetailError').style.display = 'block';
        document.getElementById('taskDetailErrorMessage').textContent = task.error_message;
    } else {
        document.getElementById('taskDetailError').style.display = 'none';
    }
    
    // 显示模态框
    const modal = new bootstrap.Modal(document.getElementById('taskDetailsModal'));
    modal.show();
}

// 导出到全局作用域
window.showTasks = showTasks;
window.refreshTasks = refreshTasks;
window.showTaskDetails = showTaskDetails;