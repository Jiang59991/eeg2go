// modules/ui-utils.js

export function showStatus(message, type = 'info') {
    const statusDiv = document.getElementById('statusMessage');
    if (!statusDiv) return;
    
    statusDiv.textContent = message;
    statusDiv.className = `alert alert-${type === 'error' ? 'danger' : type} alert-dismissible fade show`;
    statusDiv.style.display = 'block';
    
    setTimeout(() => {
        statusDiv.style.display = 'none';
    }, 5000);
}

export function showProgress(show) {
    const progressDiv = document.getElementById('progressIndicator');
    if (progressDiv) {
        progressDiv.style.display = show ? 'block' : 'none';
    }
}

export function getStatusBadge(status) {
    const statusMap = {
        'pending': 'secondary',
        'running': 'primary',
        'completed': 'success',
        'failed': 'danger'
    };
    
    const color = statusMap[status] || 'secondary';
    return `<span class="badge bg-${color}">${status}</span>`;
}

export function getTypeBadge(type) {
    const typeMap = {
        'classification': 'info',
        'correlation': 'warning',
        'feature_selection': 'success',
        'feature_statistics': 'primary'
    };
    
    const color = typeMap[type] || 'secondary';
    return `<span class="badge bg-${color}">${type}</span>`;
}