// system-mode.js - 系统模式管理模块

/**
 * 更新系统模式显示
 */
export function updateSystemModeDisplay() {
    const statusElement = document.getElementById('systemModeStatus');
    if (!statusElement) {
        console.error('System mode status element not found');
        return;
    }

    fetch('/api/system/mode')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                const modeInfo = data.mode_info;
                const isLocalMode = modeInfo.mode === 'local';
                
                // 根据模式设置不同的样式
                const alertClass = isLocalMode ? 'alert-warning' : 'alert-info';
                const icon = isLocalMode ? 'bi-cpu' : 'bi-cloud';
                
                statusElement.className = `alert ${alertClass} alert-sm`;
                statusElement.innerHTML = `
                    <small>
                        <i class="bi ${icon}"></i> 
                        <strong>${modeInfo.mode.toUpperCase()}</strong><br>
                        <span class="text-muted">${modeInfo.description}</span>
                        ${modeInfo.workers !== 'N/A' ? `<br><small>Workers: ${modeInfo.workers}</small>` : ''}
                    </small>
                `;
            } else {
                statusElement.className = 'alert alert-danger alert-sm';
                statusElement.innerHTML = `
                    <small>
                        <i class="bi bi-exclamation-triangle"></i> 
                        Failed to load system mode
                    </small>
                `;
            }
        })
        .catch(error => {
            console.error('Failed to fetch system mode:', error);
            statusElement.className = 'alert alert-danger alert-sm';
            statusElement.innerHTML = `
                <small>
                    <i class="bi bi-exclamation-triangle"></i> 
                    Error loading system mode
                </small>
            `;
        });
}

/**
 * 初始化系统模式模块
 */
export function initializeSystemMode() {
    console.log('Initializing system mode module...');
    
    // 立即更新一次
    updateSystemModeDisplay();
    
    // 每30秒更新一次系统模式状态
    setInterval(updateSystemModeDisplay, 30000);
    
    console.log('System mode module initialized');
}

/**
 * 获取当前系统模式信息
 */
export async function getSystemMode() {
    try {
        const response = await fetch('/api/system/mode');
        const data = await response.json();
        return data.success ? data.mode_info : null;
    } catch (error) {
        console.error('Failed to get system mode:', error);
        return null;
    }
}

/**
 * Check if system is in local mode
 */
export async function isLocalMode() {
    const modeInfo = await getSystemMode();
    return modeInfo ? modeInfo.mode === 'local' : false;
}
