// modules/datasets.js
import { setActiveNavButton, hideAllViews, updateBreadcrumb } from './navigation.js';
import { showStatus } from './ui-utils.js';

export function initializeDatasets() {
    console.log('Initializing datasets module...');
    
    // 初始化全局变量
    if (!window.selectedRecordings) {
        window.selectedRecordings = new Set();
    }
    if (!window.currentDatasetId) {
        window.currentDatasetId = null;
    }
    if (!window.currentDatasetName) {
        window.currentDatasetName = null;
    }
}

export async function showDatasets() {
    console.log('showDatasets called');
    
    const navBtn = document.getElementById('navDatasetsBtn');
    if (navBtn) {
        setActiveNavButton(navBtn);
    }
    
    hideAllViews();
    
    const datasetsView = document.getElementById('datasetsView');
    if (datasetsView) {
        datasetsView.style.display = 'block';
    } else {
        console.error('datasetsView element not found');
    }
    
    updateBreadcrumb('datasets');
    
    // Clear current dataset - 添加安全检查
    window.currentDatasetId = null;
    window.currentDatasetName = null;
    if (window.selectedRecordings && typeof window.selectedRecordings.clear === 'function') {
        window.selectedRecordings.clear();
    } else {
        window.selectedRecordings = new Set();
    }
    
    await loadDatasets();
}

export async function loadDatasets() {
    try {
        console.log('Loading datasets...');
        const response = await fetch('/api/datasets');
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const datasets = await response.json();
        console.log('Datasets loaded:', datasets);
        
        displayDatasets(datasets);
    } catch (error) {
        console.error('Error loading datasets:', error);
        showStatus('Error loading datasets', 'error');
    }
}

export function displayDatasets(datasets) {
    const grid = document.getElementById('datasetsGrid');
    if (!grid) {
        console.error('datasetsGrid element not found');
        return;
    }
    
    console.log('Displaying datasets:', datasets);
    
    if (datasets.length === 0) {
        grid.innerHTML = '<div class="col-12"><p class="text-center text-muted">No datasets found</p></div>';
        return;
    }
    
    grid.innerHTML = '';
    
    datasets.forEach(dataset => {
        const col = document.createElement('div');
        col.className = 'col-lg-4 col-md-6 col-sm-12 mb-3';
        
        col.innerHTML = `
            <div class="dataset-card" onclick="showRecordings(${dataset.id}, '${dataset.name}')">
                <div class="d-flex align-items-center mb-2">
                    <i class="bi bi-database text-primary me-2"></i>
                    <h5 class="mb-0">${dataset.name}</h5>
                </div>
                <div class="dataset-description">
                    ${dataset.description || 'No description available'}
                </div>
                <div class="dataset-stats">
                    <span><i class="bi bi-file-earmark"></i> Source: ${dataset.source_type || 'Unknown'}</span>
                </div>
            </div>
        `;
        
        grid.appendChild(col);
    });
}

export async function showRecordings(datasetId, datasetName) {
    console.log('showRecordings called:', datasetId, datasetName);
    
    // 确保全局变量已初始化
    if (!window.selectedRecordings) {
        window.selectedRecordings = new Set();
    }
    
    window.currentDatasetId = datasetId;
    window.currentDatasetName = datasetName;
    
    hideAllViews();
    
    const recordingsView = document.getElementById('recordingsView');
    if (recordingsView) {
        recordingsView.style.display = 'block';
    }
    
    // Update breadcrumb
    const breadcrumbRecordings = document.getElementById('breadcrumb-recordings');
    if (breadcrumbRecordings) {
        breadcrumbRecordings.style.display = 'block';
    }
    
    const breadcrumbRecordingsLink = document.getElementById('breadcrumb-recordings-link');
    if (breadcrumbRecordingsLink) {
        breadcrumbRecordingsLink.textContent = datasetName;
    }
    
    // Update title
    const recordingsTitle = document.getElementById('recordingsTitle');
    if (recordingsTitle) {
        recordingsTitle.textContent = `Recordings - ${datasetName}`;
    }
    
    // Load recordings for this dataset
    await loadRecordings(datasetId);
}

export async function loadRecordings(datasetId) {
    try {
        console.log('Loading recordings for dataset:', datasetId);
        showStatus('Loading recordings...', 'info');
        
        const url = `/api/recordings?dataset_id=${datasetId}`;
        const response = await fetch(url);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const recordings = await response.json();
        console.log('Recordings loaded:', recordings);
        
        displayRecordings(recordings);
        showStatus(`Loaded ${recordings.length} recordings`, 'success');
    } catch (error) {
        console.error('Error loading recordings:', error);
        showStatus('Error loading recordings', 'error');
    }
}

export function displayRecordings(recordings) {
    const tbody = document.getElementById('recordingsTableBody');
    if (!tbody) {
        console.error('recordingsTableBody element not found');
        return;
    }
    
    if (recordings.length === 0) {
        tbody.innerHTML = '<tr><td colspan="10" class="text-center text-muted">No recordings found in this dataset</td></tr>';
        return;
    }
    
    tbody.innerHTML = '';
    
    recordings.forEach(recording => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td><input type="checkbox" class="recording-checkbox" value="${recording.id}"></td>
            <td>${recording.id}</td>
            <td>${recording.subject_id || '-'}</td>
            <td>${recording.filename || '-'}</td>
            <td>${recording.duration ? recording.duration.toFixed(2) : '-'}</td>
            <td>${recording.channels || '-'}</td>
            <td>${recording.sampling_rate ? recording.sampling_rate.toFixed(0) : '-'}</td>
            <td>${recording.age || '-'}</td>
            <td>${recording.sex || '-'}</td>
            <td>
                <button class="btn btn-outline-primary btn-sm" onclick="viewFeatureValues(${recording.id})">
                    <i class="bi bi-eye"></i> View Features
                </button>
            </td>
        `;
        
        // Add event listener to checkbox
        const checkbox = row.querySelector('.recording-checkbox');
        checkbox.addEventListener('change', function() {
            const recordingId = parseInt(this.value);
            if (this.checked) {
                window.selectedRecordings.add(recordingId);
            } else {
                window.selectedRecordings.delete(recordingId);
            }
            updateButtonStates();
        });
        
        tbody.appendChild(row);
    });
}

export function selectAllRecordings() {
    const checkboxes = document.querySelectorAll('.recording-checkbox');
    checkboxes.forEach(checkbox => {
        checkbox.checked = true;
        window.selectedRecordings.add(parseInt(checkbox.value));
    });
    document.getElementById('selectAllCheckbox').checked = true;
    updateButtonStates();
}

export function clearSelection() {
    const checkboxes = document.querySelectorAll('.recording-checkbox');
    checkboxes.forEach(checkbox => {
        checkbox.checked = false;
    });
    document.getElementById('selectAllCheckbox').checked = false;
    if (window.selectedRecordings && typeof window.selectedRecordings.clear === 'function') {
        window.selectedRecordings.clear();
    } else {
        window.selectedRecordings = new Set();
    }
    updateButtonStates();
}

export function updateButtonStates() {
    // Button states are now handled by individual views
}

// 导出到全局作用域
window.showDatasets = showDatasets;
window.showRecordings = showRecordings;
window.selectAllRecordings = selectAllRecordings;
window.clearSelection = clearSelection;