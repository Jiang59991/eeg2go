// Global variables
let selectedRecordings = new Set();
let currentDatasetId = null;
let currentFeatureSetId = null;
let currentDatasetName = null;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    loadDatasets();
    loadFeatureSets();
    setupEventListeners();
    showDatasets(); // Start with datasets view
});

// Setup event listeners
function setupEventListeners() {
    // Feature set selection
    document.getElementById('featureSetSelect').addEventListener('change', function() {
        currentFeatureSetId = this.value;
        document.getElementById('viewFeatureSetBtn').disabled = !currentFeatureSetId;
        updateButtonStates();
    });

    // View feature set details
    document.getElementById('viewFeatureSetBtn').addEventListener('click', function() {
        if (currentFeatureSetId) {
            showFeatureSetDetails(currentFeatureSetId);
        }
    });

    // Extract features
    document.getElementById('extractBtn').addEventListener('click', function() {
        if (selectedRecordings.size > 0 && currentFeatureSetId) {
            extractFeatures();
        }
    });

    // Export data
    document.getElementById('exportBtn').addEventListener('click', function() {
        if (selectedRecordings.size > 0 && currentFeatureSetId) {
            exportFeatures();
        }
    });

    // Select all recordings
    document.getElementById('selectAllBtn').addEventListener('click', function() {
        selectAllRecordings();
    });

    // Clear selection
    document.getElementById('clearSelectionBtn').addEventListener('click', function() {
        clearSelection();
    });

    // Select all checkbox
    document.getElementById('selectAllCheckbox').addEventListener('change', function() {
        const checkboxes = document.querySelectorAll('.recording-checkbox');
        checkboxes.forEach(checkbox => {
            checkbox.checked = this.checked;
            const recordingId = parseInt(checkbox.value);
            if (this.checked) {
                selectedRecordings.add(recordingId);
            } else {
                selectedRecordings.delete(recordingId);
            }
        });
        updateButtonStates();
    });
}

// Show datasets view
function showDatasets() {
    document.getElementById('datasetsView').style.display = 'block';
    document.getElementById('recordingsView').style.display = 'none';
    
    // Update breadcrumb
    document.getElementById('breadcrumb-datasets').classList.add('active');
    document.getElementById('breadcrumb-recordings').style.display = 'none';
    
    // Clear current dataset
    currentDatasetId = null;
    currentDatasetName = null;
    selectedRecordings.clear();
    updateButtonStates();
}

// Show recordings view
function showRecordings(datasetId, datasetName) {
    currentDatasetId = datasetId;
    currentDatasetName = datasetName;
    
    document.getElementById('datasetsView').style.display = 'none';
    document.getElementById('recordingsView').style.display = 'block';
    
    // Update breadcrumb
    document.getElementById('breadcrumb-datasets').classList.remove('active');
    document.getElementById('breadcrumb-recordings').style.display = 'block';
    document.getElementById('breadcrumb-recordings-link').textContent = datasetName;
    
    // Update title
    document.getElementById('recordingsTitle').textContent = `Recordings - ${datasetName}`;
    
    // Load recordings for this dataset
    loadRecordings(datasetId);
}

// Load datasets
async function loadDatasets() {
    try {
        const response = await fetch('/api/datasets');
        const datasets = await response.json();
        
        displayDatasets(datasets);
    } catch (error) {
        console.error('Error loading datasets:', error);
        showStatus('Error loading datasets', 'error');
    }
}

// Display datasets as cards
function displayDatasets(datasets) {
    const grid = document.getElementById('datasetsGrid');
    
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

// Load feature sets
async function loadFeatureSets() {
    try {
        const response = await fetch('/api/feature_sets');
        const featureSets = await response.json();
        
        const select = document.getElementById('featureSetSelect');
        select.innerHTML = '<option value="">Select Feature Set...</option>';
        
        featureSets.forEach(featureSet => {
            const option = document.createElement('option');
            option.value = featureSet.id;
            option.textContent = featureSet.name;
            select.appendChild(option);
        });
    } catch (error) {
        console.error('Error loading feature sets:', error);
        showStatus('Error loading feature sets', 'error');
    }
}

// Load recordings
async function loadRecordings(datasetId) {
    try {
        showStatus('Loading recordings...', 'info');
        
        const url = `/api/recordings?dataset_id=${datasetId}`;
        const response = await fetch(url);
        const recordings = await response.json();
        
        displayRecordings(recordings);
        showStatus(`Loaded ${recordings.length} recordings`, 'success');
    } catch (error) {
        console.error('Error loading recordings:', error);
        showStatus('Error loading recordings', 'error');
    }
}

// Display recordings in table
function displayRecordings(recordings) {
    const tbody = document.getElementById('recordingsTableBody');
    
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
                selectedRecordings.add(recordingId);
            } else {
                selectedRecordings.delete(recordingId);
            }
            updateButtonStates();
        });
        
        tbody.appendChild(row);
    });
}

// Select all recordings
function selectAllRecordings() {
    const checkboxes = document.querySelectorAll('.recording-checkbox');
    checkboxes.forEach(checkbox => {
        checkbox.checked = true;
        selectedRecordings.add(parseInt(checkbox.value));
    });
    document.getElementById('selectAllCheckbox').checked = true;
    updateButtonStates();
}

// Clear selection
function clearSelection() {
    const checkboxes = document.querySelectorAll('.recording-checkbox');
    checkboxes.forEach(checkbox => {
        checkbox.checked = false;
    });
    document.getElementById('selectAllCheckbox').checked = false;
    selectedRecordings.clear();
    updateButtonStates();
}

// Update button states
function updateButtonStates() {
    const extractBtn = document.getElementById('extractBtn');
    const exportBtn = document.getElementById('exportBtn');
    
    const canExtract = selectedRecordings.size > 0 && currentFeatureSetId;
    const canExport = selectedRecordings.size > 0 && currentFeatureSetId;
    
    extractBtn.disabled = !canExtract;
    exportBtn.disabled = !canExport;
}

// Show feature set details
async function showFeatureSetDetails(featureSetId) {
    try {
        const response = await fetch(`/api/feature_set_details/${featureSetId}`);
        const data = await response.json();
        
        const modal = new bootstrap.Modal(document.getElementById('featureSetModal'));
        
        // Display feature set info
        document.getElementById('featureSetInfo').innerHTML = `
            <div class="mb-3">
                <h6>Feature Set Information</h6>
                <p><strong>Name:</strong> ${data.feature_set.name}</p>
                <p><strong>Description:</strong> ${data.feature_set.description || 'No description'}</p>
                <p><strong>Features Count:</strong> ${data.features.length}</p>
            </div>
        `;
        
        // Display features list
        const featuresList = document.getElementById('featuresList');
        if (data.features.length > 0) {
            let featuresHtml = '<h6>Features:</h6><div class="table-responsive"><table class="table table-sm">';
            featuresHtml += '<thead><tr><th>Name</th><th>Function</th><th>Channels</th><th>Pipeline</th></tr></thead><tbody>';
            
            data.features.forEach(feature => {
                featuresHtml += `
                    <tr>
                        <td>${feature.shortname}</td>
                        <td>${feature.func}</td>
                        <td>${feature.chans || '-'}</td>
                        <td>${feature.pipeline_name || '-'}</td>
                    </tr>
                `;
            });
            
            featuresHtml += '</tbody></table></div>';
            featuresList.innerHTML = featuresHtml;
        } else {
            featuresList.innerHTML = '<p class="text-muted">No features found in this feature set.</p>';
        }
        
        modal.show();
    } catch (error) {
        console.error('Error loading feature set details:', error);
        showStatus('Error loading feature set details', 'error');
    }
}

// Extract features
async function extractFeatures() {
    if (selectedRecordings.size === 0 || !currentFeatureSetId) {
        showStatus('Please select recordings and a feature set', 'warning');
        return;
    }
    
    try {
        showStatus('Extracting features...', 'info');
        showProgress(true);
        
        const response = await fetch('/api/extract_features', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                recording_ids: Array.from(selectedRecordings),
                feature_set_id: currentFeatureSetId
            })
        });
        
        const result = await response.json();
        
        if (response.ok) {
            const message = `Feature extraction completed. ${result.success_count} successful, ${result.error_count} failed.`;
            showStatus(message, result.error_count > 0 ? 'warning' : 'success');
            
            if (result.errors.length > 0) {
                console.log('Errors:', result.errors);
            }
        } else {
            showStatus(result.error || 'Feature extraction failed', 'error');
        }
    } catch (error) {
        console.error('Error extracting features:', error);
        showStatus('Error extracting features', 'error');
    } finally {
        showProgress(false);
    }
}

// Export features
async function exportFeatures() {
    if (selectedRecordings.size === 0 || !currentFeatureSetId) {
        showStatus('Please select recordings and a feature set', 'warning');
        return;
    }
    
    try {
        showStatus('Preparing export...', 'info');
        
        const response = await fetch('/api/export_features', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                recording_ids: Array.from(selectedRecordings),
                feature_set_id: currentFeatureSetId
            })
        });
        
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = response.headers.get('content-disposition')?.split('filename=')[1] || 'eeg_features.csv';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            
            showStatus('Export completed successfully', 'success');
        } else {
            const result = await response.json();
            showStatus(result.error || 'Export failed', 'error');
        }
    } catch (error) {
        console.error('Error exporting features:', error);
        showStatus('Error exporting features', 'error');
    }
}

// View feature values
async function viewFeatureValues(recordingId) {
    if (!currentFeatureSetId) {
        showStatus('Please select a feature set first', 'warning');
        return;
    }
    
    try {
        const response = await fetch(`/api/feature_values?recording_id=${recordingId}&feature_set_id=${currentFeatureSetId}`);
        const featureValues = await response.json();
        
        const modal = new bootstrap.Modal(document.getElementById('featureValuesModal'));
        
        let content = '<div class="table-responsive"><table class="table table-sm feature-value-table">';
        content += '<thead><tr><th>Feature</th><th>Value</th><th>Dimension</th><th>Shape</th><th>Notes</th></tr></thead><tbody>';
        
        Object.entries(featureValues).forEach(([featureName, data]) => {
            const value = data.value !== null ? JSON.stringify(data.value) : 'N/A';
            const shape = data.shape.length > 0 ? JSON.stringify(data.shape) : 'N/A';
            
            content += `
                <tr>
                    <td><strong>${featureName}</strong></td>
                    <td><code>${value}</code></td>
                    <td>${data.dim}</td>
                    <td><code>${shape}</code></td>
                    <td>${data.notes || '-'}</td>
                </tr>
            `;
        });
        
        content += '</tbody></table></div>';
        
        document.getElementById('featureValuesContent').innerHTML = content;
        modal.show();
    } catch (error) {
        console.error('Error loading feature values:', error);
        showStatus('Error loading feature values', 'error');
    }
}

// Show status message
function showStatus(message, type = 'info') {
    const statusInfo = document.getElementById('statusInfo');
    const statusText = document.getElementById('statusText');
    
    statusText.textContent = message;
    
    const alert = statusInfo.querySelector('.alert');
    alert.className = `alert alert-${type}`;
    
    statusInfo.style.display = 'block';
    
    // Auto-hide after 5 seconds for success messages
    if (type === 'success') {
        setTimeout(() => {
            statusInfo.style.display = 'none';
        }, 5000);
    }
}

// Show/hide progress bar
function showProgress(show) {
    const progressContainer = document.getElementById('progressContainer');
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    
    if (show) {
        progressContainer.style.display = 'block';
        progressBar.style.width = '0%';
        progressText.textContent = '0%';
        
        // Simulate progress
        let progress = 0;
        const interval = setInterval(() => {
            progress += Math.random() * 10;
            if (progress > 90) progress = 90;
            
            progressBar.style.width = progress + '%';
            progressText.textContent = Math.round(progress) + '%';
        }, 500);
        
        // Store interval ID for clearing
        progressContainer.dataset.interval = interval;
    } else {
        progressContainer.style.display = 'none';
        
        // Clear interval if exists
        if (progressContainer.dataset.interval) {
            clearInterval(parseInt(progressContainer.dataset.interval));
            delete progressContainer.dataset.interval;
        }
    }
} 