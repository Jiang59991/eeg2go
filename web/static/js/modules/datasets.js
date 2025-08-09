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

// 查看某条 recording 的特征值并显示到模态框
export function viewFeatureValues(recordingId) {
    if (!recordingId) {
        console.error('viewFeatureValues called without recordingId');
        return;
    }

    const reqFeatureValues = fetch(`/api/feature_values?recording_id=${recordingId}`).then(r => r.json());
    const reqRecording = fetch(`/api/recording_details?recording_id=${recordingId}`).then(r => r.json());
    const reqMetadata = fetch(`/api/recording_metadata?recording_id=${recordingId}`).then(r => r.json());
    const reqEvents = fetch(`/api/recording_events?recording_id=${recordingId}`).then(r => r.json());

    Promise.all([reqRecording, reqMetadata, reqEvents, reqFeatureValues])
        .then(([rec, meta, events, data]) => {
            if (data && data.error) {
                showStatus(`Error: ${data.error}`, 'error');
                return;
            }

            const container = document.getElementById('featureValuesContent');
            if (!container) {
                console.error('featureValuesContent element not found');
                return;
            }

            const formatNum = (v, digits) => (typeof v === 'number' ? v.toFixed(digits) : (v ?? '-'));

            // 概览
            const overviewHtml = (rec && !rec.error) ? `
                <div class="card mb-3">
                    <div class="card-header"><strong>Recording Overview</strong></div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-6">
                                <table class="table table-sm">
                                    <tr><td><strong>ID</strong></td><td>${rec.id ?? '-'}</td></tr>
                                    <tr><td><strong>Dataset</strong></td><td>${rec.dataset_name ?? '-'}</td></tr>
                                    <tr><td><strong>Subject</strong></td><td>${rec.subject_id ?? '-'}</td></tr>
                                    <tr><td><strong>Filename</strong></td><td>${rec.filename ?? '-'}</td></tr>
                                </table>
                            </div>
                            <div class="col-md-6">
                                <table class="table table-sm">
                                    <tr><td><strong>Duration</strong></td><td>${formatNum(rec.duration, 2)} ${typeof rec.duration === 'number' ? 's' : ''}</td></tr>
                                    <tr><td><strong>Channels</strong></td><td>${rec.channels ?? '-'}</td></tr>
                                    <tr><td><strong>Sampling Rate</strong></td><td>${formatNum(rec.sampling_rate, 0)} ${typeof rec.sampling_rate === 'number' ? 'Hz' : ''}</td></tr>
                                    <tr><td><strong>Age/Sex</strong></td><td>${rec.age ?? '-'} / ${rec.sex ?? '-'}</td></tr>
                                </table>
                            </div>
                        </div>
                        <div class="row">
                            <div class="col-12">
                                <small class="text-muted">Ref: ${rec.original_reference || '-'}; Type: ${rec.recording_type || '-'}; Manufacturer: ${rec.manufacturer || '-'}</small>
                            </div>
                        </div>
                    </div>
                </div>
            ` : '';

            // 元数据
            const metaKeys = meta && typeof meta === 'object' ? Object.keys(meta) : [];
            const metadataHtml = metaKeys.length > 0 ? `
                <div class="card mb-3">
                    <div class="card-header"><strong>Recording Metadata</strong></div>
                    <div class="card-body">
                        <div class="table-responsive">
                            <table class="table table-sm">
                                <tbody>
                                    ${metaKeys.map(k => `<tr><td><strong>${k}</strong></td><td>${meta[k] ?? '-'}</td></tr>`).join('')}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            ` : '';

            // 事件（摘要 + 可展开详情）
            const eventsArr = Array.isArray(events) ? events : [];
            const eventCounts = eventsArr.reduce((acc, ev) => {
                const key = ev && ev.event_type ? ev.event_type : 'unknown';
                acc[key] = (acc[key] || 0) + 1;
                return acc;
            }, {});
            const onsetValues = eventsArr.map(e => Number(e.onset)).filter(v => Number.isFinite(v));
            const durationValues = eventsArr.map(e => Number(e.duration)).filter(v => Number.isFinite(v));
            const earliestOnset = onsetValues.length ? Math.min(...onsetValues) : null;
            const latestOnset = onsetValues.length ? Math.max(...onsetValues) : null;
            const totalDuration = durationValues.length ? durationValues.reduce((a, b) => a + b, 0) : null;
            const typesSummary = Object.keys(eventCounts).length
                ? Object.entries(eventCounts)
                    .map(([t, c]) => `<span class=\"badge bg-secondary me-1\">${t}: ${c}</span>`)
                    .join(' ')
                : '<span class=\"text-muted\">No types</span>';

            const eventsDetailsId = `events-details-${recordingId}`;
            const eventsToggleId = `events-toggle-${recordingId}`;
            const eventsHtml = `
                <div class=\"card mb-3\">
                    <div class=\"card-header d-flex justify-content-between align-items-center\">
                        <strong>Events</strong>
                        <div>
                            <span class=\"text-muted me-2\">${eventsArr.length} event(s)</span>
                            ${eventsArr.length ? `<button id=\"${eventsToggleId}\" class=\"btn btn-sm btn-outline-secondary\" onclick=\"(function(btn){var el=document.getElementById('${eventsDetailsId}');if(el){var hidden=el.classList.toggle('d-none');btn.textContent=hidden?'Show details':'Hide details';}})(this)\">Show details</button>` : ''}
                        </div>
                    </div>
                    <div class=\"card-body\">
                        ${eventsArr.length === 0 ? '<div class=\"text-muted\">No events</div>' : `
                        <div class=\"mb-2\"><strong>By Type:</strong> ${typesSummary}</div>
                        <div class=\"mb-2\"><strong>Onset Range:</strong> ${earliestOnset !== null ? earliestOnset.toFixed(2) : '-'} ~ ${latestOnset !== null ? latestOnset.toFixed(2) : '-'}</div>
                        <div class=\"mb-3\"><strong>Total Duration:</strong> ${totalDuration !== null ? totalDuration.toFixed(2) + ' s' : '-'}</div>
                        <div id=\"${eventsDetailsId}\" class=\"d-none\">
                            <div class=\"table-responsive\">
                                <table class=\"table table-sm\">
                                    <thead>
                                        <tr>
                                            <th>Type</th>
                                            <th>Onset (s)</th>
                                            <th>Duration (s)</th>
                                            <th>Value</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        ${eventsArr.map(ev => `
                                            <tr>
                                                <td>${ev.event_type ?? '-'}</td>
                                                <td>${Number.isFinite(Number(ev.onset)) ? Number(ev.onset).toFixed(2) : '-'}</td>
                                                <td>${Number.isFinite(Number(ev.duration)) ? Number(ev.duration).toFixed(2) : '-'}</td>
                                                <td>${ev.value ?? '-'}</td>
                                            </tr>
                                        `).join('')}
                                    </tbody>
                                </table>
                            </div>
                        </div>`}
                    </div>
                </div>
            `;

            // 特征值
            const entries = data ? Object.entries(data) : [];
            if (entries.length === 0) {
                container.innerHTML = overviewHtml + metadataHtml + eventsHtml + '<div class="text-muted">No feature values available for this recording.</div>';
            } else {
                const rowsHtml = entries.map(([shortname, info]) => {
                    const dim = info && info.dim ? info.dim : '-';
                    const shape = info && Array.isArray(info.shape) && info.shape.length > 0 ? info.shape.join('×') : '-';
                    const notes = info && info.notes ? info.notes : '';
                    let valueRendered = '<span class="text-muted">None</span>';
                    if (info && info.value !== undefined && info.value !== null) {
                        if (typeof info.value === 'object') {
                            try {
                                valueRendered = `<pre class=\"mb-0\">${JSON.stringify(info.value, null, 2)}</pre>`;
                            } catch (e) {
                                valueRendered = String(info.value);
                            }
                        } else {
                            valueRendered = String(info.value);
                        }
                    }
                    return `
                        <tr>
                            <td><strong>${shortname}</strong></td>
                            <td>${dim}</td>
                            <td>${shape}</td>
                            <td style=\"max-width: 600px;\">${valueRendered}</td>
                            <td>${notes}</td>
                        </tr>
                    `;
                }).join('');

                container.innerHTML = overviewHtml + metadataHtml + eventsHtml + `
                    <div class=\"card\">
                        <div class=\"card-header\"><strong>Feature Values Details</strong></div>
                        <div class=\"card-body\">
                        <div class=\"table-responsive\">
                            <table class=\"table table-sm align-middle\">
                                <thead>
                                    <tr>
                                        <th>Feature</th>
                                        <th>Dim</th>
                                        <th>Shape</th>
                                        <th>Value</th>
                                        <th>Notes</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    ${rowsHtml}
                                </tbody>
                            </table>
                        </div>
                        </div>
                    </div>
                `;
            }

            const modalEl = document.getElementById('featureValuesModal');
            if (modalEl && window.bootstrap && typeof window.bootstrap.Modal === 'function') {
                const modal = new window.bootstrap.Modal(modalEl);
                modal.show();
            } else if (modalEl) {
                modalEl.style.display = 'block';
            }
        })
        .catch(err => {
            console.error('Failed to load feature values:', err);
            showStatus('Failed to load feature values', 'error');
        });
}

// 导出到全局作用域
window.showDatasets = showDatasets;
window.showRecordings = showRecordings;
window.selectAllRecordings = selectAllRecordings;
window.clearSelection = clearSelection;
window.viewFeatureValues = viewFeatureValues;