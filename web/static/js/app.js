console.log('app.js loaded');

// Global variables
let selectedRecordings = new Set();
let currentDatasetId = null;
let currentFeatureSetId = null;
let currentDatasetName = null;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM fully loaded');
    loadDatasets();
    setupEventListeners();
    showDatasets(); // Start with datasets view
    const btn = document.getElementById('addPipelineBtn');
    console.log('btn:', btn);
    if (btn) {
        btn.addEventListener('click', function() {
            console.log('Add Pipeline clicked');
            const modalEl = document.getElementById('addPipelineModal');
            if (modalEl) {
                const modal = new bootstrap.Modal(modalEl);
                modal.show();
            } else {
                console.log('addPipelineModal not found');
            }
        });
    } else {
        console.log('addPipelineBtn not found');
    }
});

// Setup event listeners
function setupEventListeners() {
    // Navigation buttons
    const navDatasetsBtn = document.getElementById('navDatasetsBtn');
    if (navDatasetsBtn) {
        navDatasetsBtn.addEventListener('click', function() {
            setActiveNavButton(this);
            showDatasets();
        });
    }
    const navPipelinesBtn = document.getElementById('navPipelinesBtn');
    if (navPipelinesBtn) {
        navPipelinesBtn.addEventListener('click', function() {
            setActiveNavButton(this);
            showPipelines();
        });
    }
    const navFxdefsBtn = document.getElementById('navFxdefsBtn');
    if (navFxdefsBtn) {
        navFxdefsBtn.addEventListener('click', function() {
            setActiveNavButton(this);
            showFxdefs();
        });
    }
    const navFeaturesetsBtn = document.getElementById('navFeaturesetsBtn');
    if (navFeaturesetsBtn) {
        navFeaturesetsBtn.addEventListener('click', function() {
            setActiveNavButton(this);
            showFeaturesets();
        });
    }

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

    // 打开弹窗
    const btn = document.getElementById('addPipelineBtn');
    if (btn) {
        btn.addEventListener('click', function() {
            const modal = new bootstrap.Modal(document.getElementById('addPipelineModal'));
            modal.show();
        });
    }

    // 提交表单
    document.getElementById('submitAddPipelineBtn').addEventListener('click', async function() {
        const form = document.getElementById('addPipelineForm');
        const formData = new FormData(form);
        const data = {};
        for (let [key, value] of formData.entries()) {
            data[key] = value;
        }
        // steps需要转为JSON
        try {
            data.steps = JSON.parse(data.steps);
        } catch (e) {
            showStatus('Steps must be valid JSON array', 'error');
            return;
        }
        // 提交到后端
        const resp = await fetch('/api/add_pipeline', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(data)
        });
        const result = await resp.json();
        if (result.success) {
            showStatus('Pipeline added successfully', 'success');
            bootstrap.Modal.getInstance(document.getElementById('addPipelineModal')).hide();
            // 可选：刷新pipeline列表
        } else {
            showStatus(result.error || 'Failed to add pipeline', 'error');
        }
    });

    document.getElementById('addPipelineForm').addEventListener('submit', function(e) {
        // 1. HTML5 校验自动生效
        // 2. 额外自定义校验
        const rating = this.sample_rating.value;
        if (rating < 1 || rating > 10) {
            alert('Sample rating must be between 1 and 10!');
            e.preventDefault();
            return false;
        }
        // 其他自定义校验...
    });
}

// Set active navigation button
function setActiveNavButton(activeButton) {
    // Remove active class from all nav buttons
    document.querySelectorAll('.nav-buttons .btn').forEach(btn => {
        btn.classList.remove('btn-outline-primary', 'active');
        btn.classList.add('btn-outline-secondary');
    });
    
    // Add active class to clicked button
    activeButton.classList.remove('btn-outline-secondary');
    activeButton.classList.add('btn-outline-primary', 'active');
}

// Show datasets view
function showDatasets() {
    hideAllViews();
    document.getElementById('datasetsView').style.display = 'block';
    
    // Update breadcrumb
    updateBreadcrumb('datasets');
    
    // Clear current dataset
    currentDatasetId = null;
    currentDatasetName = null;
    selectedRecordings.clear();
    updateButtonStates();
}

// Show pipelines view
function showPipelines() {
    hideAllViews();
    document.getElementById('pipelinesView').style.display = 'block';
    
    // Update breadcrumb
    updateBreadcrumb('pipelines');
    
    // Load pipelines
    loadPipelines();
}

// Show feature definitions view
function showFxdefs() {
    hideAllViews();
    document.getElementById('fxdefsView').style.display = 'block';
    
    // Update breadcrumb
    updateBreadcrumb('fxdefs');
    
    // Load feature definitions
    loadFxdefs();
}

// Show feature sets view
function showFeaturesets() {
    hideAllViews();
    document.getElementById('featuresetsView').style.display = 'block';
    
    // Update breadcrumb
    updateBreadcrumb('featuresets');
    
    // Load feature sets
    loadFeaturesetsDetailed();
}

// Hide all views
function hideAllViews() {
    document.getElementById('datasetsView').style.display = 'none';
    document.getElementById('recordingsView').style.display = 'none';
    document.getElementById('pipelinesView').style.display = 'none';
    document.getElementById('fxdefsView').style.display = 'none';
    document.getElementById('featuresetsView').style.display = 'none';
}

// Update breadcrumb
function updateBreadcrumb(activeItem) {
    // Hide all breadcrumb items
    document.getElementById('breadcrumb-datasets').style.display = 'none';
    document.getElementById('breadcrumb-recordings').style.display = 'none';
    document.getElementById('breadcrumb-pipelines').style.display = 'none';
    document.getElementById('breadcrumb-fxdefs').style.display = 'none';
    document.getElementById('breadcrumb-featuresets').style.display = 'none';
    
    // Show active item
    document.getElementById(`breadcrumb-${activeItem}`).style.display = 'block';
}

// Show recordings view
function showRecordings(datasetId, datasetName) {
    currentDatasetId = datasetId;
    currentDatasetName = datasetName;
    
    hideAllViews();
    document.getElementById('recordingsView').style.display = 'block';
    
    // Update breadcrumb
    document.getElementById('breadcrumb-recordings').style.display = 'block';
    document.getElementById('breadcrumb-recordings-link').textContent = datasetName;
    
    // Update title
    document.getElementById('recordingsTitle').textContent = `Recordings - ${datasetName}`;
    
    // Load recordings for this dataset
    loadRecordings(datasetId);
}

// Load pipelines
async function loadPipelines() {
    try {
        showStatus('Loading pipelines...', 'info');
        
        const response = await fetch('/api/pipelines');
        const pipelines = await response.json();
        
        displayPipelines(pipelines);
        showStatus(`Loaded ${pipelines.length} pipelines`, 'success');
    } catch (error) {
        console.error('Error loading pipelines:', error);
        showStatus('Error loading pipelines', 'error');
    }
}

// Display pipelines as cards
function displayPipelines(pipelines) {
    const grid = document.getElementById('pipelinesGrid');
    
    if (pipelines.length === 0) {
        grid.innerHTML = '<div class="col-12"><p class="text-center text-muted">No pipelines found</p></div>';
        return;
    }
    
    grid.innerHTML = '';
    
    pipelines.forEach(pipeline => {
        const col = document.createElement('div');
        col.className = 'col-lg-4 col-md-6 col-sm-12 mb-3';
        
        col.innerHTML = `
            <div class="dataset-card" onclick="showPipelineDetails(${pipeline.id})">
                <div class="d-flex align-items-center mb-2">
                    <i class="bi bi-diagram-3 text-primary me-2"></i>
                    <h5 class="mb-0">${pipeline.shortname}</h5>
                </div>
                <div class="dataset-description">
                    ${pipeline.description || 'No description available'}
                </div>
                <div class="dataset-stats">
                    <span><i class="bi bi-gear"></i> Source: ${pipeline.source || 'Unknown'}</span>
                    <br>
                    <span><i class="bi bi-speedometer2"></i> Sampling Rate: ${pipeline.fs || 'N/A'} Hz</span>
                </div>
            </div>
        `;
        
        grid.appendChild(col);
    });
}

// Load feature definitions
async function loadFxdefs() {
    try {
        showStatus('Loading feature definitions...', 'info');
        
        const response = await fetch('/api/fxdefs');
        const fxdefs = await response.json();
        
        displayFxdefs(fxdefs);
        showStatus(`Loaded ${fxdefs.length} feature definitions`, 'success');
    } catch (error) {
        console.error('Error loading feature definitions:', error);
        showStatus('Error loading feature definitions', 'error');
    }
}

// Display feature definitions in table
function displayFxdefs(fxdefs) {
    const tbody = document.getElementById('fxdefsTableBody');
    
    if (fxdefs.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" class="text-center text-muted">No feature definitions found</td></tr>';
        return;
    }
    
    tbody.innerHTML = '';
    
    fxdefs.forEach(fxdef => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${fxdef.id}</td>
            <td><strong>${fxdef.shortname}</strong></td>
            <td>${fxdef.ver || 'N/A'}</td>
            <td>${fxdef.dim || 'N/A'}</td>
            <td>${fxdef.pipeline_name || 'N/A'}</td>
            <td><span class="badge bg-secondary">${fxdef.feature_type || 'single_channel'}</span></td>
            <td>
                <button class="btn btn-outline-primary btn-sm" onclick="showFxdefDetails(${fxdef.id})">
                    <i class="bi bi-eye"></i> View
                </button>
            </td>
        `;
        tbody.appendChild(row);
    });
}

// Load feature sets detailed
async function loadFeaturesetsDetailed() {
    try {
        showStatus('Loading feature sets...', 'info');
        
        const response = await fetch('/api/featuresets_detailed');
        const featuresets = await response.json();
        
        displayFeaturesetsDetailed(featuresets);
        showStatus(`Loaded ${featuresets.length} feature sets`, 'success');
    } catch (error) {
        console.error('Error loading feature sets:', error);
        showStatus('Error loading feature sets', 'error');
    }
}

// Display feature sets as cards
function displayFeaturesetsDetailed(featuresets) {
    const grid = document.getElementById('featuresetsGrid');
    
    if (featuresets.length === 0) {
        grid.innerHTML = '<div class="col-12"><p class="text-center text-muted">No feature sets found</p></div>';
        return;
    }
    
    grid.innerHTML = '';
    
    featuresets.forEach(item => {
        const fs = item.feature_set;
        const col = document.createElement('div');
        col.className = 'col-lg-4 col-md-6 col-sm-12 mb-3';
        
        col.innerHTML = `
            <div class="dataset-card" onclick="showFeatureSetDetails(${fs.id})">
                <div class="d-flex align-items-center mb-2">
                    <i class="bi bi-collection text-primary me-2"></i>
                    <h5 class="mb-0">${fs.name}</h5>
                </div>
                <div class="dataset-description">
                    ${fs.description || 'No description available'}
                </div>
                <div class="dataset-stats">
                    <span><i class="bi bi-gear"></i> Features: ${item.fxdef_count}</span>
                </div>
            </div>
        `;
        
        grid.appendChild(col);
    });
}

// Show pipeline details
async function showPipelineDetails(pipelineId) {
    try {
        const modal = new bootstrap.Modal(document.getElementById('pipelineModal'));
        
        // 显示加载状态
        document.getElementById('pipelineInfoPanel').innerHTML = `
            <div class="pipeline-visualization-loading">
                <div class="spinner-border" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <div class="ms-3">Loading pipeline details...</div>
            </div>
        `;
        
        modal.show();
        
        // 并行获取pipeline详情和可视化数据
        const [response, vizResponse] = await Promise.all([
            fetch(`/api/pipeline_details/${pipelineId}`),
            fetch(`/api/pipeline_visualization/${pipelineId}`)
        ]);
        
        if (!response.ok) {
            throw new Error(`Failed to load pipeline details: ${response.status}`);
        }
        
        const data = await response.json();
        const vizData = await vizResponse.json();
        
        // Pipeline info
        const pipeline = data.pipeline;
        // 渲染信息
        document.getElementById('pipelineInfoPanel').innerHTML = `
            <p><strong>Name:</strong> ${pipeline.shortname}</p>
            <p><strong>Description:</strong> ${pipeline.description || 'N/A'}</p>
            <p><strong>Source:</strong> ${pipeline.source || 'N/A'}</p>
            <p><strong>Sampling Rate:</strong> ${pipeline.fs || 'N/A'} Hz</p>
            <p><strong>High Pass:</strong> ${pipeline.hp || 'N/A'} Hz</p>
            <p><strong>Low Pass:</strong> ${pipeline.lp || 'N/A'} Hz</p>
        `;
        
        // Pipeline visualization with Cytoscape.js
        if (vizData.cytoscape_data && !vizData.error) {
            // 渲染 Cytoscape 图（保持原有逻辑即可）
            initializeCytoscape(vizData.cytoscape_data);
        } else {
            document.getElementById('pipelineInfoPanel').innerHTML = `
                <div class="pipeline-visualization-error">
                    <i class="bi bi-exclamation-triangle"></i>
                    <p class="mb-0">Unable to generate pipeline visualization.</p>
                    <small>${vizData.error || 'This might be due to missing pipeline structure.'}</small>
                </div>
            `;
        }
        
        // Pipeline nodes
        const nodes = data.nodes;
        const fxdefs = data.fxdefs || [];
        document.getElementById('pipelineFxdefsHeader').innerHTML = 
            `Feature Definitions Using This Pipeline (${fxdefs.length})`;

        document.getElementById('pipelineFxdefs').innerHTML = fxdefs.length > 0 ? `
            <div class="table-responsive">
                <table class="table table-sm">
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Short Name</th>
                            <th>Version</th>
                            <th>Dimension</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${fxdefs.map(fxdef => `
                            <tr>
                                <td>${fxdef.id}</td>
                                <td><strong>${fxdef.shortname}</strong></td>
                                <td>${fxdef.ver || 'N/A'}</td>
                                <td>${fxdef.dim || 'N/A'}</td>
                                <td>
                                    <button class="btn btn-outline-primary btn-sm" onclick="showFxdefDetails(${fxdef.id})">
                                        <i class="bi bi-eye"></i> View
                                    </button>
                                </td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        ` : '<p class="text-muted">No feature definitions use this pipeline.</p>';
        
        // 更新内容
        document.getElementById('pipelineInfoPanel').innerHTML = `
            <p><strong>Name:</strong> ${pipeline.shortname}</p>
            <p><strong>Description:</strong> ${pipeline.description || 'N/A'}</p>
            <p><strong>Source:</strong> ${pipeline.source || 'N/A'}</p>
            <p><strong>Sampling Rate:</strong> ${pipeline.fs || 'N/A'} Hz</p>
            <p><strong>High Pass:</strong> ${pipeline.hp || 'N/A'} Hz</p>
            <p><strong>Low Pass:</strong> ${pipeline.lp || 'N/A'} Hz</p>
        `;
        
        // 在showPipelineDetails最后，监听modal显示后再初始化Cytoscape
        const modalEl = document.getElementById('pipelineModal');
        modalEl.addEventListener('shown.bs.modal', function handler() {
            if (vizData.cytoscape_data && vizData.cytoscape_data.nodes.length > 0) {
                initializeCytoscape(vizData.cytoscape_data);
            }
            // 只监听一次
            modalEl.removeEventListener('shown.bs.modal', handler);
        });
        
    } catch (error) {
        console.error('Error loading pipeline details:', error);
        document.getElementById('pipelineInfoPanel').innerHTML = `
            <div class="pipeline-visualization-error">
                <i class="bi bi-exclamation-triangle"></i>
                <p class="mb-0">Error loading pipeline details</p>
                <small>${error.message}</small>
            </div>
        `;
        showStatus('Error loading pipeline details', 'error');
    }
}

// 初始化Cytoscape.js可视化
function initializeCytoscape(cytoscapeData) {
    const container = document.getElementById('pipeline-cytoscape');
    if (!container) return;
    
    // 清除之前的可视化
    container.innerHTML = '';
    
    // 创建Cytoscape实例
    const cy = cytoscape({
        container: container,
        elements: {
            nodes: cytoscapeData.nodes,
            edges: cytoscapeData.edges
        },
        style: [
            {
                selector: 'node',
                style: {
                    'background-color': '#ADD8E6',
                    'label': 'data(label)',
                    'text-wrap': 'wrap',
                    'text-max-width': '120px',
                    'font-size': '14px', // 字号适中
                    'font-family': 'Consolas, Menlo, Monaco, \"Fira Mono\", \"Roboto Mono\", \"Courier New\", monospace',
                    'text-valign': 'center',
                    'text-halign': 'center',
                    'width': '120px',
                    'height': '60px',
                    'border-width': 2,
                    'border-color': '#666',
                    'border-opacity': 0.8,
                    'shape': 'rectangle'
                }
            },
            {
                selector: 'node.input',
                style: {
                    'background-color': '#90EE90'
                }
            },
            {
                selector: 'node.output',
                style: {
                    'background-color': '#F0A0A0'
                }
            },
            {
                selector: 'node.filter',
                style: {
                    'background-color': '#FFFFE0'
                }
            },
            {
                selector: 'node.process',
                style: {
                    'background-color': '#ADD8E6'
                }
            },
            {
                selector: 'edge',
                style: {
                    'width': 3,
                    'line-color': '#666',
                    'target-arrow-color': '#666',
                    'target-arrow-shape': 'triangle',
                    'curve-style': 'bezier',
                    'arrow-scale': 1.5
                }
            }
        ],
        layout: {
            name: 'dagre',
            rankDir: 'TB',
            nodeSep: 50,
            edgeSep: 30,
            rankSep: 80,
            padding: 20
        },
        userZoomingEnabled: false,
        userPanningEnabled: true,
        boxSelectionEnabled: true,
        autoungrabify: false,
        autolock: false
    });
    
    // 添加交互功能
    cy.on('mouseover', 'node', function(e) {
        const node = e.target;
        node.style('border-width', 4);
        node.style('border-color', '#007bff');
    });
    
    cy.on('mouseout', 'node', function(e) {
        const node = e.target;
        node.style('border-width', 2);
        node.style('border-color', '#666');
    });
    
    // 先 resize 一次
    cy.resize();

    // 关键：延迟 fit/center，确保容器宽度已渲染
    setTimeout(() => {
        cy.resize();
        cy.fit();
        cy.center();
    }, 50);

    // 可选：监听窗口大小变化
    window.addEventListener('resize', () => {
        cy.resize();
        cy.fit();
    });

    // 放大缩小按钮事件
    document.getElementById('cy-zoom-in').onclick = () => cy.zoom({ level: cy.zoom() * 1.2, renderedPosition: { x: cy.width()/2, y: cy.height()/2 } });
    document.getElementById('cy-zoom-out').onclick = () => cy.zoom({ level: cy.zoom() / 1.2, renderedPosition: { x: cy.width()/2, y: cy.height()/2 } });
    document.getElementById('cy-zoom-fit').onclick = () => { cy.fit(); cy.center(); };
}

// Show feature definition details
async function showFxdefDetails(fxdefId) {
    try {
        const response = await fetch(`/api/fxdef_details/${fxdefId}`);
        const data = await response.json();
        
        const modal = new bootstrap.Modal(document.getElementById('fxdefModal'));
        
        // Fxdef info
        const fxdef = data.fxdef;
        document.getElementById('fxdefInfo').innerHTML = `
            <div class="card mb-3">
                <div class="card-header">
                    <h6 class="mb-0">Feature Definition Information</h6>
                </div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-6">
                            <p><strong>ID:</strong> ${fxdef.id}</p>
                            <p><strong>Short Name:</strong> ${fxdef.shortname}</p>
                            <p><strong>Version:</strong> ${fxdef.ver || 'N/A'}</p>
                            <p><strong>Dimension:</strong> ${fxdef.dim || 'N/A'}</p>
                        </div>
                        <div class="col-md-6">
                            <p><strong>Pipeline:</strong> ${fxdef.pipeline_name || 'N/A'}</p>
                            <p><strong>Feature Type:</strong> ${fxdef.feature_type || 'single_channel'}</p>
                            <p><strong>Channels:</strong> ${fxdef.chans || 'N/A'}</p>
                            <p><strong>Function:</strong> <code>${fxdef.func}</code></p>
                        </div>
                    </div>
                    ${fxdef.notes ? `<p><strong>Notes:</strong> ${fxdef.notes}</p>` : ''}
                    ${fxdef.params ? `<p><strong>Parameters:</strong> <code>${fxdef.params}</code></p>` : ''}
                </div>
            </div>
        `;
        
        // Feature sets that include this fxdef
        const featureSets = data.feature_sets;
        document.getElementById('fxdefFeatureSets').innerHTML = `
            <div class="card mb-3">
                <div class="card-header">
                    <h6 class="mb-0">Feature Sets Including This Definition (${featureSets.length})</h6>
                </div>
                <div class="card-body">
                    ${featureSets.length > 0 ? `
                        <ul class="list-group list-group-flush">
                            ${featureSets.map(fs => `
                                <li class="list-group-item d-flex justify-content-between align-items-center">
                                    ${fs.name}
                                    <button class="btn btn-outline-primary btn-sm" onclick="showFeatureSetDetails(${fs.id})">
                                        <i class="bi bi-eye"></i> View
                                    </button>
                                </li>
                            `).join('')}
                        </ul>
                    ` : '<p class="text-muted">This feature definition is not included in any feature sets.</p>'}
                </div>
            </div>
        `;
        
        // Sample values
        const sampleValues = data.sample_values;
        document.getElementById('fxdefSampleValues').innerHTML = `
            <div class="card">
                <div class="card-header">
                    <h6 class="mb-0">Sample Feature Values (${sampleValues.length})</h6>
                </div>
                <div class="card-body">
                    ${sampleValues.length > 0 ? `
                        <div class="table-responsive">
                            <table class="table table-sm">
                                <thead>
                                    <tr>
                                        <th>Recording</th>
                                        <th>Value</th>
                                        <th>Dimension</th>
                                        <th>Shape</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    ${sampleValues.map(sv => `
                                        <tr>
                                            <td>${sv.filename}</td>
                                            <td><code>${sv.value || 'N/A'}</code></td>
                                            <td>${sv.dim || 'N/A'}</td>
                                            <td>${sv.shape || 'N/A'}</td>
                                        </tr>
                                    `).join('')}
                                </tbody>
                            </table>
                        </div>
                    ` : '<p class="text-muted">No feature values calculated yet.</p>'}
                </div>
            </div>
        `;
        
        modal.show();
    } catch (error) {
        console.error('Error loading feature definition details:', error);
        showStatus('Error loading feature definition details', 'error');
    }
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
        if (select) {
            select.innerHTML = '<option value="">Select Feature Set...</option>';
            
            featureSets.forEach(featureSet => {
                const option = document.createElement('option');
                option.value = featureSet.id;
                option.textContent = featureSet.name;
                select.appendChild(option);
            });
        }
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
        console.log('showFeatureSetDetails called', featureSetId);
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
    try {
        const response = await fetch(`/api/feature_values?recording_id=${recordingId}`);
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