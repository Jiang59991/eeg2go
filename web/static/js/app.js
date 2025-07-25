console.log('app.js loaded');

// Global variables
let selectedRecordings = new Set();
let currentDatasetId = null;
let currentFeatureSetId = null;
let currentDatasetName = null;
let pipelineSteps = [];
let FUNC_DIM_MAP = {};

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
    document.getElementById('addStepBtn').addEventListener('click', addStep);

    // 绑定modal事件
    const modal = document.getElementById('addPipelineModal');
    if (modal) {
        modal.addEventListener('show.bs.modal', function () {
            hidePipelineError();
        });
        modal.addEventListener('hidden.bs.modal', function () {
            document.querySelectorAll('.modal-backdrop').forEach(el => el.remove());
            document.body.classList.remove('modal-open');
            const mainBtn = document.getElementById('addPipelineBtn');
            if (mainBtn) mainBtn.focus();
        });
    }
    
    // 绑定实验弹窗事件
    const experimentModal = document.getElementById('addExperimentModal');
    if (experimentModal) {
        experimentModal.addEventListener('show.bs.modal', function () {
            hideExperimentError();
        });
        
        // 监听实验类型变化
        const experimentTypeSelect = experimentModal.querySelector('select[name="experiment_type"]');
        if (experimentTypeSelect) {
            experimentTypeSelect.addEventListener('change', function() {
                loadExperimentParams(this.value);
            });
        }
        
        // 绑定提交按钮
        const submitBtn = document.getElementById('submitAddExperimentBtn');
        if (submitBtn) {
            submitBtn.addEventListener('click', submitAddExperiment);
        }
    }
    
    // 绑定Feature Extraction导航按钮
    const navFeatureExtractionBtn = document.getElementById('navFeatureExtractionBtn');
    if (navFeatureExtractionBtn) {
        navFeatureExtractionBtn.addEventListener('click', showFeatureExtraction);
    }

    // 打开Add Feature Definition弹窗时，加载函数和pipeline选项
    document.getElementById('addFxdefBtn').addEventListener('click', function() {
        loadFeatureFunctions();
        loadPipelinesForFxdef();
        hideFxdefError();
        const modal = new bootstrap.Modal(document.getElementById('addFxdefModal'));
        modal.show();
    });

    // 新建Pipeline按钮
    document.getElementById('openAddPipelineBtn').addEventListener('click', function() {
        const modal = new bootstrap.Modal(document.getElementById('addPipelineModal'));
        modal.show();
        // 监听Pipeline添加成功后刷新下拉框
        document.getElementById('addPipelineModal').addEventListener('hidden.bs.modal', function handler() {
            loadPipelinesForFxdef();
            // 可选：自动选中新建的pipeline
            // document.getElementById('fxdefPipelineSelect').value = 新pipeline的id;
            document.getElementById('addPipelineModal').removeEventListener('hidden.bs.modal', handler);
        });
    });

    // 提交Feature Definition
    document.getElementById('submitAddFxdefBtn').addEventListener('click', async function() {
        hideFxdefError();
        const form = document.getElementById('addFxdefForm');
        const formData = new FormData(form);
        const data = {};
        for (let [key, value] of formData.entries()) {
            data[key] = value;
        }
        // 处理channels为数组
        data.channels = data.channels.split(',').map(s => s.trim()).filter(Boolean);
        // 处理params为对象
        if (data.params) {
            try {
                data.params = JSON.parse(data.params);
            } catch {
                showFxdefError('Params必须是合法的JSON字符串');
                return;
            }
        } else {
            data.params = {};
        }
        // 其它类型转换
        data.pipeid = parseInt(data.pipeid);
        // 自动推断dimension
        if (FUNC_DIM_MAP[data.func]) {
            data.dim = FUNC_DIM_MAP[data.func];
        } else {
            showFxdefError('无法推断dimension，请检查feature function');
            return;
        }
        // 校验
        if (!data.func || !data.pipeid || !data.shortname || !data.channels.length) {
            showFxdefError('请填写所有必填项');
            return;
        }
        // 提交
        try {
            const resp = await fetch('/api/add_fxdef', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(data)
            });
            const result = await resp.json();
            if (resp.ok && result.success) {
                // 成功，关闭弹窗并刷新Feature Definitions列表
                bootstrap.Modal.getInstance(document.getElementById('addFxdefModal')).hide();
                showFxdefs(); // 刷新列表
            } else {
                showFxdefError(result.error || '添加失败');
            }
        } catch (e) {
            showFxdefError('网络错误或服务器无响应');
        }
    });

    // 打开Add Feature Set弹窗
    document.getElementById('addFeaturesetBtn').addEventListener('click', function() {
        loadFxdefsForFeatureset();
        hideFeaturesetError();
        const modal = new bootstrap.Modal(document.getElementById('addFeaturesetModal'));
        modal.show();
    });

    // 从Feature Set弹窗中打开Add Feature Definition弹窗
    document.getElementById('openAddFxdefFromFeaturesetBtn').addEventListener('click', function() {
        // 先关闭Feature Set弹窗
        const featuresetModal = bootstrap.Modal.getInstance(document.getElementById('addFeaturesetModal'));
        if (featuresetModal) featuresetModal.hide();
        
        // 等待动画结束后打开Add Feature Definition弹窗
        setTimeout(() => {
            loadFeatureFunctions();
            loadPipelinesForFxdef();
            hideFxdefError();
            const fxdefModal = new bootstrap.Modal(document.getElementById('addFxdefModal'));
            fxdefModal.show();
            
            // 监听Feature Definition添加成功后，重新打开Feature Set弹窗并刷新列表
            document.getElementById('addFxdefModal').addEventListener('hidden.bs.modal', function handler() {
                // 重新打开Feature Set弹窗
                const newFeaturesetModal = new bootstrap.Modal(document.getElementById('addFeaturesetModal'));
                newFeaturesetModal.show();
                // 刷新Feature Definitions列表
                loadFxdefsForFeatureset();
                // 移除事件监听器，避免重复绑定
                document.getElementById('addFxdefModal').removeEventListener('hidden.bs.modal', handler);
            });
        }, 300);
    });

    // 提交Feature Set
    document.getElementById('submitAddFeaturesetBtn').addEventListener('click', async function() {
        hideFeaturesetError();
        const form = document.getElementById('addFeaturesetForm');
        const formData = new FormData(form);
        const data = {};
        for (let [key, value] of formData.entries()) {
            data[key] = value;
        }
        
        // 获取选中的fxdef_ids
        const selectedCheckboxes = document.querySelectorAll('#fxdefCheckboxes input[type="checkbox"]:checked');
        const fxdef_ids = Array.from(selectedCheckboxes).map(cb => parseInt(cb.value));
        
        if (!data.name || fxdef_ids.length === 0) {
            showFeaturesetError('请填写Feature Set名称并至少选择一个Feature Definition');
            return;
        }
        
        data.fxdef_ids = fxdef_ids;
        
        try {
            const resp = await fetch('/api/add_featureset', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(data)
            });
            const result = await resp.json();
            if (resp.ok && result.success) {
                // 成功，关闭弹窗并刷新Feature Sets列表
                bootstrap.Modal.getInstance(document.getElementById('addFeaturesetModal')).hide();
                showFeaturesets(); // 刷新列表
            } else {
                showFeaturesetError(result.error || '添加失败');
            }
        } catch (e) {
            showFeaturesetError('网络错误或服务器无响应');
        }
    });

    // 提交Add Pipeline
    document.getElementById('submitAddPipelineBtn').addEventListener('click', async function() {
        hidePipelineError();
        updateStepInputnames(); // 保证 inputnames 正确

        const form = document.getElementById('addPipelineForm');
        const formData = new FormData(form);
        const data = {};
        for (let [key, value] of formData.entries()) {
            data[key] = value;
        }
        data.steps = pipelineSteps;
        try {
            const resp = await fetch('/api/add_pipeline', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(data)
            });
            let result;
            try {
                result = await resp.json();
            } catch (e) {
                result = {};
            }
            if (resp.ok && result.success) {
                showStatus('Pipeline added successfully', 'success');
                const modal = bootstrap.Modal.getInstance(document.getElementById('addPipelineModal'));
                if (modal) modal.hide();

                // 关键：刷新下拉框并选中新建的 pipeline
                // 假设后端返回 {success: true, pipeline_id: 新id}
                await loadPipelinesForFxdef(result.pipeline_id);
            } else {
                showStatus(result.error || 'Failed to add pipeline', 'error');
                showPipelineError(result.error || 'Failed to add pipeline');
            }
        } catch (error) {
            showStatus('Internal server error', 'error');
        }
    });
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
    const navExperimentsBtn = document.getElementById('navExperimentsBtn');
    if (navExperimentsBtn) {
        navExperimentsBtn.addEventListener('click', function() {
            setActiveNavButton(this);
            showExperiments();
        });
    }



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

// Show experiments view
function showExperiments() {
    hideAllViews();
    document.getElementById('experimentsView').style.display = 'block';
    
    // Update breadcrumb
    updateBreadcrumb('experiments');
    
    // Load experiments
    loadExperiments();
}

// Hide all views
function hideAllViews() {
    document.getElementById('datasetsView').style.display = 'none';
    document.getElementById('recordingsView').style.display = 'none';
    document.getElementById('pipelinesView').style.display = 'none';
    document.getElementById('fxdefsView').style.display = 'none';
    document.getElementById('featuresetsView').style.display = 'none';
    document.getElementById('featureExtractionView').style.display = 'none';
    document.getElementById('experimentsView').style.display = 'none';
}

// Update breadcrumb
function updateBreadcrumb(activeItem) {
    // Hide all breadcrumb items
    document.getElementById('breadcrumb-datasets').style.display = 'none';
    document.getElementById('breadcrumb-recordings').style.display = 'none';
    document.getElementById('breadcrumb-pipelines').style.display = 'none';
    document.getElementById('breadcrumb-fxdefs').style.display = 'none';
    document.getElementById('breadcrumb-featuresets').style.display = 'none';
    document.getElementById('breadcrumb-experiments').style.display = 'none';
    
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
    // Button states are now handled by individual views
}

// Show feature set details
async function showFeatureSetDetails(featureSetId) {
    const fxdefModal = bootstrap.Modal.getInstance(document.getElementById('fxdefModal'));
    if (fxdefModal) fxdefModal.hide();
    
    const modal = new bootstrap.Modal(document.getElementById('featureSetModal'));
    modal.show();

    try {
        console.log('showFeatureSetDetails called', featureSetId);
        
        // 并行获取feature set详情和DAG数据
        const [response, dagResponse] = await Promise.all([
            fetch(`/api/feature_set_details/${featureSetId}`),
            fetch(`/api/featureset_dag/${featureSetId}`)
        ]);
        
        const data = await response.json();
        const dagData = await dagResponse.json();
        
        // Display feature set info
        document.getElementById('featureSetInfo').innerHTML = `
            <div class="mb-3">
                <h6>Feature Set Information</h6>
                <p><strong>Name:</strong> ${data.feature_set.name}</p>
                <p><strong>Description:</strong> ${data.feature_set.description || 'No description'}</p>
                <p><strong>Features Count:</strong> ${data.features.length}</p>
                ${dagData.success ? `<p><strong>DAG Nodes:</strong> ${dagData.node_count}</p>` : ''}
                ${dagData.success ? `<p><strong>DAG Edges:</strong> ${dagData.edge_count}</p>` : ''}
            </div>
        `;
        
        // Display features list
        const featuresList = document.getElementById('featuresList');
        document.getElementById('featureSetFeaturesHeader').innerHTML = 
            `Features in This Set (${data.features.length})`;
            
        if (data.features.length > 0) {
            let featuresHtml = '<div class="table-responsive"><table class="table table-sm">';
            featuresHtml += '<thead><tr><th>Name</th><th>Function</th><th>Channels</th><th>Pipeline</th><th>Dimension</th></tr></thead><tbody>';
            
            data.features.forEach(feature => {
                featuresHtml += `
                    <tr>
                        <td><strong>${feature.shortname}</strong></td>
                        <td><code>${feature.func}</code></td>
                        <td>${feature.chans || '-'}</td>
                        <td>${feature.pipeline_name || '-'}</td>
                        <td>${feature.dim || '-'}</td>
                    </tr>
                `;
            });
            
            featuresHtml += '</tbody></table></div>';
            featuresList.innerHTML = featuresHtml;
        } else {
            featuresList.innerHTML = '<p class="text-muted">No features found in this feature set.</p>';
        }
        
        // Initialize DAG visualization
        if (dagData.success && dagData.cytoscape_data) {
            // 监听modal显示后再初始化Cytoscape
            const modalEl = document.getElementById('featureSetModal');
            modalEl.addEventListener('shown.bs.modal', function handler() {
                if (dagData.cytoscape_data.nodes.length > 0) {
                    initializeFeaturesetCytoscape(dagData.cytoscape_data);
                }
                // 绑定DAG执行事件
                bindDagExecutionEvents(featureSetId);
                // 只监听一次
                modalEl.removeEventListener('shown.bs.modal', handler);
            });
        } else {
            // 显示DAG加载错误
            document.getElementById('featureset-cytoscape').innerHTML = `
                <div class="text-center text-muted">
                    <i class="bi bi-exclamation-triangle"></i>
                    <p class="mb-0">Unable to generate DAG visualization.</p>
                    <small>${dagData.error || 'This might be due to missing pipeline structure.'}</small>
                </div>
            `;
        }
        
    } catch (error) {
        console.error('Error loading feature set details:', error);
        showStatus('Error loading feature set details', 'error');
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
    // 关键：将 error 映射为 danger
    const typeMap = { error: 'danger', success: 'success', info: 'info', warning: 'warning' };
    alert.className = `alert alert-${typeMap[type] || 'info'}`;
    
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

function renderPipelineSteps() {
    updateStepInputnames();
    const list = document.getElementById('pipelineStepsList');
    list.innerHTML = '';
    pipelineSteps.forEach((step, idx) => {
        const paramDefs = STEP_REGISTRY[step.func]?.params || {};
        // 自动补全 params
        for (const pname in paramDefs) {
            if (step.params[pname] === undefined) {
                step.params[pname] = paramDefs[pname].default || '';
            }
        }
        let paramInputs = '';
        for (const pname in paramDefs) {
            const pinfo = paramDefs[pname];
            const val = step.params[pname];
            paramInputs += `
                <label class="me-1">${pname}:</label>
                <input class="form-control form-control-sm me-2" style="width:80px;display:inline-block"
                    type="${pinfo.type === 'float' ? 'number' : 'text'}"
                    value="${val}"
                    min="${pinfo.min !== undefined ? pinfo.min : ''}"
                    max="${pinfo.max !== undefined ? pinfo.max : ''}"
                    step="any"
                    onchange="changeStepParamValue(${idx}, '${pname}', this.value)">
            `;
        }
        const div = document.createElement('div');
        div.className = 'd-flex align-items-center mb-2';
        div.innerHTML = `
            <span class="me-2">${idx + 1}.</span>
            <select class="form-select form-select-sm me-2" style="width:120px" onchange="changeStepFunc(${idx}, this.value)">
                ${Object.keys(STEP_REGISTRY).map(func =>
                    `<option value="${func}" ${step.func === func ? 'selected' : ''}>${func}</option>`
                ).join('')}
            </select>
            ${paramInputs}
            <button class="btn btn-outline-secondary btn-sm me-1" onclick="moveStepUp(${idx})" ${idx === 0 ? 'disabled' : ''}>↑</button>
            <button class="btn btn-outline-secondary btn-sm me-1" onclick="moveStepDown(${idx})" ${idx === pipelineSteps.length - 1 ? 'disabled' : ''}>↓</button>
            <button class="btn btn-outline-danger btn-sm" onclick="deleteStep(${idx})">删除</button>
        `;
        list.appendChild(div);
    });
}

function addStep() {
    const firstFunc = Object.keys(STEP_REGISTRY)[0];
    const paramDefs = STEP_REGISTRY[firstFunc]?.params || {};
    const params = {};
    for (const pname in paramDefs) {
        params[pname] = paramDefs[pname].default || '';
    }
    pipelineSteps.push({ step_name: `step${pipelineSteps.length+1}`, func: firstFunc, inputnames: [], params });
    renderPipelineSteps();
}

function changeStepFunc(idx, func) {
    pipelineSteps[idx].func = func;
    // 自动重置参数
    const paramDefs = STEP_REGISTRY[func]?.params || {};
    const params = {};
    for (const pname in paramDefs) {
        params[pname] = paramDefs[pname].default || '';
    }
    pipelineSteps[idx].params = params;
    renderPipelineSteps();
}

function changeStepParams(idx, val) {
    try {
        pipelineSteps[idx].params = JSON.parse(val);
    } catch {
        alert('参数必须是合法JSON');
    }
}

function moveStepUp(idx) {
    if (idx > 0) {
        [pipelineSteps[idx-1], pipelineSteps[idx]] = [pipelineSteps[idx], pipelineSteps[idx-1]];
        renderPipelineSteps();
    }
}

function moveStepDown(idx) {
    if (idx < pipelineSteps.length - 1) {
        [pipelineSteps[idx+1], pipelineSteps[idx]] = [pipelineSteps[idx], pipelineSteps[idx+1]];
        renderPipelineSteps();
    }
}

function deleteStep(idx) {
    pipelineSteps.splice(idx, 1);
    renderPipelineSteps();
} 

function changeStepParamValue(idx, pname, value) {
    const paramDef = STEP_REGISTRY[pipelineSteps[idx].func].params[pname];
    if (paramDef.type === 'float') {
        value = parseFloat(value);
        if (isNaN(value)) value = undefined;
    }
    pipelineSteps[idx].params[pname] = value;
    // 重新渲染以保持输入框和数据同步
    renderPipelineSteps();
} 

function showPipelineError(message) {
    const alertDiv = document.getElementById('pipelineErrorAlert');
    const textSpan = document.getElementById('pipelineErrorText');
    textSpan.textContent = message;
    alertDiv.style.display = 'block';
}

function hidePipelineError() {
    document.getElementById('pipelineErrorAlert').style.display = 'none';
} 

// 加载pipeline下拉框
async function loadPipelinesForFxdef(selectedId) {
    const select = document.getElementById('fxdefPipelineSelect');
    select.innerHTML = '';
    try {
        const resp = await fetch('/api/pipelines');
        const pipelines = await resp.json();
        pipelines.forEach(p => {
            const opt = document.createElement('option');
            opt.value = p.id;
            opt.textContent = p.shortname;
            if (selectedId && p.id == selectedId) {
                opt.selected = true;
            }
            select.appendChild(opt);
        });
    } catch (e) {
        select.innerHTML = '<option value=\"\">Loading failed</option>';
    }
}

// 加载特征函数下拉框
async function loadFeatureFunctions() {
    const select = document.querySelector('select[name="func"]');
    select.innerHTML = '';
    FUNC_DIM_MAP = {};
    try {
        const resp = await fetch('/api/feature_functions');
        const funcs = await resp.json();
        funcs.forEach(f => {
            const opt = document.createElement('option');
            opt.value = f.name;
            opt.textContent = f.name;
            select.appendChild(opt);
            if (f.name && f.dim) FUNC_DIM_MAP[f.name] = f.dim;
        });
    } catch (e) {
        select.innerHTML = '<option value="">加载失败</option>';
    }
    // 监听函数选择变化，自动设置dimension
    select.onchange = function() {
        // 你可以在这里做其它联动
    };
    // 初始化一次
    if (select.value && FUNC_DIM_MAP[select.value]) {
        const dimInput = document.querySelector('input[name="dim"], select[name="dim"]');
        if (dimInput) dimInput.value = FUNC_DIM_MAP[select.value];
    }
}

// 错误提示
function showFxdefError(msg) {
    const alert = document.getElementById('fxdefErrorAlert');
    alert.textContent = msg;
    alert.style.display = 'block';
}
function hideFxdefError() {
    document.getElementById('fxdefErrorAlert').style.display = 'none';
} 

function updateStepInputnames() {
    for (let i = 0; i < pipelineSteps.length; i++) {
        if (i === 0) {
            pipelineSteps[i].inputnames = ["raw"];
        } else {
            pipelineSteps[i].inputnames = [pipelineSteps[i-1].step_name];
        }
    }
} 

// 加载Feature Definitions供Feature Set选择
async function loadFxdefsForFeatureset() {
    const container = document.getElementById('fxdefCheckboxes');
    container.innerHTML = '<div class="text-center"><div class="spinner-border spinner-border-sm" role="status"></div> Loading...</div>';
    
    try {
        const resp = await fetch('/api/fxdefs');
        const fxdefs = await resp.json();
        
        if (fxdefs.length === 0) {
            container.innerHTML = '<p class="text-muted">No feature definitions available. Please add some feature definitions first.</p>';
            return;
        }
        
        let html = '';
        fxdefs.forEach(fxdef => {
            html += `
                <div class="form-check">
                    <input class="form-check-input" type="checkbox" value="${fxdef.id}" id="fxdef_${fxdef.id}">
                    <label class="form-check-label" for="fxdef_${fxdef.id}">
                        <strong>${fxdef.shortname}</strong> (${fxdef.func})
                        <br><small class="text-muted">Pipeline: ${fxdef.pipeline_name || 'N/A'} | Dim: ${fxdef.dim || 'N/A'}</small>
                    </label>
                </div>
            `;
        });
        container.innerHTML = html;
    } catch (e) {
        container.innerHTML = '<p class="text-danger">Failed to load feature definitions</p>';
    }
}

// Feature Set错误提示
function showFeaturesetError(msg) {
    const alert = document.getElementById('featuresetErrorAlert');
    const textSpan = document.getElementById('featuresetErrorText');
    textSpan.textContent = msg;
    alert.style.display = 'block';
}

function hideFeaturesetError() {
    document.getElementById('featuresetErrorAlert').style.display = 'none';
}

// 加载实验列表
async function loadExperiments() {
    try {
        showStatus('Loading experiments...', 'info');
        
        const response = await fetch('/api/experiments');
        const experiments = await response.json();
        
        displayExperiments(experiments);
        showStatus(`Loaded ${experiments.length} experiments`, 'success');
    } catch (error) {
        console.error('Error loading experiments:', error);
        showStatus('Error loading experiments', 'error');
    }
}

// 显示实验列表
function displayExperiments(experiments) {
    const tbody = document.getElementById('experimentsTableBody');
    
    if (experiments.length === 0) {
        tbody.innerHTML = '<tr><td colspan="9" class="text-center text-muted">No experiments found</td></tr>';
        return;
    }
    
    tbody.innerHTML = '';
    
    experiments.forEach(exp => {
        const row = document.createElement('tr');
        
        // 格式化运行时间
        const runTime = new Date(exp.run_time).toLocaleString();
        const duration = exp.duration_seconds ? `${exp.duration_seconds.toFixed(1)}s` : 'N/A';
        
        // 状态徽章
        const statusBadge = getStatusBadge(exp.status);
        
        // 实验类型徽章
        const typeBadge = getTypeBadge(exp.experiment_type);
        
        row.innerHTML = `
            <td>${exp.id}</td>
            <td><strong>${exp.experiment_name || 'Unnamed'}</strong></td>
            <td>${typeBadge}</td>
            <td>${exp.dataset_name || 'N/A'}</td>
            <td>${exp.feature_set_name || 'N/A'}</td>
            <td>${statusBadge}</td>
            <td>${runTime}</td>
            <td>${duration}</td>
            <td>
                <button class="btn btn-outline-primary btn-sm" onclick="showExperimentDetails(${exp.id})">
                    <i class="bi bi-eye"></i> View
                </button>
            </td>
        `;
        
        tbody.appendChild(row);
    });
}

// 获取状态徽章
function getStatusBadge(status) {
    const badges = {
        'completed': '<span class="badge bg-success">Completed</span>',
        'running': '<span class="badge bg-warning">Running</span>',
        'failed': '<span class="badge bg-danger">Failed</span>'
    };
    return badges[status] || `<span class="badge bg-secondary">${status}</span>`;
}

// 获取类型徽章
function getTypeBadge(type) {
    const badges = {
        'correlation': '<span class="badge bg-info">Correlation</span>',
        'classification': '<span class="badge bg-primary">Classification</span>',
        'feature_selection': '<span class="badge bg-warning">Feature Selection</span>',
        'feature_statistics': '<span class="badge bg-secondary">Statistics</span>'
    };
    return badges[type] || `<span class="badge bg-secondary">${type}</span>`;
}

// 解析实验摘要文本，提取结构化信息
function parseExperimentSummary(summaryText) {
    if (!summaryText) return {};
    
    const result = {
        dataset_overview: {},
        distribution_analysis: {},
        outlier_analysis: {},
        top_features: [],
        key_findings: []
    };
    
    const lines = summaryText.split('\n');
    let currentSection = '';
    
    for (let line of lines) {
        line = line.trim();
        if (!line) continue;
        
        // 检测章节标题
        if (line.includes('Dataset Overview:')) {
            currentSection = 'dataset_overview';
            continue;
        } else if (line.includes('Distribution Analysis:')) {
            currentSection = 'distribution_analysis';
            continue;
        } else if (line.includes('Outlier Analysis:')) {
            currentSection = 'outlier_analysis';
            continue;
        } else if (line.includes('Top 10 Most Important Features:')) {
            currentSection = 'top_features';
            continue;
        } else if (line.includes('Key Findings:')) {
            currentSection = 'key_findings';
            continue;
        }
        
        // 解析具体内容
        if (line.startsWith('-') || line.startsWith('•')) {
            const content = line.substring(1).trim();
            
            if (currentSection === 'dataset_overview') {
                if (content.includes('Total features analyzed:')) {
                    result.dataset_overview.total_features = content.split(':')[1].trim();
                } else if (content.includes('Total samples:')) {
                    result.dataset_overview.total_samples = content.split(':')[1].trim();
                }
            } else if (currentSection === 'distribution_analysis') {
                if (content.includes('Normal-like distributions:')) {
                    result.distribution_analysis.normal_count = content.split(':')[1].trim();
                } else if (content.includes('Non-normal distributions:')) {
                    result.distribution_analysis.non_normal_count = content.split(':')[1].trim();
                }
            } else if (currentSection === 'outlier_analysis') {
                if (content.includes('Features with >10% outliers:')) {
                    result.outlier_analysis.features_with_outliers = content.split(':')[1].trim();
                } else if (content.includes('Average outlier percentage:')) {
                    result.outlier_analysis.avg_outlier_percentage = content.split(':')[1].trim();
                }
            } else if (currentSection === 'top_features') {
                // 解析排名特征
                const match = content.match(/^(\d+)\.\s+(.+)$/);
                if (match) {
                    result.top_features.push({
                        rank: match[1],
                        feature: match[2]
                    });
                }
            } else if (currentSection === 'key_findings') {
                result.key_findings.push(content);
            }
        }
    }
    
    return result;
}

// 生成结构化的摘要HTML
function generateStructuredSummaryHtml(parsedSummary, experimentType) {
    let html = '';
    
    // 数据集概览
    if (parsedSummary.dataset_overview.total_features || parsedSummary.dataset_overview.total_samples) {
        html += `
            <div class="mb-3">
                <h6 class="text-primary">Dataset Overview</h6>
                <div class="row">
        `;
        if (parsedSummary.dataset_overview.total_features) {
            html += `
                <div class="col-md-6">
                    <div class="text-center">
                        <h5 class="text-primary">${parsedSummary.dataset_overview.total_features}</h5>
                        <small>Total Features</small>
                    </div>
                </div>
            `;
        }
        if (parsedSummary.dataset_overview.total_samples) {
            html += `
                <div class="col-md-6">
                    <div class="text-center">
                        <h5 class="text-info">${parsedSummary.dataset_overview.total_samples}</h5>
                        <small>Total Samples</small>
                    </div>
                </div>
            `;
        }
        html += `
                </div>
            </div>
        `;
    }
    
    // 分布分析
    if (parsedSummary.distribution_analysis.normal_count || parsedSummary.distribution_analysis.non_normal_count) {
        html += `
            <div class="mb-3">
                <h6 class="text-success">Distribution Analysis</h6>
                <div class="row">
        `;
        if (parsedSummary.distribution_analysis.normal_count) {
            html += `
                <div class="col-md-6">
                    <div class="text-center">
                        <h5 class="text-success">${parsedSummary.distribution_analysis.normal_count}</h5>
                        <small>Normal-like Distributions</small>
                    </div>
                </div>
            `;
        }
        if (parsedSummary.distribution_analysis.non_normal_count) {
            html += `
                <div class="col-md-6">
                    <div class="text-center">
                        <h5 class="text-warning">${parsedSummary.distribution_analysis.non_normal_count}</h5>
                        <small>Non-normal Distributions</small>
                    </div>
                </div>
            `;
        }
        html += `
                </div>
            </div>
        `;
    }
    
    // 异常值分析
    if (parsedSummary.outlier_analysis.features_with_outliers || parsedSummary.outlier_analysis.avg_outlier_percentage) {
        html += `
            <div class="mb-3">
                <h6 class="text-warning">Outlier Analysis</h6>
                <div class="row">
        `;
        if (parsedSummary.outlier_analysis.features_with_outliers) {
            html += `
                <div class="col-md-6">
                    <div class="text-center">
                        <h5 class="text-warning">${parsedSummary.outlier_analysis.features_with_outliers}</h5>
                        <small>Features with >10% Outliers</small>
                    </div>
                </div>
            `;
        }
        if (parsedSummary.outlier_analysis.avg_outlier_percentage) {
            html += `
                <div class="col-md-6">
                    <div class="text-center">
                        <h5 class="text-info">${parsedSummary.outlier_analysis.avg_outlier_percentage}</h5>
                        <small>Average Outlier Percentage</small>
                    </div>
                </div>
            `;
        }
        html += `
                </div>
            </div>
        `;
    }
    
    // 重要特征列表
    if (parsedSummary.top_features.length > 0) {
        html += `
            <div class="mb-3">
                <h6 class="text-danger">Top Important Features</h6>
                <div class="list-group list-group-flush">
        `;
        parsedSummary.top_features.slice(0, 5).forEach(feature => {
            html += `
                <div class="list-group-item d-flex justify-content-between align-items-center py-2">
                    <span class="badge bg-primary me-2">${feature.rank}</span>
                    <small class="text-muted">${feature.feature}</small>
                </div>
            `;
        });
        if (parsedSummary.top_features.length > 5) {
            html += `
                <div class="list-group-item text-center py-1">
                    <small class="text-muted">... and ${parsedSummary.top_features.length - 5} more</small>
                </div>
            `;
        }
        html += `
                </div>
            </div>
        `;
    }
    
    // 关键发现
    if (parsedSummary.key_findings.length > 0) {
        html += `
            <div class="mb-3">
                <h6 class="text-secondary">Key Findings</h6>
                <ul class="list-unstyled">
        `;
        parsedSummary.key_findings.forEach(finding => {
            html += `<li class="mb-1"><small class="text-muted">• ${finding}</small></li>`;
        });
        html += `
                </ul>
            </div>
        `;
    }
    
    return html;
}

async function showExperimentDetails(experimentId) {
    try {
        const modal = new bootstrap.Modal(document.getElementById('experimentModal'));
        modal.show();
        
        // 并行获取实验详情和统计信息
        const [detailsResponse, summaryResponse] = await Promise.all([
            fetch(`/api/experiment_details/${experimentId}`),
            fetch(`/api/experiment_summary/${experimentId}`)
        ]);
        
        const details = await detailsResponse.json();
        const summary = await summaryResponse.json();
        
        // 显示实验信息
        const exp = details.experiment;
        document.getElementById('experimentInfo').innerHTML = `
            <p><strong>Name:</strong> ${exp.experiment_name || 'Unnamed'}</p>
            <p><strong>Type:</strong> ${getTypeBadge(exp.experiment_type)}</p>
            <p><strong>Dataset:</strong> ${exp.dataset_name || 'N/A'}</p>
            <p><strong>Feature Set:</strong> ${exp.feature_set_name || 'N/A'}</p>
            <p><strong>Status:</strong> ${getStatusBadge(exp.status)}</p>
            <p><strong>Run Time:</strong> ${new Date(exp.run_time).toLocaleString()}</p>
            <p><strong>Duration:</strong> ${exp.duration_seconds ? `${exp.duration_seconds.toFixed(1)}s` : 'N/A'}</p>
            ${exp.notes ? `<p><strong>Notes:</strong> ${exp.notes}</p>` : ''}
        `;
        
        // 解析摘要文本并生成结构化显示
        let statsHtml = '';
        if (exp.summary) {
            const parsedSummary = parseExperimentSummary(exp.summary);
            statsHtml = generateStructuredSummaryHtml(parsedSummary, exp.experiment_type);
        }
        
        // 如果没有解析到结构化信息，回退到原来的统计显示
        if (!statsHtml) {
            const stats = summary.statistics;
            
            if (exp.experiment_type === 'correlation') {
                statsHtml = `
                    <div class="row">
                        <div class="col-md-3">
                            <div class="text-center">
                                <h4 class="text-primary">${stats.total_features}</h4>
                                <small>Total Features</small>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="text-center">
                                <h4 class="text-success">${stats.significant_features}</h4>
                                <small>Significant Features</small>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="text-center">
                                <h4 class="text-info">${stats.avg_correlation?.toFixed(3) || 'N/A'}</h4>
                                <small>Avg Correlation</small>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="text-center">
                                <h4 class="text-warning">${stats.max_correlation?.toFixed(3) || 'N/A'}</h4>
                                <small>Max Correlation</small>
                            </div>
                        </div>
                    </div>
                `;
            } else if (exp.experiment_type === 'classification') {
                statsHtml = `
                    <div class="row">
                        <div class="col-md-3">
                            <div class="text-center">
                                <h4 class="text-primary">${stats.total_features}</h4>
                                <small>Total Features</small>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="text-center">
                                <h4 class="text-success">${stats.important_features}</h4>
                                <small>Important Features</small>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="text-center">
                                <h4 class="text-info">${stats.avg_importance?.toFixed(3) || 'N/A'}</h4>
                                <small>Avg Importance</small>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="text-center">
                                <h4 class="text-warning">${stats.max_importance?.toFixed(3) || 'N/A'}</h4>
                                <small>Max Importance</small>
                            </div>
                        </div>
                    </div>
                `;
            } else {
                statsHtml = `
                    <div class="row">
                        <div class="col-md-3">
                            <div class="text-center">
                                <h4 class="text-primary">${stats.total_features}</h4>
                                <small>Total Features</small>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="text-center">
                                <h4 class="text-info">${stats.avg_metric?.toFixed(3) || 'N/A'}</h4>
                                <small>Avg Metric</small>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="text-center">
                                <h4 class="text-warning">${stats.max_metric?.toFixed(3) || 'N/A'}</h4>
                                <small>Max Metric</small>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="text-center">
                                <h4 class="text-secondary">${stats.min_metric?.toFixed(3) || 'N/A'}</h4>
                                <small>Min Metric</small>
                            </div>
                        </div>
                    </div>
                `;
            }
        }
        
        document.getElementById('experimentStats').innerHTML = statsHtml;
        
        // 显示特征结果
        const results = details.feature_results;
        document.getElementById('experimentResultsHeader').innerHTML = 
            `Feature Results (${results.length})`;
        
        const resultsBody = document.getElementById('experimentResultsBody');
        if (results.length > 0) {
            let resultsHtml = '';
            results.forEach(result => {
                const significanceBadge = result.significance_level ? 
                    `<span class="badge bg-${getSignificanceColor(result.significance_level)}">${result.significance_level}</span>` : 
                    'N/A';
                
                resultsHtml += `
                    <tr>
                        <td>${result.rank_position || 'N/A'}</td>
                        <td><strong>${result.feature_shortname || result.feature_name}</strong></td>
                        <td>${result.feature_channels || 'N/A'}</td>
                        <td>${result.target_variable}</td>
                        <td>${result.metric_name}</td>
                        <td><code>${result.metric_value?.toFixed(4) || 'N/A'}</code></td>
                        <td>${significanceBadge}</td>
                    </tr>
                `;
            });
            resultsBody.innerHTML = resultsHtml;
        } else {
            resultsBody.innerHTML = '<tr><td colspan="7" class="text-center text-muted">No feature results found</td></tr>';
        }
        
    } catch (error) {
        console.error('Error loading experiment details:', error);
        showStatus('Error loading experiment details', 'error');
    }
}



// 初始化Feature Set DAG的Cytoscape.js可视化
function initializeFeaturesetCytoscape(cytoscapeData) {
    const container = document.getElementById('featureset-cytoscape');
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
                    'font-size': '12px',
                    'font-family': 'Consolas, Menlo, Monaco, "Fira Mono", "Roboto Mono", "Courier New", monospace',
                    'text-valign': 'center',
                    'text-halign': 'center',
                    'width': '120px',
                    'height': '50px',
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
                selector: 'node.process',
                style: {
                    'background-color': '#ADD8E6'
                }
            },
            {
                selector: 'edge',
                style: {
                    'width': 2,
                    'line-color': '#666',
                    'target-arrow-color': '#666',
                    'target-arrow-shape': 'triangle',
                    'curve-style': 'bezier',
                    'arrow-scale': 1.2
                }
            }
        ],
        layout: {
            name: 'dagre',
            rankDir: 'TB',
            nodeSep: 40,
            edgeSep: 20,
            rankSep: 60,
            padding: 15
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
    document.getElementById('cy-zoom-in-fs').onclick = () => cy.zoom({ level: cy.zoom() * 1.2, renderedPosition: { x: cy.width()/2, y: cy.height()/2 } });
    document.getElementById('cy-zoom-out-fs').onclick = () => cy.zoom({ level: cy.zoom() / 1.2, renderedPosition: { x: cy.width()/2, y: cy.height()/2 } });
    document.getElementById('cy-zoom-fit-fs').onclick = () => { cy.fit(); cy.center(); };
}

// 显示新增实验弹窗
function showAddExperimentModal() {
    const modal = new bootstrap.Modal(document.getElementById('addExperimentModal'));
    
    // 加载数据集和特征集
    loadDatasetsForExperiment();
    loadFeatureSetsForExperiment();
    
    modal.show();
}

// 为实验弹窗加载数据集
async function loadDatasetsForExperiment() {
    try {
        const response = await fetch('/api/datasets');
        const datasets = await response.json();
        
        const select = document.querySelector('#addExperimentModal select[name="dataset_id"]');
        select.innerHTML = '<option value="">Select dataset...</option>';
        
        datasets.forEach(dataset => {
            const option = document.createElement('option');
            option.value = dataset.id;
            option.textContent = dataset.name;
            select.appendChild(option);
        });
    } catch (error) {
        console.error('Error loading datasets:', error);
    }
}

// 为实验弹窗加载特征集
async function loadFeatureSetsForExperiment() {
    try {
        const response = await fetch('/api/feature_sets');
        const featureSets = await response.json();
        
        const select = document.querySelector('#addExperimentModal select[name="feature_set_id"]');
        select.innerHTML = '<option value="">Select feature set...</option>';
        
        featureSets.forEach(fs => {
            const option = document.createElement('option');
            option.value = fs.id;
            option.textContent = fs.name;
            select.appendChild(option);
        });
    } catch (error) {
        console.error('Error loading feature sets:', error);
    }
}

// 根据实验类型加载参数配置
async function loadExperimentParams(experimentType) {
    const container = document.getElementById('experimentParamsContainer');
    
    if (!experimentType) {
        container.innerHTML = '';
        return;
    }
    
    // 根据实验类型显示不同的参数配置
    let paramsHtml = '';
    
    switch (experimentType) {
        case 'correlation':
            paramsHtml = `
                <div class="card">
                    <div class="card-header">
                        <h6 class="mb-0">Correlation Parameters</h6>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-6">
                                <label class="form-label">Target Variables</label>
                                <input type="text" name="target_vars" class="form-control" 
                                       placeholder="age,sex" value="age,sex">
                                <small class="text-muted">Comma-separated list of target variables</small>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Correlation Method</label>
                                <select name="method" class="form-select">
                                    <option value="pearson">Pearson</option>
                                    <option value="spearman">Spearman</option>
                                </select>
                            </div>
                        </div>
                        <div class="row mt-3">
                            <div class="col-md-6">
                                <label class="form-label">Minimum Correlation</label>
                                <input type="number" name="min_corr" class="form-control" 
                                       placeholder="0.3" value="0.3" step="0.1" min="0" max="1">
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Top N Results</label>
                                <input type="number" name="top_n" class="form-control" 
                                       placeholder="20" value="20" min="1">
                            </div>
                        </div>
                    </div>
                </div>
            `;
            break;
            
        case 'classification':
            paramsHtml = `
                <div class="card">
                    <div class="card-header">
                        <h6 class="mb-0">Classification Parameters</h6>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-6">
                                <label class="form-label">Target Variable</label>
                                <input type="text" name="target_var" class="form-control" 
                                       placeholder="age_group" value="age_group">
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Age Threshold</label>
                                <input type="number" name="age_threshold" class="form-control" 
                                       placeholder="65" value="65">
                            </div>
                        </div>
                        <div class="row mt-3">
                            <div class="col-md-6">
                                <label class="form-label">Test Size</label>
                                <input type="number" name="test_size" class="form-control" 
                                       placeholder="0.2" value="0.2" step="0.1" min="0.1" max="0.9">
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Cross-validation Folds</label>
                                <input type="number" name="n_splits" class="form-control" 
                                       placeholder="5" value="5" min="2">
                            </div>
                        </div>
                    </div>
                </div>
            `;
            break;
            
        case 'feature_selection':
            paramsHtml = `
                <div class="card">
                    <div class="card-header">
                        <h6 class="mb-0">Feature Selection Parameters</h6>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-6">
                                <label class="form-label">Target Variable</label>
                                <input type="text" name="target_var" class="form-control" 
                                       placeholder="age" value="age">
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Number of Features</label>
                                <input type="number" name="n_features" class="form-control" 
                                       placeholder="20" value="20" min="1">
                            </div>
                        </div>
                        <div class="row mt-3">
                            <div class="col-md-6">
                                <label class="form-label">Variance Threshold</label>
                                <input type="number" name="variance_threshold" class="form-control" 
                                       placeholder="0.01" value="0.01" step="0.01" min="0">
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Correlation Threshold</label>
                                <input type="number" name="correlation_threshold" class="form-control" 
                                       placeholder="0.95" value="0.95" step="0.05" min="0" max="1">
                            </div>
                        </div>
                    </div>
                </div>
            `;
            break;
            
        case 'feature_statistics':
            paramsHtml = `
                <div class="card">
                    <div class="card-header">
                        <h6 class="mb-0">Feature Statistics Parameters</h6>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-6">
                                <label class="form-label">Outlier Method</label>
                                <select name="outlier_method" class="form-select">
                                    <option value="iqr">IQR</option>
                                    <option value="zscore">Z-Score</option>
                                </select>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Outlier Threshold</label>
                                <input type="number" name="outlier_threshold" class="form-control" 
                                       placeholder="1.5" value="1.5" step="0.1" min="0">
                            </div>
                        </div>
                        <div class="row mt-3">
                            <div class="col-md-6">
                                <label class="form-label">Top N Features</label>
                                <input type="number" name="top_n_features" class="form-control" 
                                       placeholder="20" value="20" min="1">
                            </div>
                        </div>
                    </div>
                </div>
            `;
            break;
    }
    
    container.innerHTML = paramsHtml;
}

// 提交新增实验
async function submitAddExperiment() {
    const form = document.getElementById('addExperimentForm');
    const formData = new FormData(form);
    
    // 收集基本参数
    const data = {
        experiment_type: formData.get('experiment_type'),
        dataset_id: formData.get('dataset_id'),
        feature_set_id: formData.get('feature_set_id'),
        experiment_name: formData.get('experiment_name'),
        notes: formData.get('notes'),
        parameters: {}
    };
    
    // 收集实验特定参数
    const experimentType = data.experiment_type;
    const paramsContainer = document.getElementById('experimentParamsContainer');
    const paramInputs = paramsContainer.querySelectorAll('input, select');
    
    paramInputs.forEach(input => {
        if (input.name && input.value) {
            // 处理特殊参数类型
            if (input.name === 'target_vars') {
                data.parameters[input.name] = input.value.split(',').map(v => v.trim());
            } else if (input.type === 'number') {
                data.parameters[input.name] = parseFloat(input.value);
            } else {
                data.parameters[input.name] = input.value;
            }
        }
    });
    
    // 验证必需参数
    if (!data.experiment_type || !data.dataset_id || !data.feature_set_id) {
        showExperimentError('Please fill in all required fields');
        return;
    }
    
    try {
        // 显示进度
        showProgress(true);
        showStatus('Starting experiment...', 'info');
        
        const response = await fetch('/api/run_experiment', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (response.ok) {
            showStatus(`Experiment started successfully! ID: ${result.experiment_id}`, 'success');
            
            // 关闭弹窗
            const modal = bootstrap.Modal.getInstance(document.getElementById('addExperimentModal'));
            modal.hide();
            
            // 刷新实验列表
            await loadExperiments();
            
            // 开始监控实验状态
            if (result.experiment_id) {
                startExperimentMonitoring(result.experiment_id);
            }
            
        } else {
            showExperimentError(result.error || 'Failed to start experiment');
        }
        
    } catch (error) {
        console.error('Error starting experiment:', error);
        showExperimentError('Failed to start experiment: ' + error.message);
    } finally {
        showProgress(false);
    }
}

// 监控实验状态
function startExperimentMonitoring(experimentId) {
    const checkStatus = async () => {
        try {
            const response = await fetch(`/api/experiment_status/${experimentId}`);
            const status = await response.json();
            
            if (response.ok) {
                if (status.status === 'running') {
                    const duration = status.duration ? Math.floor(status.duration) : 0;
                    showStatus(`Experiment ${experimentId} is running... (${duration}s)`, 'info');
                    
                    // 继续监控
                    setTimeout(checkStatus, 5000); // 每5秒检查一次
                } else if (status.status === 'completed') {
                    const duration = status.duration ? Math.floor(status.duration) : 0;
                    showStatus(`Experiment ${experimentId} completed successfully! (${duration}s)`, 'success');
                    
                    // 刷新实验列表
                    await loadExperiments();
                } else if (status.status === 'failed') {
                    showStatus(`Experiment ${experimentId} failed: ${status.notes}`, 'error');
                    
                    // 刷新实验列表
                    await loadExperiments();
                }
            }
        } catch (error) {
            console.error('Error checking experiment status:', error);
        }
    };
    
    // 开始监控
    checkStatus();
}

// 显示实验错误
function showExperimentError(message) {
    const alert = document.getElementById('experimentErrorAlert');
    const text = document.getElementById('experimentErrorText');
    text.textContent = message;
    alert.style.display = 'block';
}

// 隐藏实验错误
function hideExperimentError() {
    const alert = document.getElementById('experimentErrorAlert');
    alert.style.display = 'none';
}

// Feature Extraction 相关函数

// 显示Feature Extraction页面
function showFeatureExtraction() {
    console.log('showFeatureExtraction called');
    hideAllViews();
    document.getElementById('featureExtractionView').style.display = 'block';
    
    // 设置活动导航按钮
    const navBtn = document.getElementById('navFeatureExtractionBtn');
    if (navBtn) {
        setActiveNavButton(navBtn);
    }
    
    updateBreadcrumb('feature-extraction');
    
    // 延迟加载数据，确保DOM已经渲染
    setTimeout(() => {
        console.log('Loading extraction data...');
        loadExtractionDatasets();
        loadExtractionFeatureSets();
        loadFeatureExtractionTasks();
    }, 100);
}

// 加载提取任务的数据集
async function loadExtractionDatasets() {
    try {
        const response = await fetch('/api/datasets');
        const datasets = await response.json();
        
        console.log('Datasets loaded:', datasets);
        
        const select = document.getElementById('extractionDatasetSelect');
        if (!select) {
            console.error('extractionDatasetSelect element not found');
            return;
        }
        
        select.innerHTML = '<option value="">Choose dataset...</option>';
        
        if (datasets && datasets.length > 0) {
            datasets.forEach(dataset => {
                const option = document.createElement('option');
                option.value = dataset.id;
                option.textContent = dataset.name;
                select.appendChild(option);
            });
        } else {
            console.log('No datasets found in database');
        }
    } catch (error) {
        console.error('Error loading datasets for extraction:', error);
    }
}

// 加载提取任务的特征集
async function loadExtractionFeatureSets() {
    try {
        const response = await fetch('/api/feature_sets');
        const featureSets = await response.json();
        
        console.log('Feature sets loaded:', featureSets);
        
        const select = document.getElementById('extractionFeatureSetSelect');
        if (!select) {
            console.error('extractionFeatureSetSelect element not found');
            return;
        }
        
        select.innerHTML = '<option value="">Choose feature set...</option>';
        
        if (featureSets && featureSets.length > 0) {
            featureSets.forEach(fs => {
                const option = document.createElement('option');
                option.value = fs.id;
                option.textContent = fs.name;
                select.appendChild(option);
            });
        } else {
            console.log('No feature sets found in database');
        }
    } catch (error) {
        console.error('Error loading feature sets for extraction:', error);
    }
}

// 加载特征提取任务列表
async function loadFeatureExtractionTasks() {
    try {
        const response = await fetch('/api/feature_extraction_tasks');
        const tasks = await response.json();
        
        displayFeatureExtractionTasks(tasks);
    } catch (error) {
        console.error('Error loading feature extraction tasks:', error);
        showStatus('Error loading extraction tasks', 'error');
    }
}

// 显示特征提取任务列表
function displayFeatureExtractionTasks(tasks) {
    const tbody = document.getElementById('extractionTableBody');
    
    if (tasks.length === 0) {
        tbody.innerHTML = '<tr><td colspan="8" class="text-center text-muted">No extraction tasks found</td></tr>';
        return;
    }
    
    let html = '';
    tasks.forEach(task => {
        const statusBadge = getExtractionStatusBadge(task.status);
        const progressBar = getProgressBar(task.processed_recordings, task.total_recordings);
        const duration = task.duration_seconds ? `${Math.floor(task.duration_seconds)}s` : 'N/A';
        const startTime = task.start_time ? new Date(task.start_time).toLocaleString() : 'N/A';
        const downloadBtn = task.status === 'completed' ? 
            `<button class="btn btn-sm btn-outline-primary" onclick="downloadExtractionResult(${task.id})">
                <i class="bi bi-download"></i> Download
            </button>` : '';
        
        html += `
            <tr>
                <td>${task.id}</td>
                <td>${task.dataset_name || 'N/A'}</td>
                <td>${task.feature_set_name || 'N/A'}</td>
                <td>${statusBadge}</td>
                <td>${progressBar}</td>
                <td>${startTime}</td>
                <td>${duration}</td>
                <td>${downloadBtn}</td>
            </tr>
        `;
    });
    
    tbody.innerHTML = html;
}

// 获取提取任务状态徽章
function getExtractionStatusBadge(status) {
    const badges = {
        'running': '<span class="badge bg-warning">Running</span>',
        'completed': '<span class="badge bg-success">Completed</span>',
        'failed': '<span class="badge bg-danger">Failed</span>'
    };
    return badges[status] || `<span class="badge bg-secondary">${status}</span>`;
}

// 获取进度条
function getProgressBar(processed, total) {
    if (total === 0) return '<div class="progress"><div class="progress-bar" style="width: 0%">0%</div></div>';
    
    const percentage = Math.round((processed / total) * 100);
    return `
        <div class="progress">
            <div class="progress-bar" style="width: ${percentage}%">
                ${processed}/${total} (${percentage}%)
            </div>
        </div>
    `;
}

// 开始特征提取
async function startFeatureExtraction() {
    const datasetId = document.getElementById('extractionDatasetSelect').value;
    const featureSetId = document.getElementById('extractionFeatureSetSelect').value;
    
    if (!datasetId || !featureSetId) {
        showStatus('Please select both dataset and feature set', 'error');
        return;
    }
    
    try {
        showProgress(true);
        showStatus('Starting feature extraction...', 'info');
        
        const response = await fetch('/api/start_feature_extraction', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                dataset_id: parseInt(datasetId),
                feature_set_id: parseInt(featureSetId),
                task_name: `Extraction ${new Date().toLocaleString()}`
            })
        });
        
        const result = await response.json();
        
        if (response.ok) {
            showStatus(`Feature extraction started! Task ID: ${result.task_id}`, 'success');
            
            // 刷新任务列表
            await loadFeatureExtractionTasks();
            
            // 开始监控任务状态
            if (result.task_id) {
                startExtractionMonitoring(result.task_id);
            }
            
        } else {
            showStatus(result.error || 'Failed to start extraction', 'error');
        }
        
    } catch (error) {
        console.error('Error starting feature extraction:', error);
        showStatus('Failed to start extraction: ' + error.message, 'error');
    } finally {
        showProgress(false);
    }
}

// 监控提取任务状态
function startExtractionMonitoring(taskId) {
    const checkStatus = async () => {
        try {
            const response = await fetch(`/api/feature_extraction_status/${taskId}`);
            const status = await response.json();
            
            if (response.ok) {
                if (status.status === 'running') {
                    const progress = Math.round(status.progress);
                    const duration = status.duration ? Math.floor(status.duration) : 0;
                    showStatus(`Extraction ${taskId} is running... ${progress}% (${duration}s)`, 'info');
                    
                    // 继续监控
                    setTimeout(checkStatus, 3000); // 每3秒检查一次
                } else if (status.status === 'completed') {
                    const duration = status.duration ? Math.floor(status.duration) : 0;
                    showStatus(`Extraction ${taskId} completed successfully! (${duration}s)`, 'success');
                    
                    // 刷新任务列表
                    await loadFeatureExtractionTasks();
                } else if (status.status === 'failed') {
                    showStatus(`Extraction ${taskId} failed: ${status.notes}`, 'error');
                    
                    // 刷新任务列表
                    await loadFeatureExtractionTasks();
                }
            }
        } catch (error) {
            console.error('Error checking extraction status:', error);
        }
    };
    
    // 开始监控
    checkStatus();
}

// 下载提取结果
function downloadExtractionResult(taskId) {
    const link = document.createElement('a');
    link.href = `/api/download_extraction_result/${taskId}`;
    link.download = `extraction_${taskId}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// DAG Execution Functions
function bindDagExecutionEvents(featureSetId) {
    // 绑定执行按钮
    const executeBtn = document.getElementById('dagExecuteBtn');
    const statusBtn = document.getElementById('dagStatusBtn');
    
    if (executeBtn) {
        executeBtn.onclick = () => executeDag(featureSetId);
    }
    
    if (statusBtn) {
        statusBtn.onclick = () => showDagStatus(featureSetId);
    }
}

async function executeDag(featureSetId) {
    try {
        showStatus('Executing DAG...', 'info');
        showProgress(true);
        
        const response = await fetch(`/api/execute_dag/${featureSetId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                recording_id: 22  // 可以改为可配置的
            })
        });
        
        const result = await response.json();
        
        if (response.ok) {
            showStatus('DAG execution completed successfully', 'success');
            showDagStatus(featureSetId, result);
        } else {
            showStatus(`DAG execution failed: ${result.error}`, 'error');
        }
    } catch (error) {
        console.error('Error executing DAG:', error);
        showStatus('Error executing DAG', 'error');
    } finally {
        showProgress(false);
    }
}

async function showDagStatus(featureSetId, executionData = null) {
    try {
        let data = executionData;
        
        if (!data) {
            const response = await fetch(`/api/dag_status/${featureSetId}`);
            data = await response.json();
        }
        
        if (data) {
            // 更新摘要信息
            document.getElementById('totalNodes').textContent = data.total_nodes || 0;
            document.getElementById('successNodes').textContent = data.status_counts?.success || 0;
            document.getElementById('failedNodes').textContent = data.status_counts?.failed || 0;
            document.getElementById('totalDuration').textContent = `${(data.total_duration || 0).toFixed(2)}s`;
            
            // 更新节点详情表格
            const tbody = document.getElementById('nodeDetailsTableBody');
            tbody.innerHTML = '';
            
            if (data.node_details) {
                Object.entries(data.node_details).forEach(([nodeId, details]) => {
                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <td><code>${nodeId}</code></td>
                        <td>${details.function || 'N/A'}</td>
                        <td>${getNodeStatusBadge(details.status)}</td>
                        <td>${details.duration ? `${details.duration.toFixed(3)}s` : 'N/A'}</td>
                        <td>${details.pipeline_count || 0}</td>
                        <td>${details.fxdef_count || 0}</td>
                        <td>${details.error || '-'}</td>
                    `;
                    tbody.appendChild(row);
                });
            }
            
            // 显示模态框
            const modal = new bootstrap.Modal(document.getElementById('dagStatusModal'));
            modal.show();
        }
    } catch (error) {
        console.error('Error loading DAG status:', error);
        showStatus('Error loading DAG status', 'error');
    }
}

function getNodeStatusBadge(status) {
    const statusMap = {
        'success': 'success',
        'failed': 'danger',
        'running': 'warning',
        'pending': 'secondary'
    };
    
    const badgeClass = statusMap[status] || 'secondary';
    return `<span class="badge bg-${badgeClass}">${status}</span>`;
}

async function loadRecordingsForDagSelect() {
    const select = document.getElementById('dagRecordingSelect');
    select.innerHTML = '<option value="">选择Recording...</option>';
    const resp = await fetch('/api/recordings');
    const recordings = await resp.json();
    recordings.forEach(r => {
        select.innerHTML += `<option value="${r.id}">${r.filename} (${(r.duration||0).toFixed(1)}s, ${r.channels}ch, ${r.sampling_rate}Hz)</option>`;
    });
    // 选中第一个或默认22
    select.value = 22;
    // 绑定change事件，显示详细信息
    select.onchange = function() {
        const rec = recordings.find(x => x.id == select.value);
        document.getElementById('dagRecordingInfo').innerText = rec
            ? `文件大小: ${rec.file_size||'未知'}，通道: ${rec.channels}，采样率: ${rec.sampling_rate}Hz，时长: ${rec.duration}s`
            : '';
    };
    select.onchange();
} 