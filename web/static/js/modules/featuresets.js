// modules/featuresets.js
import { setActiveNavButton, hideAllViews, updateBreadcrumb } from './navigation.js';
import { showStatus } from './ui-utils.js';

export function initializeFeaturesets() {
    console.log('Initializing featuresets module...');
}

export function showFeaturesets() {
    setActiveNavButton(document.getElementById('navFeaturesetsBtn'));
    hideAllViews();
    document.getElementById('featuresetsView').style.display = 'block';
    updateBreadcrumb('featuresets');
    
    loadFeaturesets();
}

async function loadFeaturesets() {
    try {
        const response = await fetch('/api/featuresets_detailed');
        const featuresets = await response.json();
        
        const grid = document.getElementById('featuresetsGrid');
        if (featuresets.length === 0) {
            grid.innerHTML = '<div class="col-12"><p class="text-center text-muted">No feature sets found</p></div>';
            return;
        }
        
        grid.innerHTML = '';
        
        featuresets.forEach(fs => {
            const col = document.createElement('div');
            col.className = 'col-lg-4 col-md-6 col-sm-12 mb-3';
            
            col.innerHTML = `
                <div class="dataset-card" onclick="showFeatureSetDetails(${fs.feature_set.id})">
                    <div class="d-flex align-items-center mb-2">
                        <i class="bi bi-collection text-primary me-2"></i>
                        <h5 class="mb-0">${fs.feature_set.name}</h5>
                    </div>
                    <div class="dataset-description">
                        ${fs.feature_set.description || 'No description available'}
                    </div>
                    <div class="dataset-stats">
                        <span><i class="bi bi-gear"></i> Features: ${fs.fxdef_count}</span>
                    </div>
                </div>
            `;
            
            grid.appendChild(col);
        });
        
    } catch (error) {
        console.error('Failed to load featuresets:', error);
        showStatus('Failed to load feature sets', 'error');
    }
}

export function showFeatureSetDetails(featureSetId) {
    fetch(`/api/feature_set_details/${featureSetId}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showStatus(`Error: ${data.error}`, 'error');
                return;
            }
            
            // 填充特征集信息
            document.getElementById('featureSetInfo').innerHTML = `
                <table class="table table-sm">
                    <tr><td><strong>ID:</strong></td><td>${data.feature_set.id}</td></tr>
                    <tr><td><strong>Name:</strong></td><td>${data.feature_set.name}</td></tr>
                    <tr><td><strong>Description:</strong></td><td>${data.feature_set.description || 'N/A'}</td></tr>
                    <tr><td><strong>Features:</strong></td><td>${data.features.length}</td></tr>
                </table>
            `;
            
            // 填充特征列表
            document.getElementById('featuresList').innerHTML = `
                <div class="table-responsive">
                    <table class="table table-sm">
                        <thead>
                            <tr>
                                <th>Short Name</th>
                                <th>Function</th>
                                <th>Channels</th>
                                <th>Pipeline</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${data.features.map(feature => `
                                <tr>
                                    <td><strong>${feature.shortname}</strong></td>
                                    <td>${feature.func}</td>
                                    <td>${feature.chans || 'N/A'}</td>
                                    <td>${feature.pipeline_name || 'N/A'}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            `;
            
            // 显示模态框
            const modal = new bootstrap.Modal(document.getElementById('featureSetModal'));
            modal.show();
            
            // 加载DAG可视化
            loadFeatureSetDAG(featureSetId);
        })
        .catch(error => {
            console.error('Failed to load feature set details:', error);
            showStatus('Failed to load feature set details', 'error');
        });
}

async function loadFeatureSetDAG(featureSetId) {
    try {
        const response = await fetch(`/api/featureset_dag/${featureSetId}`);
        const data = await response.json();
        
        if (data.success && data.cytoscape_data) {
            initializeFeatureSetCytoscape(data.cytoscape_data);
        }
    } catch (error) {
        console.error('Failed to load feature set DAG:', error);
    }
}

function initializeFeatureSetCytoscape(cytoscapeData) {
    const container = document.getElementById('featureset-cytoscape');
    if (!container) return;
    
    // 清除之前的可视化
    container.innerHTML = '';
    
    // 创建Cytoscape实例（类似pipelines.js中的实现）
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
                    'width': '100px',
                    'height': '50px',
                    'border-width': 2,
                    'border-color': '#666',
                    'shape': 'rectangle'
                }
            },
            {
                selector: 'edge',
                style: {
                    'width': 2,
                    'line-color': '#666',
                    'target-arrow-color': '#666',
                    'target-arrow-shape': 'triangle',
                    'curve-style': 'bezier'
                }
            }
        ],
        layout: {
            name: 'dagre',
            rankDir: 'TB',
            nodeSep: 30,
            edgeSep: 20,
            rankSep: 60
        }
    });
    
    cy.fit();
    cy.center();
}

// 导出到全局作用域
window.showFeaturesets = showFeaturesets;
window.showFeatureSetDetails = showFeatureSetDetails;