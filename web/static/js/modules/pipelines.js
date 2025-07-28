// modules/pipelines.js
import { setActiveNavButton, hideAllViews, updateBreadcrumb } from './navigation.js';
import { showStatus } from './ui-utils.js';
import { apiGet } from './api-client.js';

export function initializePipelines() {
    // 管道相关初始化
}

export async function showPipelines() {
    setActiveNavButton(document.getElementById('navPipelinesBtn'));
    hideAllViews();
    document.getElementById('pipelinesView').style.display = 'block';
    updateBreadcrumb('pipelines');
    
    await loadPipelines();
}

export async function loadPipelines() {
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

export function displayPipelines(pipelines) {
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

export async function showPipelineDetails(pipelineId) {
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
export function initializeCytoscape(cytoscapeData) {
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
                    'font-size': '14px',
                    'font-family': 'Consolas, Menlo, Monaco, "Fira Mono", "Roboto Mono", "Courier New", monospace',
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

// 导出到全局作用域
window.showPipelines = showPipelines;
window.showPipelineDetails = showPipelineDetails;