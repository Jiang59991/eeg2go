// app.js - 主入口文件
import { initializeNavigation } from './modules/navigation.js';
import { initializeDatasets } from './modules/datasets.js';
import { initializePipelines } from './modules/pipelines.js';
import { initializeFxdefs } from './modules/fxdefs.js';
import { initializeFeaturesets } from './modules/featuresets.js';
import { initializeExperiments } from './modules/experiments.js';
import { initializeFeatureExtraction } from './modules/feature-extraction.js';
import { initializeTaskQueue } from './modules/task-queue.js';
import { initializeSystemMode } from './modules/system-mode.js';
import { showStatus, showProgress } from './modules/ui-utils.js';

// 全局变量
let STEP_REGISTRY = {};

// 初始化应用
async function initializeApp() {
    try {
        console.log('Initializing application...');
        
        // 加载步骤注册表
        const response = await fetch('/api/steps/registry');
        STEP_REGISTRY = await response.json();
        
        // 初始化各个模块
        initializeNavigation();
        initializeDatasets();
        initializePipelines();
        initializeFxdefs();
        initializeFeaturesets();
        initializeExperiments();
        initializeFeatureExtraction();
        initializeTaskQueue();
        initializeSystemMode();
        
        // 设置事件监听器
        setupEventListeners();
        
        // 等待模块初始化完成后再显示默认页面
        setTimeout(() => {
            // 直接调用模块中的函数，而不是通过 window
            if (typeof window.showDatasets === 'function') {
                window.showDatasets();
            } else {
                console.error('showDatasets function not available');
            }
        }, 100);
        
        console.log('Application initialization completed');
        
    } catch (error) {
        console.error('Failed to initialize app:', error);
        showStatus('Failed to initialize application', 'error');
    }
}

// 删除这些有问题的全局函数定义
// 这些函数应该由各个模块自己定义并暴露到 window

function setupEventListeners() {
    // 全局事件监听器
    document.addEventListener('DOMContentLoaded', function() {
        console.log('DOM fully loaded');
        // 模态框事件监听器
        setupModalEventListeners();
    });
}

function setupModalEventListeners() {
    // 管道模态框
    const addPipelineModal = document.getElementById('addPipelineModal');
    if (addPipelineModal) {
        addPipelineModal.addEventListener('hidden.bs.modal', function() {
            resetPipelineForm();
        });
    }
    
    // 特征定义模态框
    const addFxdefModal = document.getElementById('addFxdefModal');
    if (addFxdefModal) {
        addFxdefModal.addEventListener('hidden.bs.modal', function() {
            resetFxdefForm();
        });
    }
}

// 启动应用
document.addEventListener('DOMContentLoaded', initializeApp);

// 导出全局变量供其他模块使用
window.STEP_REGISTRY = STEP_REGISTRY;