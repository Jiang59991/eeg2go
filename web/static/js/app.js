// app.js
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

let STEP_REGISTRY = {};

async function initializeApp() {
    try {
        console.log('Initializing application...');
        
        const response = await fetch('/api/steps/registry');
        STEP_REGISTRY = await response.json();
        
        initializeNavigation();
        initializeDatasets();
        initializePipelines();
        initializeFxdefs();
        initializeFeaturesets();
        initializeExperiments();
        initializeFeatureExtraction();
        initializeTaskQueue();
        initializeSystemMode();
        
        setupEventListeners();
        
        setTimeout(() => {
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

function setupEventListeners() {
    document.addEventListener('DOMContentLoaded', function() {
        console.log('DOM fully loaded');
        setupModalEventListeners();
    });
}

function setupModalEventListeners() {
    const addPipelineModal = document.getElementById('addPipelineModal');
    if (addPipelineModal) {
        addPipelineModal.addEventListener('hidden.bs.modal', function() {
            resetPipelineForm();
        });
    }
    
    const addFxdefModal = document.getElementById('addFxdefModal');
    if (addFxdefModal) {
        addFxdefModal.addEventListener('hidden.bs.modal', function() {
            resetFxdefForm();
        });
    }
}

document.addEventListener('DOMContentLoaded', initializeApp);

window.STEP_REGISTRY = STEP_REGISTRY;