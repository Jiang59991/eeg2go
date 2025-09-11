// modules/navigation.js
import { showDatasets } from './datasets.js';
import { showPipelines } from './pipelines.js';
import { showFxdefs } from './fxdefs.js';
import { showFeaturesets } from './featuresets.js';
import { showFeatureExtraction } from './feature-extraction.js';
import { showExperiments } from './experiments.js';
import { showTasks } from './task-queue.js';

export function initializeNavigation() {
    console.log('Initializing navigation module...');
    
    document.getElementById('navDatasetsBtn').addEventListener('click', showDatasets);
    document.getElementById('navPipelinesBtn').addEventListener('click', showPipelines);
    document.getElementById('navFxdefsBtn').addEventListener('click', showFxdefs);
    document.getElementById('navFeaturesetsBtn').addEventListener('click', showFeaturesets);
    document.getElementById('navFeatureExtractionBtn').addEventListener('click', showFeatureExtraction);
    document.getElementById('navExperimentsBtn').addEventListener('click', showExperiments);
    document.getElementById('navTasksBtn').addEventListener('click', showTasks);
}

export function setActiveNavButton(activeButton) {
    document.querySelectorAll('.nav-buttons .btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    if (activeButton) {
        activeButton.classList.add('active');
    }
}

export function hideAllViews() {
    const views = [
        'datasetsView',
        'recordingsView', 
        'pipelinesView',
        'fxdefsView',
        'featuresetsView',
        'featureExtractionView',
        'experimentsView',
        'tasksView'
    ];
    
    views.forEach(viewId => {
        const view = document.getElementById(viewId);
        if (view) {
            view.style.display = 'none';
        }
    });
}

export function updateBreadcrumb(page) {
    document.querySelectorAll('.breadcrumb-item').forEach(item => {
        item.style.display = 'none';
    });
    
    const breadcrumbMap = {
        'datasets': 'breadcrumb-datasets',
        'recordings': 'breadcrumb-recordings',
        'pipelines': 'breadcrumb-pipelines',
        'fxdefs': 'breadcrumb-fxdefs',
        'featuresets': 'breadcrumb-featuresets',
        'feature-extraction': 'breadcrumb-feature-extraction',
        'experiments': 'breadcrumb-experiments',
        'tasks': 'breadcrumb-tasks'
    };
    
    const breadcrumbId = breadcrumbMap[page];
    if (breadcrumbId) {
        const breadcrumb = document.getElementById(breadcrumbId);
        if (breadcrumb) {
            breadcrumb.style.display = 'block';
        }
    }
}