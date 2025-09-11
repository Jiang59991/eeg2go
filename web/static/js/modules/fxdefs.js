// modules/fxdefs.js
import { setActiveNavButton, hideAllViews, updateBreadcrumb } from './navigation.js';
import { showStatus } from './ui-utils.js';

export function initializeFxdefs() {
    console.log('Initializing fxdefs module...');
}

export function showFxdefs() {
    setActiveNavButton(document.getElementById('navFxdefsBtn'));
    hideAllViews();
    document.getElementById('fxdefsView').style.display = 'block';
    updateBreadcrumb('fxdefs');
    
    loadFxdefs();
}

async function loadFxdefs() {
    try {
        const response = await fetch('/api/fxdefs');
        const fxdefs = await response.json();
        
        const tbody = document.getElementById('fxdefsTableBody');
        if (fxdefs.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" class="text-center text-muted">No feature definitions found</td></tr>';
            return;
        }
        
        tbody.innerHTML = fxdefs.map(fxdef => `
            <tr>
                <td>${fxdef.id}</td>
                <td><strong>${fxdef.shortname}</strong></td>
                <td>${fxdef.ver || 'N/A'}</td>
                <td>${fxdef.dim || 'N/A'}</td>
                <td>${fxdef.pipeline_name || 'N/A'}</td>
                <td>${fxdef.func || 'N/A'}</td>
                <td>
                    <button class="btn btn-outline-primary btn-sm" onclick="showFxdefDetails(${fxdef.id})">
                        <i class="bi bi-eye"></i> View
                    </button>
                </td>
            </tr>
        `).join('');
        
    } catch (error) {
        console.error('Failed to load fxdefs:', error);
        showStatus('Failed to load feature definitions', 'error');
    }
}

export function showFxdefDetails(fxdefId) {
    fetch(`/api/fxdef_details/${fxdefId}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showStatus(`Error: ${data.error}`, 'error');
                return;
            }
            
            document.getElementById('fxdefInfo').innerHTML = `
                <div class="card mb-3">
                    <div class="card-header">Feature Definition Information</div>
                    <div class="card-body">
                        <table class="table table-sm">
                            <tr><td><strong>ID:</strong></td><td>${data.fxdef.id}</td></tr>
                            <tr><td><strong>Short Name:</strong></td><td>${data.fxdef.shortname}</td></tr>
                            <tr><td><strong>Function:</strong></td><td>${data.fxdef.func}</td></tr>
                            <tr><td><strong>Version:</strong></td><td>${data.fxdef.ver || 'N/A'}</td></tr>
                            <tr><td><strong>Dimension:</strong></td><td>${data.fxdef.dim || 'N/A'}</td></tr>
                            <tr><td><strong>Pipeline:</strong></td><td>${data.fxdef.pipeline_name || 'N/A'}</td></tr>
                            <tr><td><strong>Channels:</strong></td><td>${data.fxdef.chans || 'N/A'}</td></tr>
                            <tr><td><strong>Parameters:</strong></td><td>${data.fxdef.params || 'N/A'}</td></tr>
                            <tr><td><strong>Notes:</strong></td><td>${data.fxdef.notes || 'N/A'}</td></tr>
                        </table>
                    </div>
                </div>
            `;
            
            const modal = new bootstrap.Modal(document.getElementById('fxdefModal'));
            modal.show();
        })
        .catch(error => {
            console.error('Failed to load fxdef details:', error);
            showStatus('Failed to load feature definition details', 'error');
        });
}

window.showFxdefs = showFxdefs;
window.showFxdefDetails = showFxdefDetails;