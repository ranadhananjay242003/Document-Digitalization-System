// OCR Web Application JavaScript
class OCRApp {
    constructor() {
        this.currentTaskId = null;
        this.selectedFile = null;
        this.statusCheckInterval = null;
        
        this.initializeElements();
        this.attachEventListeners();
        this.updatePipelineInfo();
    }

    initializeElements() {
        // Sections
        this.uploadSection = document.getElementById('uploadSection');
        this.progressSection = document.getElementById('progressSection');
        this.resultsSection = document.getElementById('resultsSection');
        this.errorSection = document.getElementById('errorSection');

        // Upload elements
        this.fileInput = document.getElementById('fileInput');
        this.uploadArea = document.getElementById('uploadArea');
        this.browseBtn = document.getElementById('browseBtn');
        this.pipelineSelect = document.getElementById('pipelineSelect');
        this.pipelineInfo = document.getElementById('pipelineInfo');

        // Preview elements
        this.imagePreview = document.getElementById('imagePreview');
        this.previewImg = document.getElementById('previewImg');
        this.fileName = document.getElementById('fileName');
        this.fileSize = document.getElementById('fileSize');
        this.processBtn = document.getElementById('processBtn');
        this.cancelBtn = document.getElementById('cancelBtn');

        // Progress elements
        this.progressFill = document.getElementById('progressFill');
        this.progressText = document.getElementById('progressText');
        this.progressPercentage = document.getElementById('progressPercentage');

        // Results elements
        this.lineCount = document.getElementById('lineCount');
        this.processedTime = document.getElementById('processedTime');
        this.extractedText = document.getElementById('extractedText');
        this.downloadBtn = document.getElementById('downloadBtn');
        this.newImageBtn = document.getElementById('newImageBtn');

        // Error elements
        this.errorMessage = document.getElementById('errorMessage');
        this.retryBtn = document.getElementById('retryBtn');

        // Toast container
        this.toastContainer = document.getElementById('toastContainer');
    }

    attachEventListeners() {
        // File input events
        this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        this.browseBtn.addEventListener('click', () => this.fileInput.click());

        // Drag and drop events
        this.uploadArea.addEventListener('click', () => this.fileInput.click());
        this.uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
        this.uploadArea.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        this.uploadArea.addEventListener('drop', (e) => this.handleFileDrop(e));

        // Pipeline selection
        this.pipelineSelect.addEventListener('change', () => this.updatePipelineInfo());

        // Button events
        this.processBtn.addEventListener('click', () => this.processImage());
        this.cancelBtn.addEventListener('click', () => this.cancelSelection());
        this.downloadBtn.addEventListener('click', () => this.downloadPDF());
        this.newImageBtn.addEventListener('click', () => this.resetApp());
        this.retryBtn.addEventListener('click', () => this.resetApp());
    }

    updatePipelineInfo() {
        const pipeline = this.pipelineSelect.value;
        const infoTexts = {
            'advanced': 'Advanced pipeline uses ensemble methods for highest accuracy but takes longer',
            'optimized': 'Optimized pipeline balances speed and accuracy for most use cases',
            'basic': 'Basic pipeline is fastest but may have lower accuracy on complex handwriting'
        };
        this.pipelineInfo.textContent = infoTexts[pipeline];
    }

    handleDragOver(e) {
        e.preventDefault();
        this.uploadArea.classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
    }

    handleFileDrop(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.processFileSelection(files[0]);
        }
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.processFileSelection(file);
        }
    }

    processFileSelection(file) {
        // Validate file type
        const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg'];
        if (!allowedTypes.includes(file.type)) {
            this.showToast('Please select a PNG, JPG, or JPEG image file.', 'error');
            return;
        }

        // Validate file size (16MB limit)
        const maxSize = 16 * 1024 * 1024;
        if (file.size > maxSize) {
            this.showToast('File size must be less than 16MB.', 'error');
            return;
        }

        this.selectedFile = file;
        this.showPreview(file);
    }

    showPreview(file) {
        // Show file info
        this.fileName.textContent = `File: ${file.name}`;
        this.fileSize.textContent = `Size: ${this.formatFileSize(file.size)}`;

        // Show image preview
        const reader = new FileReader();
        reader.onload = (e) => {
            this.previewImg.src = e.target.result;
        };
        reader.readAsDataURL(file);

        // Show preview section
        this.imagePreview.style.display = 'block';
        this.uploadArea.style.display = 'none';
    }

    cancelSelection() {
        this.selectedFile = null;
        this.fileInput.value = '';
        this.imagePreview.style.display = 'none';
        this.uploadArea.style.display = 'block';
    }

    async processImage() {
        if (!this.selectedFile) {
            this.showToast('Please select an image file first.', 'error');
            return;
        }

        const formData = new FormData();
        formData.append('file', this.selectedFile);
        formData.append('pipeline', this.pipelineSelect.value);

        try {
            // Show progress section
            this.showSection('progress');

            // Upload file
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.error || 'Upload failed');
            }

            this.currentTaskId = result.task_id;
            this.showToast(result.message, 'success');

            // Start monitoring progress
            this.startProgressMonitoring();

        } catch (error) {
            this.showError(`Upload failed: ${error.message}`);
        }
    }

    startProgressMonitoring() {
        if (this.statusCheckInterval) {
            clearInterval(this.statusCheckInterval);
        }

        this.statusCheckInterval = setInterval(async () => {
            try {
                const response = await fetch(`/status/${this.currentTaskId}`);
                const status = await response.json();

                if (!response.ok) {
                    throw new Error(status.error || 'Status check failed');
                }

                this.updateProgress(status);

                if (status.status === 'completed') {
                    this.handleProcessingComplete(status.result);
                    clearInterval(this.statusCheckInterval);
                } else if (status.status === 'error') {
                    this.showError(status.message || 'Processing failed');
                    clearInterval(this.statusCheckInterval);
                }

            } catch (error) {
                this.showError(`Status check failed: ${error.message}`);
                clearInterval(this.statusCheckInterval);
            }
        }, 1000); // Check every second
    }

    updateProgress(status) {
        const progress = Math.max(0, Math.min(100, status.progress || 0));
        
        this.progressFill.style.width = `${progress}%`;
        this.progressPercentage.textContent = `${progress}%`;
        this.progressText.textContent = status.message || 'Processing...';
    }

    handleProcessingComplete(result) {
        // Update results display
        this.lineCount.textContent = result.line_count || 0;
        this.processedTime.textContent = this.formatDateTime(result.processed_at);

        // Display extracted text
        this.displayExtractedText(result.lines || []);

        // Show results section
        this.showSection('results');
        this.showToast('Text extraction completed successfully!', 'success');
    }

    displayExtractedText(lines) {
        if (!lines || lines.length === 0) {
            this.extractedText.innerHTML = '<p style="color: #718096; font-style: italic;">No text was extracted from the image.</p>';
            return;
        }

        // Display each line
        const textContent = lines.map((line, index) => {
            return `<div class="text-line">${this.escapeHtml(line)}</div>`;
        }).join('');

        this.extractedText.innerHTML = textContent;
    }

    async downloadPDF() {
        if (!this.currentTaskId) {
            this.showToast('No document available for download.', 'error');
            return;
        }

        try {
            const response = await fetch(`/download/${this.currentTaskId}`);
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Download failed');
            }

            // Create download link
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `ocr_result_${this.currentTaskId}.pdf`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);

            this.showToast('PDF downloaded successfully!', 'success');

        } catch (error) {
            this.showError(`Download failed: ${error.message}`);
        }
    }

    resetApp() {
        // Clear data
        this.currentTaskId = null;
        this.selectedFile = null;
        this.fileInput.value = '';
        
        // Clear intervals
        if (this.statusCheckInterval) {
            clearInterval(this.statusCheckInterval);
        }

        // Reset UI
        this.showSection('upload');
        this.imagePreview.style.display = 'none';
        this.uploadArea.style.display = 'block';
    }

    showSection(section) {
        // Hide all sections
        this.uploadSection.style.display = 'none';
        this.progressSection.style.display = 'none';
        this.resultsSection.style.display = 'none';
        this.errorSection.style.display = 'none';

        // Show selected section
        switch (section) {
            case 'upload':
                this.uploadSection.style.display = 'block';
                break;
            case 'progress':
                this.progressSection.style.display = 'block';
                break;
            case 'results':
                this.resultsSection.style.display = 'block';
                break;
            case 'error':
                this.errorSection.style.display = 'block';
                break;
        }
    }

    showError(message) {
        this.errorMessage.textContent = message;
        this.showSection('error');
        this.showToast(message, 'error');
    }

    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <div style="display: flex; align-items: center; gap: 8px;">
                <i class="fas ${this.getToastIcon(type)}"></i>
                <span>${this.escapeHtml(message)}</span>
            </div>
        `;

        this.toastContainer.appendChild(toast);

        // Show toast with animation
        setTimeout(() => toast.classList.add('show'), 100);

        // Remove toast after 5 seconds
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => {
                if (toast.parentNode) {
                    toast.parentNode.removeChild(toast);
                }
            }, 300);
        }, 5000);
    }

    getToastIcon(type) {
        switch (type) {
            case 'success': return 'fa-check-circle';
            case 'error': return 'fa-exclamation-circle';
            case 'info': default: return 'fa-info-circle';
        }
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    formatDateTime(isoString) {
        if (!isoString) return '-';
        try {
            const date = new Date(isoString);
            return date.toLocaleString();
        } catch {
            return '-';
        }
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new OCRApp();
});

// Prevent default drag behaviors on the document
document.addEventListener('dragover', (e) => e.preventDefault());
document.addEventListener('drop', (e) => e.preventDefault());
