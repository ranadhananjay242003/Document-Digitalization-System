document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Element References ---
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const browseBtn = document.getElementById('browseBtn');
    const imagePreview = document.getElementById('imagePreview');
    const uploadSection = document.getElementById('uploadSection');
    const progressSection = document.getElementById('progressSection');
    const resultsSection = document.getElementById('resultsSection');
    const errorSection = document.getElementById('errorSection');
    const previewImg = document.getElementById('previewImg');
    const fileName = document.getElementById('fileName');
    const fileSize = document.getElementById('fileSize');
    const processBtn = document.getElementById('processBtn');
    const cancelBtn = document.getElementById('cancelBtn');
    const newImageBtn = document.getElementById('newImageBtn');
    const retryBtn = document.getElementById('retryBtn');
    const autoProcessCheckbox = document.getElementById('autoProcess');
    const featureCards = document.querySelectorAll('.feature-card');

    // --- State Management ---
    let currentFile = null;
    let isProcessing = false; // <-- *** FIX: ADD THIS STATE VARIABLE ***

    // --- Event Listeners ---

    // Open file dialog when browse button or upload area is clicked
    browseBtn.addEventListener('click', () => fileInput.click());
    uploadArea.addEventListener('click', (e) => {
        if (e.target.id === 'browseBtn' || e.target.parentElement.id === 'browseBtn') return;
        fileInput.click();
    });

    // Handle file drag-and-drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => uploadArea.classList.add('highlight'), false);
    });
    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => uploadArea.classList.remove('highlight'), false);
    });
    uploadArea.addEventListener('drop', handleDrop, false);

    // Handle file selection from input
    fileInput.addEventListener('change', handleFileSelect);

    // Handle pipeline selection
    featureCards.forEach(card => {
        card.addEventListener('click', () => {
            if (isProcessing) return; // Don't allow changes during processing
            featureCards.forEach(c => {
                c.classList.remove('selected');
                c.setAttribute('aria-pressed', 'false');
            });
            card.classList.add('selected');
            card.setAttribute('aria-pressed', 'true');
            document.getElementById(`radio-${card.dataset.pipeline}`).checked = true;
        });
    });
    
    // Handle processing and cancellation buttons
    processBtn.addEventListener('click', () => startProcessing(currentFile));
    cancelBtn.addEventListener('click', resetToInitialState);
    newImageBtn.addEventListener('click', resetToInitialState);
    retryBtn.addEventListener('click', resetToInitialState);

    // --- Core Functions ---
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const file = dt.files[0];
        if (file) {
            handleFile(file);
        }
    }

    function handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            handleFile(file);
        }
    }

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            showToast('Error: Please select an image file (PNG, JPG, JPEG).', 'error');
            return;
        }
        currentFile = file;
        displayPreview(file);
        
        if (autoProcessCheckbox.checked) {
            startProcessing(file);
        }
    }

    function displayPreview(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
        };
        reader.readAsDataURL(file);

        fileName.textContent = file.name;
        fileSize.textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB`;
        uploadArea.style.display = 'none';
        imagePreview.style.display = 'flex';
    }

    function startProcessing(file) {
        // <-- *** FIX: CHECK THE FLAG BEFORE PROCEEDING ***
        if (isProcessing) {
            console.warn("Processing is already in progress. Ignoring new request.");
            return; 
        }
        if (!file) {
            showToast('No file selected for processing.', 'error');
            return;
        }

        // <-- *** FIX: SET THE FLAG AND DISABLE BUTTONS ***
        isProcessing = true;
        processBtn.disabled = true;
        processBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
        
        // Hide upload section, show progress section
        uploadSection.style.display = 'none';
        progressSection.style.display = 'block';
        resultsSection.style.display = 'none';
        errorSection.style.display = 'none';

        // Initialize progress UI (you can connect this to real progress later)
        updateProgress(0, 'Initializing AI models...');

        // Simulate a processing pipeline
        // In a real app, you would use fetch() here and update based on response
        // This is a placeholder for your actual fetch logic
        simulatePipeline(file); 
    }
    
    // This is a placeholder. Replace this with your actual fetch call.
    async function simulatePipeline(file) {
        try {
            // Stage 1: Model Loading
            await updateStageProgress(1, 10, 'Loading CRAFT & TrOCR models...');
            
            // Stage 2: CRAFT Detection
            await updateStageProgress(2, 30, 'Detecting text regions & characters...');
            
            // Stage 3: TrOCR Recognition
            await updateStageProgress(3, 70, 'Transformer-based text extraction...');
            
            // Stage 4: Post-Processing
            await updateStageProgress(4, 90, 'AI-powered error correction...');
            
            // Stage 5: PDF Generation
            await updateStageProgress(5, 100, 'Creating formatted document...');
            showResults({ /* dummy data object */ });

        } catch (error) {
            showError('Simulation failed. Please try again.');
        } finally {
            // <-- *** FIX: RESET THE FLAG IN THE 'finally' BLOCK ***
            isProcessing = false; 
            processBtn.disabled = false;
            processBtn.innerHTML = '<i class="fas fa-cogs"></i> Start AI Processing';
        }
    }

    function updateProgress(percentage, message) {
        document.getElementById('progressFill').style.width = `${percentage}%`;
        document.getElementById('progressValue').textContent = `${percentage}%`;
        document.getElementById('progressMessage').textContent = message;
    }
    
    async function updateStageProgress(stageNum, percentage, message) {
        return new Promise(resolve => {
            setTimeout(() => {
                // Mark previous stages as complete
                for (let i = 1; i < stageNum; i++) {
                    const stage = document.getElementById(`stage${i}`);
                    stage.classList.add('complete');
                    stage.querySelector('.stage-status').innerHTML = '<i class="fas fa-check-circle"></i>';
                }
                
                // Mark current stage as active
                const currentStage = document.getElementById(`stage${stageNum}`);
                if (currentStage) {
                    currentStage.classList.add('active');
                    currentStage.querySelector('.stage-status').innerHTML = '<div class="spinner-small"></div>';
                }
                
                updateProgress(percentage, message);
                resolve();
            }, 1000); // Simulate network/processing delay
        });
    }

    function showResults(data) {
        document.getElementById('detectedRegions').textContent = '152';
        document.getElementById('recognizedLines').textContent = '48';
        document.getElementById('processingTime').textContent = '4.2s';
        document.getElementById('confidenceScore').textContent = '96.3%';
        
        progressSection.style.display = 'none';
        resultsSection.style.display = 'block';
    }

    function showError(message) {
        document.getElementById('errorMessage').textContent = message;
        progressSection.style.display = 'none';
        errorSection.style.display = 'block';
    }
    
    function resetToInitialState() {
        // <-- *** FIX: ENSURE FLAG IS RESET HERE TOO ***
        isProcessing = false;
        
        currentFile = null;
        fileInput.value = ''; // Clear the file input
        
        // Reset UI to the initial upload state
        uploadSection.style.display = 'block';
        imagePreview.style.display = 'none';
        uploadArea.style.display = 'block';
        progressSection.style.display = 'none';
        resultsSection.style.display = 'none';
        errorSection.style.display = 'none';

        // Reset progress bar and stages
        resetProgressUI();
    }
    
    function resetProgressUI() {
        updateProgress(0, 'Initializing AI models...');
        const stages = document.querySelectorAll('.processing-stages .stage');
        stages.forEach(stage => {
            stage.classList.remove('active', 'complete');
            stage.querySelector('.stage-status').innerHTML = '';
        });
    }

    function showToast(message, type = 'info') {
        const toastContainer = document.getElementById('toastContainer');
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.textContent = message;
        toastContainer.appendChild(toast);
        setTimeout(() => {
            toast.classList.add('show');
        }, 100);
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 500);
        }, 3000);
    }
});