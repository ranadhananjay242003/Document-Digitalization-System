// New Layout Feature Card Interactions

document.addEventListener('DOMContentLoaded', function() {
    // Get all feature cards
    const featureCards = document.querySelectorAll('.feature-card');
    const radioInputs = document.querySelectorAll('input[name="pipeline"]');

    // Handle feature card clicks
    featureCards.forEach(card => {
        card.addEventListener('click', function() {
            const pipeline = this.dataset.pipeline;
            
            // Remove selected class from all cards
            featureCards.forEach(c => {
                c.classList.remove('selected');
                c.setAttribute('aria-pressed', 'false');
            });
            
            // Add selected class to clicked card
            this.classList.add('selected');
            this.setAttribute('aria-pressed', 'true');
            
            // Update radio input
            const radioInput = document.getElementById(`radio-${pipeline}`);
            if (radioInput) {
                radioInput.checked = true;
            }
            
            // Trigger change event for compatibility with existing code
            const changeEvent = new Event('change', { bubbles: true });
            radioInput.dispatchEvent(changeEvent);
            
            // Update estimation time based on pipeline
            updateEstimationTime(pipeline);
            
            console.log('Pipeline selected:', pipeline);
        });
        
        // Add keyboard support
        card.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                this.click();
            }
        });
        
        // Make cards focusable
        card.setAttribute('tabindex', '0');
    });
    
    // Keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
            return; // Don't interfere with form inputs
        }
        
        switch(e.key.toLowerCase()) {
            case 'a':
                e.preventDefault();
                document.querySelector('.feature-card[data-pipeline="advanced"]').click();
                break;
            case 'o':
                e.preventDefault();
                document.querySelector('.feature-card[data-pipeline="optimized"]').click();
                break;
            case 'f':
                e.preventDefault();
                document.querySelector('.feature-card[data-pipeline="basic"]').click();
                break;
        }
    });
    
    // Function to update estimation time
    function updateEstimationTime(pipeline) {
        const estimateTimeElement = document.getElementById('estimateTime');
        if (estimateTimeElement) {
            let timeEstimate = '';
            switch(pipeline) {
                case 'advanced':
                    timeEstimate = '3-7 minutes';
                    break;
                case 'optimized':
                    timeEstimate = '2-4 minutes';
                    break;
                case 'basic':
                    timeEstimate = '1-2 minutes';
                    break;
                default:
                    timeEstimate = '2-5 minutes';
            }
            estimateTimeElement.textContent = timeEstimate;
        }
    }
    
    // Add visual feedback animations
    featureCards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            if (!this.classList.contains('selected')) {
                this.style.transform = 'translateY(-3px) scale(1.02)';
            }
        });
        
        card.addEventListener('mouseleave', function() {
            if (!this.classList.contains('selected')) {
                this.style.transform = '';
            }
        });
    });
    
    // Initialize with default selection
    const defaultCard = document.querySelector('.feature-card.selected');
    if (defaultCard) {
        updateEstimationTime(defaultCard.dataset.pipeline);
    }
});

// Export functions for use by other scripts
window.newLayoutUtils = {
    selectPipeline: function(pipeline) {
        const card = document.querySelector(`.feature-card[data-pipeline="${pipeline}"]`);
        if (card) {
            card.click();
        }
    },
    
    getCurrentPipeline: function() {
        const selectedCard = document.querySelector('.feature-card.selected');
        return selectedCard ? selectedCard.dataset.pipeline : 'advanced';
    }
};
