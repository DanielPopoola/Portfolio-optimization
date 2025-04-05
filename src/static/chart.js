// Wait for the document to load
document.addEventListener('DOMContentLoaded', function() {
    // Check if we're on the results page by looking for the results-container
    const resultsContainer = document.getElementById('results-container');
    if (!resultsContainer) return;

    // Add event listeners to portfolio cards for highlighting
    const portfolioCards = document.querySelectorAll('.portfolio-card');
    portfolioCards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.classList.add('portfolio-highlight');
        });

        card.addEventListener('mouseleave', function() {
            this.classList.remove('portfolio-highlight');
        });
    });

    // Add toggle for weight visualization
    const toggleWeightView = document.getElementById('toggle-weight-view');
    if (toggleWeightView) {
        toggleWeightView.addEventListener('click', function() {
            const weightLists = document.querySelectorAll('.weight-list');
            const weightCharts = document.querySelectorAll('.weight-chart');

            weightLists.forEach(list => {
                list.classList.toggle('d-none');
            });

            weightCharts.forEach(chart => {
                chart.classList.toggle('d-none');
            });

            // Update button text
            if (this.textContent.includes('Chart')) {
                this.textContent = 'Show as List';
            } else {
                this.textContent = 'Show as Chart';
            }
        });
    }

    // Function to create simple bar charts for portfolio weights
    function createWeightCharts() {
        const weightChartContainers = document.querySelectorAll('.weight-chart');

        weightChartContainers.forEach(container => {
            const weights = JSON.parse(container.dataset.weights);
            const symbols = Object.keys(weights);
            const values = Object.values(weights);

            // Create bars for each weight
            symbols.forEach((symbol, index) => {
                const bar = document.createElement('div');
                bar.className = 'weight-bar';
                bar.style.width = `${values[index] * 100}%`;
                bar.style.backgroundColor = getRandomColor(index);

                const label = document.createElement('span');
                label.className = 'weight-label';
                label.textContent = `${symbol}: ${(values[index] * 100).toFixed(2)}%`;

                bar.appendChild(label);
                container.appendChild(bar);
            });
        });
    }

    // Generate random colors for bars
    function getRandomColor(index) {
        const colors = [
            '#007bff', '#28a745', '#dc3545', '#ffc107', '#17a2b8',
            '#6610f2', '#fd7e14', '#20c997', '#e83e8c', '#6f42c1'
        ];
        return colors[index % colors.length];
    }

    // Initialize charts if we have the containers
    if (document.querySelectorAll('.weight-chart').length > 0) {
        createWeightCharts();
    }

    // Add copy to clipboard functionality for weights
    const copyButtons = document.querySelectorAll('.copy-weights');
    copyButtons.forEach(button => {
        button.addEventListener('click', function() {
            const weightData = JSON.parse(this.dataset.weights);
            let copyText = "Portfolio Weights:\n";

            for (const [symbol, weight] of Object.entries(weightData)) {
                copyText += `${symbol}: ${(weight * 100).toFixed(2)}%\n`;
            }

            navigator.clipboard.writeText(copyText).then(() => {
                // Show a temporary success message
                const originalText = this.textContent;
                this.textContent = 'Copied!';
                setTimeout(() => {
                    this.textContent = originalText;
                }, 2000);
            });
        });
    });
});