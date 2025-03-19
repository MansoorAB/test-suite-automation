// Add copy functionality
function addCopyButtons() {
    document.querySelectorAll('.code-area').forEach(block => {
        const button = document.createElement('button');
        button.className = 'copy-btn';
        button.textContent = 'Copy';
        
        button.addEventListener('click', () => {
            navigator.clipboard.writeText(block.textContent);
            button.textContent = 'Copied!';
            setTimeout(() => {
                button.textContent = 'Copy';
            }, 2000);
        });
        
        block.style.position = 'relative';
        block.appendChild(button);
    });
}

// Scenario search functionality
document.getElementById('search-btn').addEventListener('click', async () => {
    const scenario = document.getElementById('scenario-input').value;
    const response = await fetch('/search_scenario', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({scenario: scenario})
    });
    const matches = await response.json();
    const matchesDiv = document.getElementById('matches');
    matchesDiv.innerHTML = matches.map(match => `
        <div class="scenario-card">
            <div class="scenario-header">
                <div>
                    <div class="scenario-tags">${match.tags || 'No tags'}</div>
                    <small class="text-muted">${match.file}</small>
                </div>
                <div class="similarity-score">${(match.similarity * 100).toFixed(1)}% Match</div>
            </div>
            <div class="scenario-content">${match.scenario}</div>
        </div>
    `).join('');
    addCopyButtons();
});

// BDD generation functionality
document.getElementById('generate-btn').addEventListener('click', async () => {
    const criteria = document.getElementById('criteria-input').value;
    const feature_name = document.getElementById('feature-name').value;
    const response = await fetch('/generate_bdd', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            criteria: criteria,
            feature_name: feature_name
        })
    });
    const result = await response.json();
    document.getElementById('feature-output').textContent = result.feature_file
        .replace('```gherkin\n', '')
        .replace('```\n', '')
        .trim();
    document.getElementById('steps-output').textContent = result.step_definitions
        .replace('```java\n', '')
        .replace('```\n', '')
        .trim();
    addCopyButtons();
});

// Get LLM info on page load
document.addEventListener('DOMContentLoaded', async () => {
    // Initialize copy buttons
    addCopyButtons();
    
    // Get and display LLM info
    try {
        const response = await fetch('/llm_info');
        const info = await response.json();
        document.getElementById('model-badge').textContent = info.model + ' (' + info.version + ')';
    } catch (error) {
        console.error('Error fetching LLM info:', error);
    }
}); 