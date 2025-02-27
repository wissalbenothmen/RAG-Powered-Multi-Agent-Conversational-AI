document.getElementById('queryForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const query = document.getElementById('query').value;
    const imageFile = document.getElementById('image').files[0];
    
    const formData = new FormData();
    formData.append('query', query);
    if (imageFile) {
        formData.append('image', imageFile);
    }
    
    try {
        document.getElementById('result').classList.add('d-none');
        document.getElementById('error').classList.add('d-none');
        
        // Enhanced loading feedback
        const loadingIndicator = document.createElement('div');
        loadingIndicator.className = 'loading-overlay';
        loadingIndicator.innerHTML = '<div class="spinner-modern"></div>';
        document.body.appendChild(loadingIndicator);
        
        const response = await fetch('/process', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        document.body.removeChild(loadingIndicator);
        
        if (data.status === 'success') {
            document.getElementById('answer').textContent = data.answer;
            
            const sourcesContainer = document.getElementById('sources');
            sourcesContainer.innerHTML = '';
            
            data.sources.forEach(source => {
                const sourceElement = document.createElement('div');
                sourceElement.className = 'list-group-item modern-source-card animate__animated animate__fadeIn';
                sourceElement.innerHTML = `
                    <div>
                        <strong>${source.source}</strong>
                        <p class="mb-1">${source.focus_area}</p>
                    </div>
                    <span class="source-score modern-score">
                        Score: ${source.similarity_score.toFixed(2)}
                    </span>
                `;
                sourcesContainer.appendChild(sourceElement);
            });
            
            document.getElementById('result').classList.remove('d-none');
        } else {
            throw new Error(data.message || 'Une erreur est survenue');
        }
        
    } catch (error) {
        document.getElementById('error').textContent = error.message;
        document.getElementById('error').classList.remove('d-none');
        if (document.querySelector('.loading-overlay')) {
            document.body.removeChild(document.querySelector('.loading-overlay'));
        }
    }
});

async function sendFeedback(isPositive) {
    try {
        const response = await fetch('/feedback', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: document.getElementById('query').value,
                rating: isPositive ? 5 : 1,
                timestamp: new Date().toISOString()
            })
        });
        
        const data = await response.json();
        if (data.status === 'success') {
            alert(isPositive ? 'Merci pour votre feedback positif!' : 'Merci pour votre feedback!');
        }
    } catch (error) {
        console.error('Erreur lors de l\'envoi du feedback:', error);
    }
}