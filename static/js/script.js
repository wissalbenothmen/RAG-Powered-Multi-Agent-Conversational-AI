// Configuration de marked pour le rendu Markdown
marked.setOptions({
    breaks: true,
    gfm: true,
    sanitize: false
});

// Function to handle form submission for chat and interact
// Function to handle form submission for chat and interact
function handleFormSubmit(formId, endpoint, messagesContainerId, isInteract = false) {
    const form = document.getElementById(formId);
    if (!form) return;

    form.addEventListener('submit', async function(event) {
        event.preventDefault();

        const input = this.querySelector('input[name="user_input"]') || this.querySelector('input[name="query"]');
        const userInput = input.value;
        const messagesContainer = document.getElementById(messagesContainerId);
        const submitButton = this.querySelector('button');

        // Append new user message
        messagesContainer.innerHTML += `
            <div class="message user-message animate__animated animate__fadeInRight">
                <strong>Vous :</strong> ${userInput}
            </div>
        `;

        // Clear the input field immediately after appending the message
        input.value = '';

        // Show loading indicator
        messagesContainer.innerHTML += `
            <div class="message assistant-message loading-message animate__animated animate__fadeIn" id="loading-message">
                <div class="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        `;

        submitButton.disabled = true;
        submitButton.innerHTML = '<span class="spinner"></span>';

        try {
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(isInteract ? { query: userInput } : { user_input: userInput })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            document.querySelector('.loading-message')?.remove();

            if (data.status === 'success') {
                const formattedAnswer = marked.parse(data.answer);
                
                // Transform in-text source references (e.g., [Source: filename]) into numbered footnotes
                let transformedAnswer = formattedAnswer;
                const sourceMatches = transformedAnswer.match(/\[Source: (.+?)\]/g) || [];
                const sourceMap = new Map();
            
                sourceMatches.forEach((match, index) => {
                    const source = match.replace('[Source: ', '').replace(']', '');
                    sourceMap.set(index + 1, source);
                    transformedAnswer = transformedAnswer.replace(match, `<sup>[${index + 1}]</sup>`);
                });
            
                let messageContent = `
                    <div class="message assistant-message animate__animated animate__fadeInLeft">
                        <strong>Assistant :</strong>
                        <div class="answer markdown-body">${transformedAnswer}</div>
                `;
            
                if (sourceMap.size > 0) {
                    messageContent += `
                        <div class="sources-section animate__animated animate__fadeIn">
                            <h4>Sources <span class="toggle-collapse" onclick="toggleSources(this)">[-]</span></h4>
                            <div class="sources-list" style="display: block;">
                                <ul class="source-footnotes">
                                    ${Array.from(sourceMap.entries()).map(([num, source]) => `
                                        <li><sup>${num}</sup> ${source}</li>
                                    `).join('')}
                                </ul>
                            </div>
                        </div>
                    `;
                }

                if (isInteract) {
                    messageContent += `
                        <div class="pdf-link">
                            <strong>Article URL:</strong> <a href="${data.pdf_url}" target="_blank">${data.pdf_url}</a>
                        </div>
                    `;
                } else {
                    messageContent += `
                        <div class="sources">
                            <h4>Sources :</h4>
                            <div class="sources-grid">
                                ${data.sources.map(source => `
                                    <div class="source-card animate__animated animate__fadeIn">
                                        <div class="source-header">
                                            <i class="fas fa-file-alt"></i>
                                            <span class="source-score">${(source.similarity_score * 100).toFixed(2)}%</span>
                                        </div>
                                        <div class="source-body">
                                            <div class="source-title">${source.source}</div>
                                            <div class="source-focus">${source.focus_area}</div>
                                        </div>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                        <div class="arxiv-references">
                            <h4>ArXiv Research Articles :</h4>
                            <div class="sources-grid">
                                ${data.arxiv_references.map(ref => `
                                    <div class="source-card animate__animated animate__fadeIn">
                                        <div class="source-header">
                                            <i class="fas fa-book"></i>
                                            <span class="source-score">${(ref.similarity_score * 100).toFixed(2)}%</span>
                                        </div>
                                        <div class="source-body">
                                            <div class="source-title">${ref.title}</div>
                                            <div class="source-focus">ArXiv ID: ${ref.id}</div>
                                            <div class="arxiv-actions">
                                                <a href="/download/${ref.id}" class="download-btn" download="${ref.id}.pdf">Download</a>
                                                <a href="/interact/${ref.id}" class="interact-btn" target="_blank">Interact</a>
                                            </div>
                                        </div>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                    `;

                    if (data.answer !== "Hello! How can I assist you today?") {
                        messageContent += `
                            <div class="feedback-form animate__animated animate__fadeIn">
                                <h4>Votre avis compte !</h4>
                                <p class="feedback-subtitle">Aidez-nous à améliorer votre expérience</p>
                                <div class="satisfaction-rating">
                                    <span>Satisfaction :</span>
                                    <div class="rating">
                                        ${[1, 2, 3, 4, 5].map(score => `
                                            <i class="fas fa-star" data-score="${score}" title="${score} étoile${score > 1 ? 's' : ''}"></i>
                                        `).join('')}
                                    </div>
                                </div>
                                <div class="feedback-fields">
                                    <select class="domain-select" required>
                                        <option value="">Choisissez un domaine</option>
                                        <option value="Machine Learning">Machine Learning</option>
                                        <option value="NLP">NLP</option>
                                        <option value="Deep Learning">Deep Learning</option>
                                        <option value="Mathématiques">Mathématiques</option>
                                        <option value="Simulation">Simulation</option>
                                    </select>
                                    <textarea class="feedback-textarea" placeholder="Commentaires (facultatif)" rows="3"></textarea>
                                    <div class="feedback-actions">
                                        <button class="submit-feedback" type="button">
                                            <i class="fas fa-paper-plane"></i> Envoyer
                                        </button>
                                        <button class="skip-feedback" type="button">
                                            <i class="fas fa-times"></i> Ignorer
                                        </button>
                                    </div>
                                </div>
                            </div>
                        `;
                    }
                }

                messageContent += `</div>`;
                messagesContainer.innerHTML += messageContent;

                if (!isInteract && data.answer !== "Hello! How can I assist you today?") {
                    setupFeedback(messagesContainer, userInput, data);
                }
            } else {
                throw new Error(data.message || 'Error processing request');
            }
        } catch (error) {
            console.error('Error:', error);
            document.querySelector('.loading-message')?.remove();
            messagesContainer.innerHTML += `
                <div class="message error-message animate__animated animate__shakeX">
                    Une erreur s'est produite. Veuillez réessayer. Détails : ${error.message}
                </div>
            `;
        } finally {
            submitButton.disabled = false;
            submitButton.innerHTML = 'Envoyer';
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
    });
}

// Function to setup feedback interactions (unchanged)
function setupFeedback(messagesContainer, userInput, data) {
    const lastMessage = messagesContainer.lastElementChild;
    const stars = lastMessage.querySelectorAll('.rating .fa-star');
    const submitFeedback = lastMessage.querySelector('.submit-feedback');
    const skipFeedback = lastMessage.querySelector('.skip-feedback');
    let selectedScore = 0;

    // Star rating hover and click effects
    stars.forEach(star => {
        star.addEventListener('mouseover', function() {
            const score = this.dataset.score;
            stars.forEach(s => {
                s.classList.remove('hover');
                if (s.dataset.score <= score) {
                    s.classList.add('hover');
                }
            });
        });

        star.addEventListener('mouseout', function() {
            stars.forEach(s => s.classList.remove('hover'));
        });

        star.addEventListener('click', function() {
            selectedScore = this.dataset.score;
            stars.forEach(s => {
                s.classList.remove('selected');
                if (s.dataset.score <= selectedScore) {
                    s.classList.add('selected');
                }
            });
        });
    });

    // Submit feedback
    submitFeedback.addEventListener('click', async function() {
        const feedbackContainer = this.closest('.feedback-form');
        const domain = feedbackContainer.querySelector('.domain-select').value;
        const comment = feedbackContainer.querySelector('.feedback-textarea').value;

        if (!selectedScore) {
            feedbackContainer.querySelector('.satisfaction-rating').classList.add('error-shake');
            setTimeout(() => feedbackContainer.querySelector('.satisfaction-rating').classList.remove('error-shake'), 500);
            return;
        }

        if (!domain) {
            feedbackContainer.querySelector('.domain-select').classList.add('error-shake');
            setTimeout(() => feedbackContainer.querySelector('.domain-select').classList.remove('error-shake'), 500);
            return;
        }

        try {
            const feedbackData = {
                question: userInput,
                domain: domain,
                response_type: 'text',
                satisfaction_score: parseInt(selectedScore),
                user_feedback: comment,
                response_time: data.response_time || 0,
                sources_count: data.sources_count || 0,
                accuracy_score: data.sources[0]?.similarity_score || 0,
                article_category: data.sources[0]?.focus_area || 'Non spécifié'
            };

            const response = await fetch('/feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(feedbackData)
            });

            if (!response.ok) throw new Error('Erreur lors de l’envoi');

            feedbackContainer.innerHTML = `
                <div class="feedback-success animate__animated animate__bounceIn">
                    <i class="fas fa-check-circle"></i>
                    <p>Merci pour votre retour !</p>
                </div>`;
            setTimeout(() => feedbackContainer.remove(), 2000); // Auto-remove after 2 seconds
        } catch (error) {
            feedbackContainer.innerHTML += `<p class="feedback-error">Erreur : ${error.message}</p>`;
        }
    });

    // Skip feedback
    skipFeedback.addEventListener('click', function() {
        const feedbackContainer = this.closest('.feedback-form');
        feedbackContainer.classList.add('animate__fadeOut');
        setTimeout(() => feedbackContainer.remove(), 500);
    });
}

// Rest of your script remains unchanged...
function handleQueryForm() {
    const form = document.getElementById('queryForm');
    if (!form) return;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const query = document.getElementById('query').value;
        const imageFile = document.getElementById('image')?.files[0];
        const formData = new FormData();
        formData.append('query', query);
        if (imageFile) {
            formData.append('image', imageFile);
        }

        try {
            document.getElementById('result').classList.add('d-none');
            document.getElementById('error').classList.add('d-none');

            const response = await fetch('/process', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.status === 'success') {
                document.getElementById('answer').textContent = data.answer;
                const sourcesContainer = document.getElementById('sources');
                sourcesContainer.innerHTML = '';
                data.sources.forEach(source => {
                    const sourceElement = document.createElement('div');
                    sourceElement.className = 'list-group-item';
                    sourceElement.innerHTML = `
                        <div>
                            <strong>${source.source}</strong>
                            <p class="mb-1">${source.focus_area}</p>
                        </div>
                        <span class="source-score">
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
        }
    });
}

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

function initializeSidebar() {
    const body = document.querySelector('body'),
          sidebar = body.querySelector('nav'),
          toggle = body.querySelector(".toggle"),
          searchBtn = body.querySelector(".search-box"),
          modeSwitch = body.querySelector(".toggle-switch"),
          modeText = body.querySelector(".mode-text");

    toggle.addEventListener("click", () => {
        sidebar.classList.toggle("close");
    });

    searchBtn.addEventListener("click", () => {
        sidebar.classList.remove("close");
    });

    modeSwitch.addEventListener("click", () => {
        body.classList.toggle("dark");
        if (body.classList.contains("dark")) {
            modeText.innerText = "Light mode";
        } else {
            modeText.innerText = "Dark mode";
        }
    });
}

document.addEventListener('DOMContentLoaded', function() {
    console.log('DOMContentLoaded event fired');

    handleFormSubmit('chat-form', '/chat', 'chat-messages', false);
    const pathParts = window.location.pathname.split('/');
    const arxivId = pathParts[pathParts.length - 1];
    if (pathParts.includes('interact')) {
        console.log('Initializing interact form');
        handleFormSubmit('interact-form', `/interact/${arxivId}`, 'interact-messages', true);
    }
    if (pathParts.includes('dashboard')) {
        initializeDashboard();
    }
    handleQueryForm();
    initializeSidebar();

    const positiveBtn = document.getElementById('positiveFeedback');
    const negativeBtn = document.getElementById('negativeFeedback');
    if (positiveBtn) positiveBtn.addEventListener('click', () => sendFeedback(true));
    if (negativeBtn) negativeBtn.addEventListener('click', () => sendFeedback(false));

    const existingMessages = document.getElementById('interact-messages')?.innerHTML;
    if (existingMessages) {
        document.getElementById('interact-messages').innerHTML = existingMessages;
    }
});