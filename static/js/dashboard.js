// Stockage des instances de graphiques
let charts = {
    domainSatisfaction: null,
    satisfactionDistribution: null,
    satisfactionTrend: null,
    satisfactionBoxplot: null,
    responseTime: null
};

// Configuration des couleurs
const chartColors = {
    primary: '#2196f3',
    secondary: '#4caf50',
    accent: '#ff9800',
    error: '#f44336',
    success: '#4caf50',
    warning: '#ff9800',
    info: '#2196f3',
    background: 'rgba(33, 150, 243, 0.1)',
    border: 'rgba(33, 150, 243, 1)'
};

// Fonction pour détruire un graphique existant
function destroyChart(chartId) {
    if (charts[chartId]) {
        charts[chartId].destroy();
        charts[chartId] = null;
    }
}

// Liste étendue des mots à ignorer en français et en anglais
const stopWords = new Set([
    // Français
    'le', 'la', 'les', 'de', 'des', 'du', 'un', 'une', 'et', 'est', 'en', 'que', 'qui', 
    'dans', 'sur', 'pour', 'par', 'avec', 'sans', 'ou', 'où', 'donc', 'or', 'ni', 'car',
    'ce', 'cet', 'cette', 'ces', 'mon', 'ton', 'son', 'notre', 'votre', 'leur',
    'plus', 'moins', 'très', 'bien', 'mal', 'peu', 'trop', 'beaucoup', 'aussi',
    // English
    'the', 'of', 'to', 'and', 'a', 'in', 'is', 'it', 'you', 'that', 'was', 'for',
    'on', 'are', 'with', 'as', 'at', 'this', 'but', 'they', 'be', 'from',
    'have', 'has', 'had', 'what', 'when', 'where', 'which', 'who', 'why', 'how',
    'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
    'than', 'too', 'very', 'can', 'will', 'just', 'should', 'now'
]);

// Fonction générique pour créer un nuage de mots
function createWordCloud(containerId, words, maxSize = 50) {
    const width = document.getElementById(containerId).offsetWidth;
    const height = document.getElementById(containerId).offsetHeight || 300;

    const layout = d3.layout.cloud()
        .size([width, height])
        .words(words)
        .padding(5)
        .rotate(() => (~~(Math.random() * 2) * 90))
        .fontSize(d => d.size)
        .on('end', draw);

    function draw(words) {
        d3.select(`#${containerId}`).selectAll('*').remove();
        
        const svg = d3.select(`#${containerId}`)
            .append('svg')
            .attr('width', width)
            .attr('height', height)
            .append('g')
            .attr('transform', `translate(${width/2},${height/2})`);

        svg.selectAll('text')
            .data(words)
            .enter()
            .append('text')
            .style('font-size', d => `${d.size}px`)
            .style('fill', () => `hsl(${Math.random() * 360}, 70%, 50%)`)
            .attr('text-anchor', 'middle')
            .attr('transform', d => `translate(${d.x},${d.y})rotate(${d.rotate})`)
            .text(d => d.text)
            .append('title')  // Ajouter un tooltip
            .text(d => `Frequency: ${d.freq}`);
    }

    layout.start();
}

// Fonction pour traiter le texte et obtenir les fréquences des mots
function getWordFrequencies(texts) {
    const wordFreq = new Map();
    
    texts.forEach(text => {
        if (!text) return;
        const words = text.toLowerCase()
            .replace(/[.,\/#!$%\^&\*;:{}=\-_`~()]/g, ' ')
            .split(/\s+/);
        
        words.forEach(word => {
            if (word.length > 2 && !stopWords.has(word)) {
                wordFreq.set(word, (wordFreq.get(word) || 0) + 1);
            }
        });
    });
    
    return Array.from(wordFreq.entries())
        .sort((a, b) => b[1] - a[1])
        .map(([text, freq]) => ({
            text,
            size: 10 + Math.min(freq * 15, 50),
            freq
        }));
}

// Fonction pour générer les deux nuages de mots
function generateWordClouds(feedbacks) {
    // Nuage de mots des commentaires
    const feedbackWords = getWordFrequencies(
        feedbacks
            .filter(f => f.user_feedback)
            .map(f => f.user_feedback)
    );
    
    // Nuage de mots des questions
    const questionWords = getWordFrequencies(
        feedbacks
            .filter(f => f.question)
            .map(f => f.question)
    );

    console.log('Feedback words:', feedbackWords);
    console.log('Question words:', questionWords);

    if (feedbackWords.length > 0) {
        createWordCloud('feedback-cloud-container', feedbackWords);
    } else {
        document.getElementById('feedback-cloud-container').innerHTML = 
            '<div class="no-data">No feedback data available</div>';
    }

    if (questionWords.length > 0) {
        createWordCloud('questions-cloud-container', questionWords);
    } else {
        document.getElementById('questions-cloud-container').innerHTML = 
            '<div class="no-data">No question data available</div>';
    }
}

// Fonction pour mettre à jour le tableau de feedback
function updateFeedbackTable(data) {
    const feedbacks = data.all_feedbacks || [];
    const tableBody = document.querySelector('#feedbackTable tbody');
    const domainFilter = document.getElementById('domainFilter');
    const scoreFilter = document.getElementById('scoreFilter');
    
    // Mettre à jour les options de domaine
    const domains = new Set(feedbacks.map(f => f.domain || 'Unspecified'));
    domainFilter.innerHTML = '<option value="">All domains</option>';
    domains.forEach(domain => {
        domainFilter.innerHTML += `<option value="${domain}">${domain}</option>`;
    });
    
    // Fonction pour filtrer les feedbacks
    function filterFeedbacks() {
        const selectedDomain = domainFilter.value;
        const selectedScore = scoreFilter.value;
        
        return feedbacks
            .filter(f => !selectedDomain || f.domain === selectedDomain)
            .filter(f => !selectedScore || f.satisfaction_score === parseInt(selectedScore))
            .sort((a, b) => a.satisfaction_score - b.satisfaction_score); // Trier par score croissant
    }
    
    // Fonction pour mettre à jour l'affichage
    function updateDisplay() {
        const filteredFeedbacks = filterFeedbacks();
        tableBody.innerHTML = filteredFeedbacks
            .map(f => `
                <tr>
                    <td>${'⭐'.repeat(f.satisfaction_score)}</td>
                    <td>${f.domain || 'Unspecified'}</td>
                    <td>${f.question}</td>
                    <td>${f.user_feedback || '-'}</td>
                    <td>${moment(f.timestamp).format('DD/MM/YY HH:mm')}</td>
                </tr>
            `)
            .join('');
    }
    
    // Gestionnaires d'événements pour les filtres
    domainFilter.addEventListener('change', updateDisplay);
    scoreFilter.addEventListener('change', updateDisplay);
    
    // Gestionnaire pour l'exportation
    document.getElementById('exportFeedback').addEventListener('click', () => {
        const filteredFeedbacks = filterFeedbacks();
        const csv = [
            ['Score', 'Domain', 'Question', 'Comment', 'Date'],
            ...filteredFeedbacks.map(f => [
                f.satisfaction_score,
                f.domain || 'Unspecified',
                f.question,
                f.user_feedback || '-',
                moment(f.timestamp).format('DD/MM/YY HH:mm')
            ])
        ]
        .map(row => row.map(cell => `"${cell}"`).join(','))
        .join('\n');
        
        const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = 'feedbacks.csv';
        link.click();
    });
    
    // Affichage initial
    updateDisplay();
}

// Modification de la fonction updateDashboard
function updateDashboard(data) {
    console.log('Received data:', data);
    
    if (!data.all_feedbacks || data.all_feedbacks.length === 0) {
        console.error('No feedback data received');
        return;
    }

    // Calcul des statistiques globales
    const feedbacks = data.all_feedbacks;
    const totalFeedback = feedbacks.length;
    
    // Calcul de la satisfaction moyenne
    const validSatisfactionScores = feedbacks.filter(f => typeof f.satisfaction_score === 'number');
    const avgSatisfaction = validSatisfactionScores.length > 0 
        ? validSatisfactionScores.reduce((sum, f) => sum + f.satisfaction_score, 0) / validSatisfactionScores.length 
        : 0;

    // Calcul du temps de réponse moyen
    const validResponseTimes = feedbacks.filter(f => typeof f.response_time === 'number');
    const avgResponseTime = validResponseTimes.length > 0
        ? validResponseTimes.reduce((sum, f) => sum + f.response_time, 0) / validResponseTimes.length
        : 0;

    // Mise à jour des métriques
    document.getElementById('total-feedback').textContent = totalFeedback;
    document.getElementById('avg-satisfaction').textContent = avgSatisfaction.toFixed(2) + '/5';
    document.getElementById('avg-response-time').textContent = avgResponseTime.toFixed(2) + 's';
    document.getElementById('update-time').textContent = new Date().toLocaleString();

    // Mise à jour des graphiques
    generateWordClouds(feedbacks);
    updateDomainSatisfactionChart(data);
    updateSatisfactionDistributionChart(data);
    updateSatisfactionTrendChart(data);
    updateFeedbackTable(data);
}

function updateDomainSatisfactionChart(data) {
    destroyChart('domainSatisfaction');
    const ctx = document.getElementById('domain-satisfaction-chart').getContext('2d');
    
    // Calcul de la satisfaction moyenne par domaine
    const domainStats = {};
    const feedbacks = data.all_feedbacks || [];
    
    feedbacks.forEach(feedback => {
        const domain = feedback.domain || 'Unspecified';
        if (!domainStats[domain]) {
            domainStats[domain] = { total: 0, count: 0 };
        }
        domainStats[domain].total += feedback.satisfaction_score;
        domainStats[domain].count++;
    });

    const domains = Object.keys(domainStats);
    const averages = domains.map(domain => 
        domainStats[domain].total / domainStats[domain].count);

    charts.domainSatisfaction = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: domains,
            datasets: [{
                label: 'Average Satisfaction',
                data: averages,
                backgroundColor: chartColors.primary,
                borderColor: chartColors.border,
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 5
                }
            }
        }
    });
}

function updateSatisfactionDistributionChart(data) {
    destroyChart('satisfactionDistribution');
    const ctx = document.getElementById('satisfaction-distribution-chart').getContext('2d');
    
    // Calcul de la distribution des scores
    const distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0};
    const feedbacks = data.all_feedbacks || [];
    
    feedbacks.forEach(feedback => {
        if (typeof feedback.satisfaction_score === 'number') {
            distribution[feedback.satisfaction_score] = 
                (distribution[feedback.satisfaction_score] || 0) + 1;
        }
    });

    // Calculer les pourcentages
    const total = Object.values(distribution).reduce((a, b) => a + b, 0);
    const percentages = Object.values(distribution).map(value => 
        ((value / total) * 100).toFixed(1) + '%'
    );

    charts.satisfactionDistribution = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['1 ⭐', '2 ⭐', '3 ⭐', '4 ⭐', '5 ⭐'],
            datasets: [{
                data: Object.values(distribution),
                backgroundColor: [
                    '#FF6B6B',  // Red for 1 star
                    '#FFA06B',  // Orange for 2 stars
                    '#FFD93D',  // Yellow for 3 stars
                    '#6BCB77',  // Light green for 4 stars
                    '#4D96FF'   // Blue for 5 stars
                ],
                borderColor: 'white',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        generateLabels: function(chart) {
                            const data = chart.data;
                            return data.labels.map((label, i) => ({
                                text: `${label} (${percentages[i]})`,
                                fillStyle: data.datasets[0].backgroundColor[i],
                                index: i
                            }));
                        },
                        font: {
                            size: 12
                        },
                        padding: 10
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const value = context.raw;
                            const percentage = ((value / total) * 100).toFixed(1);
                            return `${value} feedback(s) (${percentage}%)`;
                        }
                    }
                }
            },
            cutout: '60%'  // Ajustement de la taille du trou central
        }
    });
}

function updateSatisfactionTrendChart(data) {
    destroyChart('satisfactionTrend');
    const ctx = document.getElementById('satisfaction-trend-chart').getContext('2d');
    
    // Trier les feedbacks par date
    const feedbacks = (data.all_feedbacks || [])
        .sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));

    charts.satisfactionTrend = new Chart(ctx, {
        type: 'line',
        data: {
            labels: feedbacks.map(f => moment(f.timestamp).format('DD/MM HH:mm')),
            datasets: [{
                label: 'Satisfaction',
                data: feedbacks.map(f => f.satisfaction_score),
                borderColor: chartColors.primary,
                backgroundColor: chartColors.background,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 5
                }
            }
        }
    });
}

function updateSatisfactionBoxplot(data) {
    destroyChart('satisfactionBoxplot');
    const ctx = document.getElementById('satisfaction-boxplot').getContext('2d');
    
    const feedbacks = data.all_feedbacks || [];
    const domainScores = {};
    
    // Regrouper les scores par domaine
    feedbacks.forEach(feedback => {
        const domain = feedback.domain || 'Unspecified';
        const score = feedback.satisfaction_score;
        if (!domainScores[domain]) {
            domainScores[domain] = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0};
        }
        domainScores[domain][score]++;
    });

    const domains = Object.keys(domainScores);
    const scores = [1, 2, 3, 4, 5];
    
    const datasets = scores.map(score => ({
        label: score + ' ⭐',
        data: domains.map(domain => domainScores[domain][score]),
        backgroundColor: [
            '#FF6B6B',  // Red for 1 star
            '#FFA06B',  // Orange for 2 stars
            '#FFD93D',  // Yellow for 3 stars
            '#6BCB77',  // Light green for 4 stars
            '#4D96FF'   // Blue for 5 stars
        ][score - 1]
    }));

    charts.satisfactionBoxplot = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: domains,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Domain'
                    }
                },
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Number of feedbacks'
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Score Distribution by Domain'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: ${context.parsed.y} feedback(s)`;
                        }
                    }
                }
            }
        }
    });
}

function updateResponseTimeChart(data) {
    destroyChart('responseTime');
    const ctx = document.getElementById('response-time-chart').getContext('2d');
    
    const feedbacks = data.all_feedbacks || [];
    const typeStats = {};
    
    // Calculer les temps de réponse moyens par type
    feedbacks.forEach(feedback => {
        if (typeof feedback.response_time !== 'number') return;
        
        const type = feedback.response_type || 'Unspecified';
        if (!typeStats[type]) {
            typeStats[type] = { total: 0, count: 0 };
        }
        typeStats[type].total += feedback.response_time;
        typeStats[type].count++;
    });

    const types = Object.keys(typeStats);
    const avgTimes = types.map(type => 
        parseFloat((typeStats[type].total / typeStats[type].count).toFixed(2))
    );

    charts.responseTime = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: types,
            datasets: [{
                label: 'Response Time (s)',
                data: avgTimes,
                backgroundColor: chartColors.secondary,
                borderColor: chartColors.border,
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Time (seconds)'
                    }
                }
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const value = context.parsed.y;
                            return `${value.toFixed(2)} seconds`;
                        }
                    }
                }
            }
        }
    });
}

// Initialisation des événements
document.addEventListener('DOMContentLoaded', function() {
    const eventSource = new EventSource('/dashboard/stream');
    eventSource.onmessage = function(event) {
        const data = JSON.parse(event.data);
        updateDashboard(data);
    };
});