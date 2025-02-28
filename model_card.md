---

# Model Card: RAG Multi-Agent Conversational AI System

## Model Details
- **Model Name**: RAG Multi-Agent Conversational AI System
- **Version**: 1.0
- **Date**: February 27, 2025
- **Author**: Wissal Ben Othmen, Master’s student in AI and Data Science, Paris Dauphine University - Campus Tunis
- **Contact**: [wissal.benothmen01@gmail.com](mailto:wissal.benothmen01@gmail.com)
- **Superviseur**: M. Florian Bastin
- **Description**: This system is a Retrieval-Augmented Generation (RAG) conversational AI designed to deliver precise, source-specific answers from a corpus of research articles and arXiv papers. It integrates Google’s Gemini 1.5 Pro for generation, FAISS with `all-MiniLM-L6-v2` embeddings for retrieval, and a Flask-based web interface with PostgreSQL on Google Cloud SQL for data persistence. The multi-agent architecture includes specialized components for input processing, retrieval, prompt engineering, answer generation, source tracking, and feedback collection.
- **Architecture**:
  - **Retrieval Component**:
    - **Vector Database**: FAISS (Facebook AI Similarity Search), optimized for cosine similarity search.
    - **Embeddings**: `all-MiniLM-L6-v2` from HuggingFace (384 dimensions), chosen for its lightweight efficiency on CPU.
    - **Storage**: Documents in `gs://rag-multiagent-documents/`, FAISS index in `gs://rag-multiagent-index/`.
  - **Generation Component**:
    - **Model**: Google Gemini 1.5 Pro, temperature 0.7, top-p 0.95, top-k 40, balancing coherence and creativity.
    - **Prompting**: Dynamic templates (default, detailed, creative) selected via `PromptEngineer`.
  - **Web Search**: arXiv API integration, retrieving up to 3 papers per query.
  - **Database**: PostgreSQL on Google Cloud SQL (IP: 34.38.195.191), storing chat sessions (`chat_sessions` table), feedback (`feedback` table), and document metadata (`documents` table).
  - **Framework**: Flask with asynchronous session saving for scalability.
- **License**: For educational purposes

## Intended Use
- **Primary Use Cases**:
  - Deliver accurate, citation-backed answers to researchers by extracting specific excerpts from a corpus of RAG-related articles and arXiv papers (e.g., "This mechanism is from [Source: Retrieval-AugmentedGenerationRAG-AdvancingAIwithDynamicKnowledgeIntegration.pdf]").
  - Enable enterprise research teams to index internal documents (e.g., technical reports) and augment them with external academic sources for enhanced insights.
  - Facilitate academic exploration of AI concepts like RAG and CAG with detailed, source-traced responses.
- **Target Users**:
  - Researchers in AI and Data Science seeking precise information from scholarly articles.
  - Enterprise R&D teams aiming to combine internal knowledge with external research.
  - Students and educators studying conversational AI and retrieval-augmented systems.
- **Benefits**:
  - Accelerates literature reviews by pinpointing exact sources and relevant text segments.
  - Enhances enterprise innovation by bridging internal data with cutting-edge research.
  - Provides transparency through explicit source attribution, fostering trust in responses.
- **Out-of-Scope Use Cases**:
  - General-purpose chatting beyond the corpus domain.
  - High-stakes decision-making without human verification.
  - Non-English queries or document processing.

## Factors
- **Relevant Factors**:
  - **Query Specificity**: Detailed queries (e.g., "How does RAG handle retrieval latency?") outperform vague ones (e.g., "Tell me about AI").
  - **Corpus Relevance**: Answers depend on the document set’s coverage of the query topic.
  - **External API Response Times**: The response time of the system is dependent on the response times of external APIs, such as the Google Gemini API (for generation) and the arXiv API (for web search augmentation). Delays in these services directly impact system latency.
- **Evaluation Factors**:
  - **Top-k Retrieval**: Set to 20 documents, influencing recall and precision.
  - **Chunking**: 1000-character chunks with 200-character overlap, impacting retrieval granularity and context retention.
  - **Prompt Selection**: Keyword-based choice (e.g., "explain" → detailed) affects response depth.

## Metrics
Evaluated on February 28, 2025, using `eval.py` with 10 question-answer pairs from `eval_questions_answers.csv`. Results saved to `eval_results_20250228_003934.json`.

| **Metric**            | **Definition**                              | **Average Score** | **Justification**                                                                 |
|-----------------------|---------------------------------------------|-------------------|-----------------------------------------------------------------------------------|
| **Latency**           | Time from query to response (seconds)       | 9.04s            | Measures user experience; high latency reflects external API delays.              |
| **Semantic Similarity** | Cosine similarity to ground truth (0-1)   | 0.71             | Assesses answer relevance; cosine is standard for embedding comparison.           |
| **Faithfulness**      | Similarity to cited sources (0-1)           | 0.66             | Ensures grounding in retrieved content; critical for RAG integrity.               |
| **Precision**         | Proportion of relevant retrieved chunks     | 0.81             | Evaluates retrieval accuracy; reduces irrelevant content in responses.            |
| **Recall**            | Proportion of ground truth covered          | 0.97             | Measures completeness; high recall ensures comprehensive coverage.                |
| **F1-Score**          | Harmonic mean of precision and recall       | 0.84             | Balances precision and recall; reflects overall retrieval effectiveness.          |
| **NDCG**              | Normalized ranking quality (0-1)            | 0.67             | Prioritizes top results; important for researcher focus on primary sources.       |


- **Methodology**: Queries sent to `/chat`, compared to ground truth using `all-MiniLM-L6-v2` embeddings. F1-Score added to harmonize precision and recall.
- **Analysis**:
  - **Strengths**: High recall (0.97) and F1-Score (0.84) indicate excellent coverage and retrieval balance.
  - **Weaknesses**: Latency (9.04s) is high due to external API response times; faithfulness (0.66) and NDCG (0.67) suggest room for improving source alignment and ranking.

## Evaluation Data
- **Document Corpus**:
  - **Description**: 32 PDF research papers on RAG, CAG, and AI, stored in `gs://rag-multiagent-documents/`.
  - **Source**: Collected from arXiv and public academic repositories, focusing on AI advancements.
  - **Preprocessing**:
    - Extracted text with PyMuPDF.
    - Split into 1000-character chunks with 200-character overlap using `RecursiveCharacterTextSplitter` (separators: `\n\n`, `\n`, `. `, ` `, “”).
    - Embedded with `all-MiniLM-L6-v2` for FAISS indexing.
  - **Motivation**: Ensures a domain-specific knowledge base for researchers and enterprises.
- **Evaluation Dataset**:
  - **Description**: 10 question-answer pairs in `eval_questions_answers.csv` (e.g., "What is RAG’s retrieval mechanism?", "Compare RAG and CAG"). The questions were manually derived from corpus content, focusing on key concepts, definitions, comparisons (e.g., RAG vs. CAG), and explanatory tasks. Efforts were made to ensure the questions covered a range of difficulty levels and represented the types of queries expected from researchers and enterprise users.
  - **Creation**: Manually crafted from corpus content to test factual extraction, synthesis, and explanation capabilities.
  - **Motivation**: Validates system utility for its target audience by simulating realistic use cases.

## Quantitative Analyses
- **Results**:
  - **Summary**:
    - Avg Semantic Similarity: 0.71
    - Avg Precision: 0.81
    - Avg Recall: 0.97
    - Avg F1-Score: 0.84
    - Avg Faithfulness: 0.66
    - Avg NDCG: 0.67
    - Avg Latency: 9.04 seconds
    - Total Examples: 10 (all valid)
  - **Interpretation**:
    - **High Recall (0.97)**: Nearly all relevant information is retrieved, ideal for comprehensive researcher needs.
    - **Solid F1-Score (0.84)**: Balances precision and recall, indicating effective retrieval.
    - **Moderate Semantic Similarity (0.71) and Faithfulness (0.66)**: Suggests answers are reasonably aligned with ground truth and sources, though generation could better reflect retrieved content.
    - **Lower NDCG (0.67)**: Indicates ranking of retrieved documents could be optimized for relevance.
    - **High Latency (9.04s)**: Reflects dependency on external API response times (Google Gemini API and arXiv API), a significant bottleneck affecting real-time use.
- **Uncertainties**:
  - Small sample size (10 examples) limits statistical confidence.
  - Ground truth subjectivity may skew similarity scores.
- **Limitations**:
  - **External API Response Times**: The system’s latency is heavily influenced by the response times of the Google Gemini API and arXiv API, which are beyond local control and vary with network conditions.
  - Recall-focused retrieval may include less relevant chunks, impacting precision slightly.
  - Faithfulness and NDCG suggest occasional misalignment with sources or suboptimal ranking.

## Ethical Considerations
- **Data Bias**: Corpus focuses on AI research, potentially underrepresenting other fields or perspectives (e.g., industry applications).
- **Misinformation**: Relies on document accuracy; errors or outdated content could mislead users.
- **Overreliance**: Researchers might accept answers without verifying sources, risking flawed conclusions.
- **Transparency**: Mitigated by exact source citations (e.g., "[Source: filename]"), enabling verification.
- **Privacy**: Stores anonymized session data and feedback in PostgreSQL; no sensitive personal data beyond optional email.
- **Equity**: English-only system may exclude non-English-speaking researchers. Future work could explore incorporating multilingual support to enhance accessibility for a broader range of users.

## Caveats and Recommendations
- **Limitations**:
  - Knowledge limited to corpus and arXiv; gaps reduce answer quality.
  - High latency (9.04s) due to external API response times impacts real-time usability.
  - English-only due to embedding model and corpus constraints.
- **Recommendations**:
  - Verify cited sources for critical research or enterprise decisions.
  - Update corpus regularly to reflect current research.
  - Mitigate latency by caching frequent queries or exploring faster APIs.
  - For enterprise use, integrate internal documents and scale with cloud infrastructure (e.g., Kubernetes).
  - As a proof-of-concept, this system demonstrates RAG principles effectively. However, for production use, further development is crucial. This includes enhancing reasoning capabilities, handling more complex queries, and undergoing rigorous testing and validation.