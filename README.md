# RAG Multi-Agent System: An Advanced Conversational AI

This project implements a sophisticated conversational AI system leveraging Retrieval-Augmented Generation (RAG) and a multi-agent architecture.  It provides a web interface for users to interact with the AI, ask questions, and receive accurate, context-aware answers supported by verifiable sources.  The system integrates Google's Gemini 1.5 Pro language model, a PostgreSQL database for chat history and feedback, and a FAISS vector database for document retrieval. The project also includes robust evaluation capabilities and a comprehensive dashboard for monitoring performance and user feedback.
## Key Features
* ✅ Retrieval-Augmented Generation (RAG): Combines real-time document retrieval with LLM generation for factual accuracy.
* ✅ Multi-Agent Architecture: Modular agents handle input processing, retrieval, prompt engineering, answer generation, source tracking, and feedback collection.
* ✅ Google Gemini 1.5 Pro: Powers natural language understanding and generation.
* ✅ PostgreSQL on Cloud SQL: Stores chat sessions and feedback with JSONB support for flexibility.
* ✅ FAISS Vector Database: Enables efficient similarity search using all-MiniLM-L6-v2 embeddings.
* ✅ Google Cloud Storage: Manages document corpus and FAISS index in rag-multiagent-documents and rag-multiagent-index buckets.
* ✅ arXiv Integration: Augments responses with downloadable research papers.
* ✅ Web Search Fallback: Supplements answers when local context is insufficient.
* ✅ Interactive Document Exploration: Allows querying specific arXiv papers via /interact/<arxiv_id>.
* ✅ User Feedback Collection: Captures satisfaction scores, domains, and comments.
* ✅ Comprehensive Dashboard: Displays real-time metrics and visualizations (e.g., word clouds, satisfaction trends).
* ✅ Chat History: Enables viewing, renaming, and deleting past sessions.
* ✅ "Explain More" Functionality: Expands previous answers on request.
* ✅ Document Processing: Manual triggering via /process-docs for index updates.
* ✅ Robust Error Handling: Ensures stability with logging and fallbacks.
* ✅ Automated Evaluation: Assesses performance with eval.py using multiple metrics.
## Project Overview

The RAG Multi-Agent System is designed to provide a robust and user-friendly conversational AI experience. It goes beyond simple question answering by incorporating the following key features:

*   **Retrieval-Augmented Generation (RAG):**  Combines the strengths of information retrieval and large language models (LLMs).  Instead of relying solely on the LLM's pre-trained knowledge, RAG retrieves relevant documents from a vector database in real-time to inform the generation process.  This significantly enhances the accuracy and factual correctness of the responses, especially for domain-specific or rapidly evolving information.


*   **Multi-Agent Architecture:**  The system employs a modular, agent-based design.  Different agents handle specific tasks, such as input processing, document retrieval, prompt engineering, answer generation, source tracking, and feedback collection.  This promotes code clarity, maintainability, and extensibility.

*   **Google Gemini 1.5 Pro:**  Utilizes Google's powerful Gemini 1.5 Pro model as the core LLM. Gemini provides excellent language understanding and generation capabilities, making it well-suited for this task.

*   **PostgreSQL Database:**  Stores chat session history and user feedback in a PostgreSQL database.  This enables persistent chat sessions, retrieval of past conversations, and comprehensive feedback analysis.

*   **FAISS Vector Database:**  Uses FAISS (Facebook AI Similarity Search) for efficient similarity search within the document corpus.  PDF documents are processed, chunked, and converted into vector embeddings using the `all-MiniLM-L6-v2` Hugging Face model.  The FAISS index is stored in a Google Cloud Storage (GCS) bucket for persistence and scalability.

*   **Google Cloud Storage (GCS):**  Leverages GCS for storing both the PDF documents and the FAISS index.  This provides a scalable, reliable, and cost-effective storage solution.

*   **arXiv Integration:**  Includes the ability to search arXiv for relevant research papers.  This augments the system's knowledge base with up-to-date scientific literature.  Users can download PDFs directly from arXiv and even interact with them through a dedicated "interact" feature.

*   **Web Search:**  If the retrieved documents do not provide sufficient information to answer a query, the system can optionally perform a (mock) web search to supplement its response. This feature significantly improves the system's ability to handle a wide range of questions.

*   **Interactive Document Exploration ("interact" feature):** Allows users to interact with specific arXiv papers. They can either view a summary of the paper or ask specific questions about its content, with the system leveraging the Gemini model to provide answers based on the PDF's content.

*   **User Feedback Collection:**  Provides a built-in mechanism for users to provide feedback on the quality of the answers, including a satisfaction rating (1-5 stars), domain selection, and optional comments. This feedback is stored in the PostgreSQL database and is crucial for continuous improvement.

*   **Comprehensive Dashboard:** Offers a real-time dashboard displaying key metrics and visualizations, such as:
    *   Total feedback received
    *   Average satisfaction score
    *   Average response time
    *   Word clouds of user comments and questions
    *   Satisfaction scores by domain
    *   Distribution of satisfaction scores
    *   Satisfaction trend over time
    *   Table of individual feedback entries (with filtering and export)

* **Chat History:** Users can view, rename, and delete their previous chat sessions.

*   **"Explain More" Functionality:** Users can request a more detailed explanation of a previous answer. The system uses the previous question, answer, and the "explain more" request to generate an expanded response.

* **Document Processing Management:** Ability to trigger document processing manually, ensuring that the index can be updated with new documents.

*   **Robust Error Handling:**  Includes comprehensive error handling throughout the application to ensure stability and provide informative messages to the user.  Exceptions are logged, and appropriate fallback mechanisms are used to prevent crashes.

*   **Automated Evaluation:** The `eval.py` script provides a comprehensive evaluation framework for assessing the performance of the RAG system.  It uses a CSV file containing question-answer pairs to measure various metrics, including latency, semantic similarity, faithfulness, precision, recall, and NDCG.

## System Architecture and Pipeline

The system follows a well-defined pipeline, from user query to generated response:

1.  **User Input:** The user enters a questionthrough the web interface `chat.html`.

2.  **Input Processing (input_processor.py):** The `input_processor` module handles the initial processing of the user's input.  If an image is provided, it's processed along with the text using the Gemini model.

3.  **Retrieval (retriever.py):**
    *   The `retrieve_information` function in `retriever.py` uses the `VectorDB` instance to search the FAISS index for relevant document chunks.
    *   The `VectorDB` class (defined in `vectordb.py`) manages the vector database, handling document indexing and retrieval. It uses the `all-MiniLM-L6-v2` Hugging Face model to create embeddings for the documents and queries.
    *   The retriever returns a list of relevant document chunks, along with their similarity scores.

4.  **Source Tracking (source_tracker.py):**
    *   The `track_sources` function in `source_tracker.py` deduplicates the retrieved sources, keeping only the most relevant chunk from each document.
    *   It also assigns a "focus area" to each source, derived from the filename (e.g., "Retrieval-Augmented Generation," "Cache-Augmented Generation," "Enterprise Applications").

5.  **Prompt Engineering (prompt_engineer.py):**
    *   The `PromptEngineer` class (defined in `prompt_engineer.py`) selects and crafts an appropriate prompt template based on the query and the retrieved sources.
    *   It supports three prompt styles: "default," "detailed," and "creative," chosen based on keywords in the query and the availability of sources.

6.  **Answer Generation (answer_generator.py):**
    *   The `generate_answer` function in `answer_generator.py` takes the user's query, the retrieved sources, and the crafted prompt as input.
    *   It uses the LangChain framework to create a chain that combines the prompt and the Gemini 1.5 Pro model.
    *   The chain invokes the LLM to generate an answer based on the provided context.
    *   If the response is insufficient and web search is enabled, it performs a mock web search and incorporates the results into the answer.

7. **ArXiv Search (app.py):**
    * The `search_arxiv` function searches ArXiv for relevant publications and adds them to the response.

8.  **Response Presentation:** The generated answer, along with the sources (and arXiv references), is displayed to the user in the web interface.

9.  **Feedback Collection (feedback_collector.py, app.py):**
    *   After receiving an answer, the user is presented with a feedback form.
    *   The `FeedbackCollector` class (defined in `feedback_collector.py`) manages the collection, storage, and retrieval of user feedback.
    *   Feedback data is stored in the PostgreSQL database.

10. **Chat History Management (app.py):**
    *   Chat sessions are saved asynchronously to the PostgreSQL database.
    *   Users can view, rename, and delete their chat history through the `/history` route.

11. **Dashboard (app.py, dashboard.js):**
    *   The `/dashboard` route renders a dashboard that displays various statistics and visualizations based on the collected feedback.
    *   The dashboard uses Chart.js, D3.js, and D3-cloud for visualizations.
    *   Real-time updates are provided via a server-sent events stream (`/dashboard/stream`).

## Detailed Component Breakdown

### Agents

*   **`answer_generator.py`:**  Generates answers using the Gemini model, retrieved documents, and optional web search results.  Uses LangChain for prompt chaining.
*   **`feedback_collector.py`:**  Manages user feedback.  Collects, stores, retrieves, and computes statistics on feedback data.  Uses PostgreSQL for data persistence.
*   **`input_processor.py`:**  Processes user input using the Gemini model.
*   **`prompt_engineer.py`:**  Dynamically selects and crafts prompt templates based on query analysis.
*   **`retriever.py`:**  Retrieves relevant documents from the vector database using LangChain and Gemini.
*   **`source_tracker.py`:**  Processes and deduplicates retrieved document sources, assigning focus areas.

### RAG

*   **`process_docs.py`:**  Processes PDF documents in a GCS bucket, creates or updates the FAISS index, and tests document retrieval.
*   **`vectordb.py`:**  Defines the `VectorDB` class, which manages the vector database (FAISS index).  Handles document loading, splitting, embedding, indexing, and searching.  Uses GCS for persistent storage of the index.
*   **`database/db_manager.py`:** Defines the `DatabaseManager` class, responsible for interacting with the PostgreSQL database to track processed documents.  Stores metadata about processed files, including filename, timestamp, chunk count, and status.

### Flask Application (`app.py`)

*   **Routing:**  Defines routes for the web interface, including:
    *   `/`:  Main index page.
    *   `/chat`:  Chat interface.
    *   `/new_session`:  Starts a new chat session.
    *   `/history`:  Chat history page.
    *   `/delete_session/<session_id>`:  Deletes a chat session.
    *   `/rename_session/<session_id>`:  Renames a chat session.
    *   `/download/<arxiv_id>`:  Downloads an arXiv PDF.
    *   `/interact/<arxiv_id>`:  Interacts with an arXiv paper.
    *   `/feedback`:  Collects user feedback.
    *   `/dashboard`:  Feedback dashboard.
    *   `/dashboard/stream`:  Real-time feedback updates.
    *   `/process-docs`: Manually triggers document processing.
    *   `/doc-summary`:  Retrieves a summary of processed documents.

*   **Session Management:** Uses Flask's session management to maintain chat state.

*   **Database Interaction:**  Uses `psycopg2` to interact with the PostgreSQL database.

*   **Asynchronous Tasks:** Uses `threading` to save chat sessions asynchronously.

*   **Error Handling:** Implements comprehensive error handling with logging.

*   **Template Rendering:** Uses Jinja2 templates (`templates/`) to render the web pages.

*   **Static Files:** Serves static files (CSS, JavaScript, images) from the `static/` directory.

### Evaluation (`eval.py`)

*   **Metrics:**
    *   **Latency:**  Measures the time taken for the API to respond.
    *   **Semantic Similarity:**  Calculates the cosine similarity between the generated answer and the ground truth answer using SentenceTransformer embeddings.
    *   **Faithfulness:**  Assesses how well the answer is supported by the retrieved sources.
    *   **Retrieval Metrics (Precision, Recall,F1-Score, NDCG):**  Evaluates the quality of the document retrieval process.

*   **Procedure:**
    1.  Loads question-answer pairs from a CSV file (`eval_questions_answers.csv`).
    2.  Sends queries to the API (`/chat`).
    3.  Computes the metrics.
    4.  Saves the results to a JSON file.

### Configuration

*   **`.env`:** Stores environment variables, including API keys, database credentials, and GCS bucket names.
*   **`config.py`:**  Defines configuration parameters for the Flask application.

### Frontend (HTML, CSS, JavaScript)

*   **`templates/`:** Contains Jinja2 templates for the web pages.
*   **`static/css/`:**  Contains CSS stylesheets.
*   **`static/js/`:**  Contains JavaScript files for client-side logic.
    *   **`script.js`:** Handles form submissions, feedback collection, and dynamic updates for the chat and interact pages.
    *   **`main.js`:** Handles form submissions for the query page.
    *   **`dashboard.js`:** Manages the dynamic updates and visualizations for the dashboard.

### Database

*   **PostgreSQL:** Used for storing chat session history and user feedback.
*   **FAISS:** Used for the vector database.
*   **Google Cloud SQL:** Used for hosting the PostgreSQL database.

## Setup and Deployment

1.  **Prerequisites:**
    *   Python 3.10+
    *   Google Cloud Platform (GCP) account with billing enabled.
    *   Service Account with appropriate permissions (Storage Object Admin, Cloud SQL Client).
    *   PostgreSQL instance on Cloud SQL.
    *   Two GCS buckets: one for documents and one for the FAISS index.

2.  **Clone the repository:**
    ```bash
    git clone https://github.com/wissalbenothmen/RAG-Powered-Multi-Agent-Conversational-AI.git
    cd RAG-Powered-Multi-Agent-Conversational-AI
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    *   Create a `.env` file in the project root.
    *   Populate the `.env` file with the necessary variables (see the provided `.env` file).  **Replace placeholders with your actual credentials.**
    *   Specifically, ensure `GOOGLE_API_KEY`, `DB_HOST`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`, `DOCUMENTS_BUCKET`, and `INDEX_BUCKET` are correctly set.

5.  **Upload PDF documents to the `DOCUMENTS_BUCKET` in GCS.**

6.  **Run the application:**
    ```bash
    python app.py
    ```
    This will:
    *   Initialize the PostgreSQL database (if it doesn't exist).
    *   Process the documents in the GCS bucket and create/update the FAISS index.
    *   Start the Flask development server.

7. **Initial document processing:**
    * Once the application is running, you can manually trigger the initial document processing by sending a POST request to `/process-docs`.  This step is *crucial* to build the initial FAISS index.  You can use a tool like `curl` or Postman:

    ```bash
    curl -X POST http://127.0.0.1:5000/process-docs
    ```
8.  **Access the application:**
    *   Open a web browser and go to `http://127.0.0.1:5000/`.

9.  **Run Evaluation (Optional):**
    *   Create a CSV file named `eval_questions_answers.csv` with "Question" and "Response" columns.
    *   Run the evaluation script:

        ```bash
        python eval.py
        ```

10. **Deployment (to Google Cloud Run - *Conceptual*):**

    *   **Containerize the application using Docker.**  You'll need to create a `Dockerfile`.
    *   **Build and push the Docker image to Google Container Registry (GCR).**
    *   **Deploy the image to Cloud Run.**  Configure the service to use the correct environment variables and connect to your Cloud SQL instance.  Ensure your Cloud Run service has the necessary service account permissions.
        *   **Service Account:**  Ensure the service account associated with your Cloud Run service has the following roles:
            *   `roles/storage.objectAdmin`:  For accessing GCS buckets.
            *   `roles/cloudsql.client`:  For connecting to the Cloud SQL instance.
            *   Add any other necessary roles based on your specific requirements.

        *  **Cloud SQL Connection:** Use the Cloud SQL connection name (e.g., `your-project:your-region:your-instance`) in your `DB_HOST` environment variable when deploying to Cloud Run.  Cloud Run automatically handles the secure connection to Cloud SQL.

## Justification of Choices

*   **PostgreSQL:** Chosen for its robustness, scalability, ACID compliance, and excellent support for JSON data (used for storing chat conversations).
*   **FAISS:** Selected for its efficiency in performing similarity searches on large datasets of high-dimensional vectors.
*   **Google Gemini 1.5 Pro:** Provides state-of-the-art language understanding and generation capabilities, making it ideal for a conversational AI system.
*   **Hugging Face `all-MiniLM-L6-v2`:** A compact and efficient sentence embedding model that provides a good balance between performance and resource usage.
*   **Google Cloud Storage:** Provides scalable and cost-effective storage for documents and the FAISS index.
*   **LangChain:** Simplifies the integration of different components (LLM, vector database, prompts) into a coherent processing chain.
*   **Flask:** A lightweight and flexible web framework well-suited for building web applications and APIs.
*   **Multi-agent Architecture:** Improves code organization, maintainability, and allows for easier extension with new functionalities.

## Evaluation Metrics

The `eval.py` script calculates the following metrics:

*   **Latency:** The time it takes for the system to generate a response. Lower latency is better.
*   **Semantic Similarity:** Measures how similar the generated answer is to the ground truth answer, using cosine similarity of sentence embeddings. Higher similarity is better.
*   **Faithfulness:** Evaluates whether the generated answer is supported by the retrieved sources.  Higher faithfulness is better.
*   **Precision:** Measures the proportion of retrieved sources that are relevant to the question.  Higher precision is better.
*   **Recall:**  Measures the proportion of relevant information that is retrieved by the system. Higher recall is better.
*   **NDCG (Normalized Discounted Cumulative Gain):**  A ranking quality metric that considers the position of relevant documents in the retrieved list. Higher NDCG is better.
*   **F1-Score:** Harmonic mean of precision and recall, providing a balanced measure of retrieval effectiveness.Higher F1-Score reflects a robust retrieval process.

These metrics provide a comprehensive assessment of the system's performance, covering both the quality of the generated answers and the effectiveness of the retrieval process. The results are saved in a JSON file for detailed analysis.
