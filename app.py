"""Flask application for a RAG-powered multi-agent conversational AI system.

This module provides a web interface for chatting with an AI system powered by
Retrieval-Augmented Generation (RAG), integrating vector databases, arXiv search,
and Google Gemini LLM. It manages chat sessions with PostgreSQL, processes documents
from Google Cloud Storage, and collects user feedback.
"""

import json
import logging
import os
import re
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime
from io import BytesIO

import arxiv
import numpy as np
import psycopg2
import psycopg2.extras
import requests
from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_file,
    session,
)
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI

from agents.answer_generator import generate_answer
from agents.feedback_collector import FeedbackCollector
from agents.source_tracker import track_sources
from rag.process_docs import process_documents
from rag.vectordb import VectorDB

# Initialize Flask app
app = Flask(__name__)
app.config.from_object("config.Config")
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your-secret-key")

# Setup logging configuration
log_file = os.path.join(os.path.dirname(__file__), f'app_{datetime.now().strftime("%Y%m%d")}.log')
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(pathname)s:%(lineno)d - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
)
logger = logging.getLogger(__name__)

def handle_exception(exc_type, exc_value, exc_traceback):
    """Handle uncaught exceptions and log them appropriately."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception

def get_db_connection():
    """Establish a connection to the PostgreSQL database."""
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),  # Public IP: 34.38.195.191
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        dbname=os.getenv("DB_NAME"),
        port=5432,
        cursor_factory=psycopg2.extras.DictCursor,
    )

def init_chat_db():
    """Initialize the chat sessions table in PostgreSQL if it doesn't exist."""
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id VARCHAR(36) PRIMARY KEY,
                    title VARCHAR(255) NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    conversation JSONB NOT NULL
                )
                """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON chat_sessions(timestamp)")
        conn.commit()
    logger.info("Chat database initialized with PostgreSQL")

init_chat_db()

# Global components for the application
feedback_collector = FeedbackCollector()
vector_db = VectorDB(data_dir="gs://rag-multiagent-documents/", index_dir="gs://rag-multiagent-index/")
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7,
    top_p=0.95,
    top_k=40,
)

class NpEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy types for serialization."""

    def default(self, obj):  # Renamed 'o' to 'obj' to match convention
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def search_arxiv(query, max_results=3):
    """Search arXiv for papers related to the given query."""
    start_time = time.time()
    try:
        search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
        results = list(search.results())
        arxiv_refs = [
            {
                "title": result.title,
                "id": result.entry_id.split("/")[-1],
                "summary": result.summary,
                "pdf_url": result.pdf_url,
                "similarity_score": 0.9 - (i * 0.1),
            }
            for i, result in enumerate(results)
        ]
        logger.debug("Retrieved %d arXiv references for query '%s' in %f seconds",
                     len(arxiv_refs), query, time.time() - start_time)
        return arxiv_refs
    except arxiv.ArxivError as e:
        logger.error("Error searching arXiv: %s", str(e), exc_info=True)
        return []

# Initial document processing at startup
logger.info("Starting document processing on application startup")
try:
    process_documents(
        data_dir="gs://rag-multiagent-documents/",
        top_k=10,
        force_replace_index=True,
        force_reindex_all=False,
    )
    logger.info("Document processing completed successfully")
except ValueError as e:
    logger.error("Failed to process documents on startup: %s", str(e), exc_info=True)

def save_chat_session_async(session_id, title, conversation):
    """Asynchronously save a chat session to the database."""
    def save_to_db():
        start_time = time.time()
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO chat_sessions (session_id, title, timestamp, conversation)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (session_id) DO UPDATE SET
                            title = EXCLUDED.title,
                            timestamp = EXCLUDED.timestamp,
                            conversation = EXCLUDED.conversation
                        """,
                        (session_id, title, datetime.now().isoformat(), json.dumps(conversation)),
                    )
                conn.commit()
            logger.debug("Saved session %s in %f seconds", session_id, time.time() - start_time)
        except psycopg2.Error as e:
            logger.error("Error saving session %s: %s", session_id, str(e), exc_info=True)

    threading.Thread(target=save_to_db).start()

def get_chat_history():
    """Retrieve all chat history entries from the database."""
    start_time = time.time()
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT session_id, title, timestamp, conversation "
                    "FROM chat_sessions ORDER BY timestamp DESC"
                )
                history = [  # 'history' redefined intentionally for local scope
                    {
                        "session_id": row["session_id"],
                        "title": row["title"],
                        "timestamp": row["timestamp"].isoformat(),
                        "conversation": row["conversation"],
                    }
                    for row in cursor.fetchall()
                ]
        logger.debug("Retrieved %d chat history entries in %f seconds",
                     len(history), time.time() - start_time)
        return history
    except psycopg2.Error as e:
        logger.error("Error retrieving chat history: %s", str(e), exc_info=True)
        return []

def delete_chat_session(session_id):
    """Delete a specific chat session from the database."""
    start_time = time.time()
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM chat_sessions WHERE session_id = %s", (session_id,))
            conn.commit()
        logger.debug("Deleted session %s in %f seconds", session_id, time.time() - start_time)
    except psycopg2.Error as e:
        logger.error("Error deleting session %s: %s", session_id, str(e), exc_info=True)

def rename_chat_session(session_id, new_title):
    """Rename a specific chat session in the database."""
    start_time = time.time()
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "UPDATE chat_sessions SET title = %s WHERE session_id = %s",
                    (new_title, session_id),
                )
            conn.commit()
        logger.debug("Renamed session %s to '%s' in %f seconds",
                     session_id, new_title, time.time() - start_time)
    except psycopg2.Error as e:
        logger.error("Error renaming session %s: %s", session_id, str(e), exc_info=True)

@app.route("/")
def index():
    """Render the main index page of the application."""
    return render_template("index.html")

@app.route("/chat", methods=["GET", "POST"])
def chat():
    """Handle user chat interactions, including queries and responses."""
    max_session_size = 50  # Maximum number of messages per session
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
        session["conversation"] = []
        logger.debug("New session created: %s", session["session_id"])

    if request.method == "POST":
        start_time = time.time()
        data = request.get_json()
        user_input = data.get("user_input")
        logger.debug("Received user input: '%s' (length: %d)", user_input, len(user_input))

        # Handle special "explain more" queries
        explain_keywords = ["explain more", "more details", "détails supplémentaires"]
        if any(keyword in user_input.lower() for keyword in explain_keywords):
            if len(session["conversation"]) >= 2:
                last_user_msg = next(
                    (msg for msg in reversed(session["conversation"]) if msg["type"] == "user"),
                    None,
                )
                last_assistant_msg = next(
                    (msg for msg in reversed(session["conversation"]) if msg["type"] == "assistant"),
                    None,
                )
                if last_user_msg and last_assistant_msg:
                    prompt = (
                        f"Previous Question: {last_user_msg['content']}\n\n"
                        f"Previous Answer: {last_assistant_msg['content']}\n\n"
                        f"Current Request: {user_input}\n\n"
                        "Please provide a detailed explanation based on the previous question and "
                        "answer, expanding on the topic with additional insights."
                    )
                    try:
                        detailed_answer = llm.invoke(prompt).content.strip()
                        response = {
                            "status": "success",
                            "answer": detailed_answer,
                            "sources": last_assistant_msg.get("sources", []),
                            "arxiv_references": last_assistant_msg.get("arxiv_references", []),
                            "response_time": time.time() - start_time,
                            "sources_count": len(last_assistant_msg.get("sources", [])),
                        }
                    except ValueError as e:
                        logger.error("Error generating detailed answer with Gemini: %s", str(e),
                                     exc_info=True)
                        response = {
                            "status": "error",
                            "answer": "Sorry, I couldn't generate a detailed explanation at this time.",
                            "sources": [],
                            "arxiv_references": [],
                            "response_time": time.time() - start_time,
                            "sources_count": 0,
                        }
            else:
                response = {
                    "status": "success",
                    "answer": "You haven’t asked anything yet to explain more about!",
                    "sources": [],
                    "arxiv_references": [],
                    "response_time": time.time() - start_time,
                    "sources_count": 0,
                }
            session["conversation"].append({"type": "user", "content": user_input})
            session["conversation"].append({
                "type": "assistant",
                "content": response["answer"],
                "sources": response["sources"],
                "arxiv_references": response["arxiv_references"],
            })
            session["conversation"] = session["conversation"][-max_session_size:]
            save_chat_session_async(
                session["session_id"],
                session["conversation"][0]["content"] if session["conversation"] else "Untitled",
                session["conversation"],
            )
            logger.debug("Response time for explain more query: %f seconds", response["response_time"])
            return Response(json.dumps(response, cls=NpEncoder), mimetype="application/json")

        # Handle "last question" queries
        last_question_keywords = ["last question", "previous question"]
        if any(keyword in user_input.lower() for keyword in last_question_keywords):
            last_question = (
                next((msg["content"] for msg in reversed(session["conversation"])
                      if msg["type"] == "user"), None)
                if len(session["conversation"]) >= 1 else None
            )
            response = {
                "status": "success",
                "answer": (f"Your last question was: '{last_question}'" if last_question
                           else "No previous question found."),
                "sources": [],
                "arxiv_references": [],
                "response_time": time.time() - start_time,
                "sources_count": 0,
            } if last_question is not None else {
                "status": "success",
                "answer": "You haven't asked any questions yet in this session!",
                "sources": [],
                "arxiv_references": [],
                "response_time": time.time() - start_time,
                "sources_count": 0,
            }
            session["conversation"].append({"type": "user", "content": user_input})
            session["conversation"].append({"type": "assistant", "content": response["answer"]})
            session["conversation"] = session["conversation"][-max_session_size:]
            logger.debug("Response time for last question query: %f seconds", response["response_time"])
            return Response(json.dumps(response, cls=NpEncoder), mimetype="application/json")

        # Handle greeting inputs
        greetings = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"]
        matched_greetings = [greeting for greeting in greetings
                             if re.search(r"\b" + re.escape(greeting) + r"\b", user_input.lower())]
        if matched_greetings:
            logger.info("Detected greeting(s): %s", matched_greetings)
            response = {
                "status": "success",
                "answer": "Hello! How can I assist you today?",
                "sources": [],
                "arxiv_references": [],
                "response_time": time.time() - start_time,
                "sources_count": 0,
            }
            session["conversation"].append({"type": "user", "content": user_input})
            session["conversation"].append({"type": "assistant", "content": response["answer"]})
            session["conversation"] = session["conversation"][-max_session_size:]
            save_chat_session_async(
                session["session_id"],
                session["conversation"][0]["content"] if session["conversation"] else "Untitled",
                session["conversation"],
            )
            logger.debug("Response time for greeting: %f seconds", response["response_time"])
            return Response(json.dumps(response, cls=NpEncoder), mimetype="application/json")

        # Process standard queries
        try:
            sources = vector_db.search(user_input, top_k=20)
            arxiv_refs = search_arxiv(user_input)
            answer = generate_answer(user_input, sources, enable_web_search=True)
            response_time = time.time() - start_time
            response = {
                "status": "success",
                "answer": answer,
                "sources": track_sources(sources),
                "arxiv_references": arxiv_refs,
                "response_time": response_time,
                "sources_count": len(sources),
            }
            session["conversation"].append({"type": "user", "content": user_input})
            session["conversation"].append({
                "type": "assistant",
                "content": answer,
                "sources": response["sources"],
                "arxiv_references": response["arxiv_references"],
            })
            session["conversation"] = session["conversation"][-max_session_size:]
            save_chat_session_async(
                session["session_id"],
                session["conversation"][0]["content"] if session["conversation"] else "Untitled",
                session["conversation"],
            )
            logger.debug("Response time for query '%s': %f seconds", user_input, response_time)
            return Response(json.dumps(response, cls=NpEncoder), mimetype="application/json")
        except ValueError as e:
            logger.error("Error processing chat request: %s", str(e), exc_info=True)
            response_time = time.time() - start_time
            return Response(
                json.dumps({"status": "error", "message": str(e), "response_time": response_time},
                           cls=NpEncoder),
                mimetype="application/json",
                status=500,
            )
    return render_template("chat.html", session_id=session["session_id"],
                          conversation=session.get("conversation", []))

@app.route("/new_session", methods=["POST"])
def new_session():
    """Create a new chat session and reset conversation history."""
    start_time = time.time()
    session["session_id"] = str(uuid.uuid4())
    session["conversation"] = []
    response_time = time.time() - start_time
    logger.debug("New session %s created in %f seconds", session["session_id"], response_time)
    return jsonify({"status": "success", "session_id": session["session_id"]})

@app.route("/history")
def history():
    """Render the chat history page with all past sessions."""
    start_time = time.time()
    chat_history = get_chat_history()
    logger.debug("Chat history data: %s", chat_history)
    response_time = time.time() - start_time
    logger.debug("History page loaded in %f seconds", response_time)
    return render_template("history.html", chat_history=chat_history)

@app.route("/delete_session/<session_id>", methods=["POST"])
def delete_session(session_id):
    """Delete a specific chat session from the database and session."""
    start_time = time.time()
    delete_chat_session(session_id)
    if session.get("session_id") == session_id:
        session.pop("session_id", None)
        session.pop("conversation", None)
    response_time = time.time() - start_time
    logger.debug("Session %s deleted in %f seconds", session_id, response_time)
    return jsonify({"status": "success"})

@app.route("/rename_session/<session_id>", methods=["POST"])
def rename_session(session_id):
    """Rename a specific chat session with a new title."""
    start_time = time.time()
    new_title = request.get_json().get("title")
    rename_chat_session(session_id, new_title)
    response_time = time.time() - start_time
    logger.debug("Session %s renamed to '%s' in %f seconds", session_id, new_title, response_time)
    return jsonify({"status": "success", "new_title": new_title})

@app.route("/download/<arxiv_id>")
def download_arxiv(arxiv_id):
    """Download an arXiv PDF by its ID."""
    start_time = time.time()
    try:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        response = requests.get(pdf_url, timeout=10)
        response.raise_for_status()

        pdf_content = BytesIO(response.content)
        response_time = time.time() - start_time
        logger.debug("Downloaded arXiv %s in %f seconds", arxiv_id, response_time)
        return send_file(
            pdf_content,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=f"{arxiv_id}.pdf",
        )
    except requests.RequestException as e:
        logger.error("Error downloading arXiv PDF %s: %s", arxiv_id, str(e), exc_info=True)
        response_time = time.time() - start_time
        return jsonify({
            "status": "error",
            "message": str(e),
            "response_time": response_time,
        }), 500

@app.route("/interact/<arxiv_id>", methods=["GET", "POST"])
def interact(arxiv_id):
    """Interact with an arXiv paper by querying its content or viewing a summary."""
    start_time = time.time()
    if request.method == "POST":
        data = request.get_json()
        query = data.get("query", "").strip()
        try:
            if not query:
                raise ValueError("Query is required")
            logger.debug("Received query for arXiv %s: %s", arxiv_id, query)

            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            response = requests.get(pdf_url, timeout=10)
            response.raise_for_status()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(response.content)
                temp_file_path = temp_file.name

            try:
                loader = PyMuPDFLoader(temp_file_path)
                documents = loader.load()
                pdf_text = "\n".join([doc.page_content for doc in documents])
                logger.debug("Loaded PDF content (first 100 chars): %s", pdf_text[:100])

                prompt = (
                    "Based on the following PDF content:\n\n"
                    f"{pdf_text[:5000]}\n\n"
                    f"Answer this question: {query}"
                )
                gemini_response = llm.invoke(prompt).content.strip()
                logger.debug("Gemini response: %s...", gemini_response[:100])

                response = {
                    "status": "success",
                    "answer": gemini_response,
                    "pdf_url": pdf_url,
                    "response_time": time.time() - start_time,
                }
                logger.debug("Sending response for arXiv %s in %f seconds",
                             arxiv_id, response["response_time"])
                return jsonify(response)
            finally:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    logger.debug("Cleaned up temporary file: %s", temp_file_path)
        except (requests.RequestException, ValueError) as e:
            logger.error("Error in /interact/%s: %s", arxiv_id, str(e), exc_info=True)
            response_time = time.time() - start_time
            return jsonify({
                "status": "error",
                "message": str(e),
                "response_time": response_time,
            }), 500
    else:
        try:
            search = arxiv.Search(id_list=[arxiv_id], max_results=1)
            result = next(search.results())
            abstract = result.summary
            pdf_url = result.pdf_url
            logger.debug("Fetched abstract for %s: %s...", arxiv_id, abstract[:100])

            prompt = (
                "Based on the following abstract:\n\n"
                f"{abstract}\n\n"
                "Provide a brief summary of what this article is about in 2-3 sentences."
            )
            summary = llm.invoke(prompt).content.strip()
            logger.debug("Generated summary: %s...", summary[:100])

            messages = [
                {"type": "user", "content": "What is this article about?"},
                {"type": "assistant", "content": summary, "pdf_url": pdf_url},
            ]
            response_time = time.time() - start_time
            logger.debug("Rendering interact.html for %s in %f seconds", arxiv_id, response_time)
            return render_template("interact.html", arxiv_id=arxiv_id, messages=messages)
        except StopIteration as e:
            logger.error("Error fetching arXiv metadata for %s: %s", arxiv_id, str(e), exc_info=True)
            response_time = time.time() - start_time
            messages = [
                {"type": "user", "content": "What is this article about?"},
                {
                    "type": "assistant",
                    "content": f"Unable to retrieve summary: {str(e)}. Please try again.",
                    "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                },
            ]
            return render_template("interact.html", arxiv_id=arxiv_id, messages=messages)

@app.route("/feedback", methods=["POST"])
def feedback():
    """Collect and store user feedback."""
    start_time = time.time()
    try:
        feedback_data = request.get_json()
        if not feedback_data:
            raise ValueError("No feedback data provided")
        collected_feedback = feedback_collector.collect_feedback(feedback_data)
        response_time = time.time() - start_time
        logger.info("Feedback collected: ID %s in %f seconds", collected_feedback['id'], response_time)
        return Response(
            json.dumps({
                "status": "success",
                "feedback": collected_feedback,
                "message": "Feedback saved successfully",
                "response_time": response_time,
            }, cls=NpEncoder),
            mimetype="application/json",
        )
    except ValueError as e:
        logger.error("Error processing feedback: %s", str(e), exc_info=True)
        response_time = time.time() - start_time
        return Response(
            json.dumps({
                "status": "error",
                "message": str(e),
                "response_time": response_time,
            }, cls=NpEncoder),
            mimetype="application/json",
            status=500,
        )

@app.route("/dashboard")
def dashboard():
    """Render the feedback dashboard with statistics."""
    start_time = time.time()
    feedback_stats = feedback_collector.get_feedback_stats()
    response_time = time.time() - start_time
    logger.debug("Dashboard loaded in %f seconds", response_time)
    return render_template("dashboard.html", stats=feedback_stats)

@app.route("/dashboard/stream")
def dashboard_stream():
    """Stream real-time feedback updates to the dashboard."""
    def generate():
        while True:
            start_time = time.time()
            all_feedbacks = feedback_collector.get_all_feedback()
            stats = {"all_feedbacks": all_feedbacks, "timestamp": datetime.now().isoformat()}
            response_time = time.time() - start_time
            logger.debug("Dashboard stream update generated in %f seconds", response_time)
            yield f"data: {json.dumps(stats, cls=NpEncoder)}\n\n"
            time.sleep(5)

    return Response(generate(), mimetype="text/event-stream")

@app.route("/process-docs", methods=["POST"])
def process_docs():
    """Manually trigger document processing for the vector database."""
    start_time = time.time()
    try:
        logger.info("Manual document processing triggered")
        process_documents(
            data_dir="gs://rag-multiagent-documents/",
            top_k=10,
            force_replace_index=True,
            force_reindex_all=False,
        )
        summary = vector_db.db_manager.get_summary()
        response_time = time.time() - start_time
        logger.info("Manual processing completed in %f seconds: %s", response_time, summary)
        return Response(
            json.dumps({
                "status": "success",
                "message": "Documents processed successfully",
                "summary": summary,
                "response_time": response_time,
            }, cls=NpEncoder),
            mimetype="application/json",
        )
    except ValueError as e:
        logger.error("Error processing documents: %s", str(e), exc_info=True)
        response_time = time.time() - start_time
        return Response(
            json.dumps({
                "status": "error",
                "message": str(e),
                "response_time": response_time,
            }, cls=NpEncoder),
            mimetype="application/json",
            status=500,
        )

@app.route("/doc-summary", methods=["GET"])
def doc_summary():
    """Retrieve and return a summary of processed documents."""
    start_time = time.time()
    try:
        summary = vector_db.db_manager.get_summary()
        response_time = time.time() - start_time
        logger.debug("Document summary requested in %f seconds: %s", response_time, summary)
        return Response(
            json.dumps({
                "status": "success",
                "summary": summary,
                "response_time": response_time,
            }, cls=NpEncoder),
            mimetype="application/json",
        )
    except AttributeError as e:
        logger.error("Error retrieving summary: %s", str(e), exc_info=True)
        response_time = time.time() - start_time
        return Response(
            json.dumps({
                "status": "error",
                "message": str(e),
                "response_time": response_time,
            }, cls=NpEncoder),
            mimetype="application/json",
            status=500,
        )

def datetimeformat(value, format_str="%Y-%m-%d %H:%M:%S"):
    """Format datetime strings for use in Jinja templates."""
    if isinstance(value, str):
        return datetime.fromisoformat(value).strftime(format_str)
    return value

app.jinja_env.filters["datetimeformat"] = datetimeformat

if __name__ == "__main__":
    """Entry point for running the Flask application."""
    logger.info("Starting Flask application")
    app.run(debug=app.config["FLASK_DEBUG"], host="0.0.0.0", port=5000)

#End-of-file
