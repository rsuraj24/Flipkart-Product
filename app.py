from flask import Flask, render_template, request, Response
from prometheus_client import Counter, generate_latest
from flipkart.data_ingestion import DataIngestor
from flipkart.rag_chain import RAGChainBuilder
from dotenv import load_dotenv

load_dotenv()

# Prometheus counter
REQUEST_COUNT = Counter("http_requests_total", "Total HTTP Request")


def create_app():
    app = Flask(__name__)

    # ------------------------------
    # Initialize vector store and RAG chain
    # ------------------------------
    vector_store = DataIngestor().ingest(load_existing=True)
    rag_chain_builder = RAGChainBuilder(vector_store)
    rag_chain = rag_chain_builder.build_chain()

    # ------------------------------
    # Home page
    # ------------------------------
    @app.route("/")
    def index():
        REQUEST_COUNT.inc()
        return render_template("index.html")

    # ------------------------------
    # Get response from RAG chain
    # ------------------------------
    @app.route("/get", methods=["POST"])
    def get_response():
        user_input = request.form["msg"]
        session_id = "user-session"  # can be dynamic per user

        # Get chat history object for this session
        history = rag_chain_builder._get_history(session_id)

        # Prepare input dict for RunnablePassthrough
        invoke_input = {
            "input": user_input,
            "chat_history": history
        }

        # Invoke RAG chain properly
        response = rag_chain.invoke(
            invoke_input,
            config={"configurable": {"session_id": session_id}}
        )["answer"]

        return response

    # ------------------------------
    # Prometheus metrics endpoint
    # ------------------------------
    @app.route("/metrics")
    def metrics():
        return Response(generate_latest(), mimetype="text/plain")

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
