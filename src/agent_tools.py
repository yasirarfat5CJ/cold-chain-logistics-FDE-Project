import os
import urllib
import requests
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings
# ==========================================
# 1. ENVIRONMENT & DYNAMIC INDEX ATTACHMENT
# ==========================================
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent  

load_dotenv(dotenv_path=project_root / ".env")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

def get_cached_ollama_embeddings(model_name: str):
    """
    Loads and caches Ollama embeddings so the embedding model
    is reused across Streamlit reruns.
    """
    @st.cache_resource(show_spinner=False)
    def _load_model(name: str):
        print(f"🧠 Loading Ollama embedding model: [{name}]")
        return OllamaEmbeddings(model=name)

    return _load_model(model_name)


EMBEDDINGS_MODEL_SETTING = os.getenv(
    "Embeddings_model",
    "nomic-embed-text"
).strip()

db_host = os.getenv("SQL_SERVER_HOST", "localhost")
db_port = os.getenv("SQL_SERVER_PORT", "1433")
db_user = os.getenv("SQL_AGENT_USER", "USR_FDE_RO")
db_password = os.getenv("SQL_AGENT_PASSWORD")

if not PINECONE_API_KEY:
    raise ValueError("CRITICAL: Ensure PINECONE_API_KEY is present in your active .env profile.")

if EMBEDDINGS_MODEL_SETTING == "OPENAI":
    print("🤖 Mode: Connecting to Cloud OpenAI Index (1536 Dim Space)...")

    embeddings = OpenAIEmbeddings()
    INDEX_NAME = "fde-sop-index-openai"

else:
    local_model_target = os.getenv(
        "Local_Embedding_Model",
        "nomic-embed-text"
    ).strip()

    print(
        f"🦙 Mode: Connecting to Local Ollama "
        f"[{local_model_target}] Embedding Index..."
    )

    from langchain_ollama import OllamaEmbeddings

    embeddings = OllamaEmbeddings(
        model=local_model_target
    )

    INDEX_NAME = "fde-sop-index-ollama"


vector_store = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)
# ==========================================
# 2. CORE FDE AGENT TOOLS
# ==========================================

@tool
def query_telemetry_db(sql_query: str) -> str:
    """
    Executes a SQL SELECT query against the
    FDE_VIEWS.VW_ACTIVE_FLEET semantic view.

    Available columns:
    Timestamp,
    Latitude,
    Longitude,
    Current_Temperature_C,
    Cargo_Condition_Code,
    Risk_Classification,
    Delay_Probability,
    Port_Congestion_Level,
    Route_Risk_Index.

    Only SELECT queries against the semantic view are allowed.
    """

    connection_string = (
        f"DRIVER={{ODBC Driver 18 for SQL Server}};"
        f"SERVER={db_host},{db_port};"
        f"DATABASE=master;"
        f"UID={db_user};"
        f"PWD={db_password};"
        f"Encrypt=no;"
        f"TrustServerCertificate=yes;"
    )

    params = urllib.parse.quote_plus(connection_string)

    engine = create_engine(
        f"mssql+pyodbc:///?odbc_connect={params}"
    )

    try:
        query = sql_query.strip()

        # ------------------------------------------
        # SECURITY CHECK 1: Only SELECT
        # ------------------------------------------

        if not query.upper().startswith("SELECT"):
            return (
                "SECURITY BLOCK: "
                "Only SELECT operations are authorized."
            )

        # ------------------------------------------
        # SECURITY CHECK 2: Only semantic view
        # ------------------------------------------

        if "FDE_VIEWS.VW_ACTIVE_FLEET" not in query.upper():
            return (
                "SECURITY BLOCK: "
                "Queries must use FDE_VIEWS.VW_ACTIVE_FLEET."
            )

        # ------------------------------------------
        # Execute query
        # ------------------------------------------

        with engine.connect() as conn:

            cursor = conn.execute(text(query))

            columns = list(cursor.keys())

            # Limit result size returned to the LLM
            rows = cursor.fetchmany(10)

            if not rows:
                return "No records matched the query criteria."

            formatted_output = (
                f"COLUMNS: {', '.join(columns)}\n"
            )

            for row in rows:
                formatted_output += (
                    str(tuple(row)) + "\n"
                )

            return formatted_output

    except Exception as e:
        return f"Database Error: {str(e)}"

    finally:
        engine.dispose()


# ==========================================
# 4. LIVE CORRIDOR CONDITIONS TOOL
# ==========================================

@tool
def fetch_corridor_conditions(
    latitude: float,
    longitude: float
) -> str:
    """
    Fetches real-time weather and corridor conditions
    from Open-Meteo for the given GPS coordinates.

    Provides:
    - External temperature
    - Wind speed
    - Computed corridor congestion index
    """

    try:

        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={latitude}"
            f"&longitude={longitude}"
            "&current_weather=true"
        )

        response = requests.get(
            url,
            timeout=6
        )

        response.raise_for_status()

        payload = response.json().get(
            "current_weather",
            {}
        )

        temp = payload.get(
            "temperature",
            "N/A"
        )

        wind = payload.get(
            "windspeed",
            0.0
        )

        # ------------------------------------------
        # Corridor risk calculation
        # ------------------------------------------

        congestion_index = (
            8.5 if wind > 10.0 else 2.5
        )

        status_note = (
            "High Transit Disruption"
            if wind > 10.0
            else "Corridor Normal"
        )

        return (
            "--- LIVE CORRIDOR TELEMETRY ---\n"
            f"Target GPS: {latitude}, {longitude}\n"
            f"External Temp: {temp}°C\n"
            f"Wind Speed: {wind} km/h\n"
            f"Corridor Risk: {status_note}\n"
            f"Congestion Index: {congestion_index}/10\n"
            "-------------------------------"
        )

    except Exception as e:

        return (
            "Corridor API Communication Failure: "
            f"{str(e)}"
        )


# ==========================================
# 5. COMPLIANCE SOP SEARCH TOOL
# ==========================================

@tool
def search_compliance_sop(query: str) -> str:
    """
    Searches enterprise Standard Operating Procedures
    indexed in the Pinecone Vector DB.

    Uses Ollama embeddings for semantic retrieval.

    Retrieves information about:
    - Regulatory thresholds
    - Cold-chain breach mitigation
    - Rerouting rules
    - Compliance procedures
    """

    try:

        matched_docs = retriever.invoke(query)

        if not matched_docs:
            return (
                "No matching compliance clauses found."
            )

        formatted_context = "\n\n".join(
            [
                (
                    f"[Source: "
                    f"{doc.metadata.get('source_file', 'SOP')} "
                    f"| Format: "
                    f"{doc.metadata.get('file_format', 'RAW')}]\n"
                    f"{doc.page_content}"
                )
                for doc in matched_docs
            ]
        )

        return (
            "--- COMPLIANCE SOP CONTEXT ---\n"
            f"{formatted_context}\n"
            "------------------------------"
        )

    except Exception as e:

        return (
            "Vector Store Retrieval Error: "
            f"{str(e)}"
        )


# ==========================================
# 6. LOCAL VERIFICATION
# ==========================================

if __name__ == "__main__":

    # ------------------------------------------
    # Test SQL Telemetry Tool
    # ------------------------------------------

    print(
        "\n--- Testing Tool 1: "
        "SQL Telemetry View ---"
    )

    print(
        query_telemetry_db.invoke(
            """
            SELECT TOP 2
                Latitude,
                Longitude,
                Current_Temperature_C
            FROM FDE_VIEWS.VW_ACTIVE_FLEET
            """
        )
    )


    # ------------------------------------------
    # Test Live Corridor API
    # ------------------------------------------

    print(
        "\n--- Testing Tool 2: "
        "Live Corridor API ---"
    )

    print(
        fetch_corridor_conditions.invoke(
            {
                "latitude": 33.77,
                "longitude": -118.19
            }
        )
    )


    # ------------------------------------------
    # Test Pinecone + Ollama Retrieval
    # ------------------------------------------

    print(
        "\n--- Testing Tool 3: "
        "Pinecone Vector Retrieval ---"
    )

    print(
        search_compliance_sop.invoke(
            "What are the temperature rules "
            "for fresh perishables?"
        )
    )