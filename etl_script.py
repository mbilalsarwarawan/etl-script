import psycopg2
from collections import defaultdict
from qdrant_client.models import VectorParams, Distance
from langchain_qdrant import QdrantVectorStore
from langchain.text_splitter import RecursiveJsonSplitter
from langchain_ollama import OllamaEmbeddings

from qdrant_client import QdrantClient

qdrant_client = QdrantClient(
    url="https://2f1176fd-c1c3-455b-aca6-c8343c1424d0.us-east4-0.gcp.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.nR1qOpeLaSNuk1lB8Rh6AFmTjkGf5K0BUQMsJ5pwvfI",
)

print(qdrant_client.get_collections())

print(qdrant_client.get_collections())

# Configuration for your PostgreSQL connection
DB_CONFIG = {
    "dbname": "nomia2",
    "user": "nomia",
    "password": "iAmBB!",
    "host": "156.38.212.10",
    "port": "5432"
}

# Initialize Qdrant client
embedding_function = None  # Define your embedding function

ETL_QUERY = """
WITH latest_element AS (
  SELECT de.*
  FROM document_elements de
  LEFT JOIN document_elements de_new 
         ON de.id = de_new.supercedes_id
  WHERE de_new.id IS NULL
)
SELECT 
    pt.id AS published_template_id,
    pt.name AS filename,
    ds.heading,
    COALESCE(de_pub.content, le.content) AS final_content,
    w.organization_id,
    pt.workspace_id
FROM document_templates pt
JOIN workspaces w 
  ON pt.workspace_id = w.id
LEFT JOIN templates_sections ts 
  ON ts.document_template_id = pt.id
LEFT JOIN document_sections ds 
  ON ds.id = ts.document_section_id
LEFT JOIN sections_sub_sections sss 
  ON sss.document_section_id = ds.id
LEFT JOIN document_sub_sections dss 
  ON dss.id = sss.document_sub_section_id
LEFT JOIN latest_element le 
  ON le.document_sub_section_id = dss.id
LEFT JOIN LATERAL (
    SELECT de2.*
    FROM document_elements de2
    WHERE de2.document_sub_section_id = dss.id
      AND de2.publication_no IS NOT NULL
    ORDER BY de2.valid_from DESC
    LIMIT 1
) de_pub ON true
WHERE pt.published_date IS NOT NULL;
"""


def extract_data(conn):
    """Extract data by executing the ETL query."""
    with conn.cursor() as cur:
        cur.execute(ETL_QUERY)
        rows = cur.fetchall()
    return rows


def transform_to_json(rows):
    """Transform rows into a hierarchical JSON structure."""
    docs = defaultdict(lambda: {
        "file_id": None,
        "filename": None,
        "organization_id": None,
        "workspace_id": None,
        "file": []
    })

    for row in rows:
        doc_id, filename, heading, final_content, org_id, ws_id = row
        doc = docs[doc_id]
        doc["file_id"] = str(doc_id)
        doc["filename"] = filename
        doc["organization_id"] = str(org_id)
        doc["workspace_id"] = str(ws_id)
        doc["file"].append({
            "heading": heading,
            "content": final_content
        })
    return list(docs.values())


def get_org_workspace_vectorstore(organization_id: str, workspace_id: str):
    """Retrieve or create a vector store for an organization and workspace."""
    collection_name = f"org_{organization_id}_workspace_{workspace_id}"
    collection = qdrant_client.collection_exists(collection_name=collection_name)
    if not collection:
        qdrant_client.create_collection(
            collection_name=collection_name, vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
    return QdrantVectorStore(client=qdrant_client, collection_name=collection_name, embedding=OllamaEmbeddings(model="deepseek-r1:1.5b"))


def create_documents(doc_data):
    """Use RecursiveJsonSplitter to split content and create document objects."""

    text_splitter = RecursiveJsonSplitter(max_chunk_size=2500, min_chunk_size=1500)
    splits = text_splitter.create_documents(texts=[doc_data], convert_lists=True )

    for split in splits:
        split.metadata = {"file_id": doc_data["file_id"], "filename": doc_data["filename"]}

    return splits


def index_document_to_chroma(doc_data: dict) -> bool:
    """Index document content into ChromaDB."""
    try:
        vectorstore = get_org_workspace_vectorstore(doc_data["organization_id"], doc_data["workspace_id"])
        documents = create_documents(doc_data)
        vectorstore.add_documents(documents)
        return True
    except Exception as e:
        print(f"Error indexing document: {e}")
        return False


def main():
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        rows = extract_data(conn)
        print(f"Extracted {len(rows)} rows from the database.")

        documents = transform_to_json(rows)
        print(f"Transformed into {len(documents)} document JSON objects.")

        # Index documents into ChromaDB
        for doc in documents:
            success = index_document_to_chroma(doc)
            if success:
                print(f"Indexed document: {doc['file_id']}")
            else:
                print(f"Failed to index document: {doc['file_id']}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
