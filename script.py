import os
import psycopg2
import time
import requests
from collections import defaultdict

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

from qdrant_client import QdrantClient

qdrant_client = QdrantClient(
    url="https://41696e48-ee43-4811-901f-cc66429757dd.us-east4-0.gcp.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.QG0B8gDPhf6LyLT-H1lQCVlb4CLv3YOJj2GFAiMpGlE",
)

print(qdrant_client.get_collections())

print("Collections:", qdrant_client.get_collections())

# PostgreSQL connection configuration
DB_CONFIG = {
    "dbname": "nomia2",
    "user": "nomia",
    "password": "iAmBB!",
    "host": "156.38.212.10",
    "port": "5432"
}

# Updated ETL query to run for all published templates
ETL_QUERY = """
WITH latest_element AS (
  SELECT de.*
  FROM document_elements de
  LEFT JOIN document_elements de_new 
         ON de.id = de_new.supercedes_id
  WHERE de_new.id IS NULL
),
element_data AS (
  SELECT 
    pt.id AS published_template_id,
    pt.name AS filename,
    ds.id AS section_id,
    ds.heading AS section_heading,
    dss.id AS sub_section_id,
    dss.heading AS sub_section_heading,
    COALESCE(de_pub.content, le.content) AS element_content,
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
  WHERE pt.published_date IS NOT NULL
),
sub_section_agg AS (
  SELECT 
    published_template_id,
    section_id,
    MAX(section_heading) AS section_heading,
    sub_section_id,
    sub_section_heading,
    string_agg(element_content, E'\n' ORDER BY element_content) AS aggregated_elements
  FROM element_data
  GROUP BY published_template_id, section_id, sub_section_id, sub_section_heading
),
section_agg AS (
  SELECT 
    published_template_id,
    section_id,
    MAX(section_heading) AS section_heading,
    string_agg(
      sub_section_heading || E'\n' || aggregated_elements, 
      E'\n\n' ORDER BY sub_section_id
    ) AS aggregated_sub_sections
  FROM sub_section_agg
  GROUP BY published_template_id, section_id
),
template_agg AS (
  SELECT 
    meta.published_template_id,
    meta.filename,
    meta.organization_id,
    meta.workspace_id,
    string_agg(
      section_heading || E'\n' || aggregated_sub_sections, 
      E'\n\n' ORDER BY section_id
    ) AS full_text
  FROM (
    SELECT DISTINCT published_template_id, filename, organization_id, workspace_id
    FROM element_data
  ) meta
  JOIN section_agg sa 
    ON meta.published_template_id = sa.published_template_id
  GROUP BY meta.published_template_id, meta.filename, meta.organization_id, meta.workspace_id
)
SELECT 
  published_template_id AS template_id,
  filename,
  organization_id,
  workspace_id,
  full_text AS full_document_text
FROM template_agg;
"""

def extract_documents():
    """Extract documents from the database using the ETL query."""
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute(ETL_QUERY)
            rows = cur.fetchall()
            docs = []
            for row in rows:
                template_id, filename, organization_id, workspace_id, full_text = row
                docs.append({
                    "template_id": template_id,
                    "filename": filename,
                    "organization_id": organization_id,
                    "workspace_id": workspace_id,
                    "full_document_text": full_text
                })
            return docs
    finally:
        conn.close()

def write_txt_files(docs, output_dir="output_txts"):
    """Write each document's full text to a TXT file."""
    os.makedirs(output_dir, exist_ok=True)
    file_data = []
    for doc in docs:
        file_name = f"template_{doc['template_id']}_{doc['filename']}.txt"
        file_path = os.path.join(output_dir, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            content = doc.get("full_document_text") or ""
            f.write(content)
        file_data.append((file_path, doc))
    return file_data

def load_document_from_txt(file_path, metadata):
    """Load a single TXT file as a LangChain Document with metadata."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    doc_metadata = {
        "template_id": metadata["template_id"],
        "filename": metadata["filename"],
        "organization_id": metadata["organization_id"],
        "workspace_id": metadata["workspace_id"],
        "source": file_path
    }
    return Document(page_content=text, metadata=doc_metadata)

def get_org_workspace_vectorstore(organization_id, workspace_id):
    """
    Retrieve or create a Qdrant vector store for a given organization and workspace.
    """
    collection_name = f"org_{organization_id}_workspace_{workspace_id}"
    try:
        qdrant_client.get_collection(collection_name=collection_name)
    except Exception:
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )
    embedding = OllamaEmbeddings(model="deepseek-r1:1.5b")  # Ensure the model is available
    return QdrantVectorStore(qdrant_client, collection_name, embedding=embedding)

def split_document(document, chunk_size=1000, chunk_overlap=50):
    """Split a document into chunks using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents([document])

def add_chunks_to_qdrant(chunks, organization_id, workspace_id, batch_size=20, max_retries=3):
    """Add document chunks to Qdrant in batches to avoid timeouts."""
    vectorstore = get_org_workspace_vectorstore(organization_id, workspace_id)
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        retries = 0
        while retries < max_retries:
            try:
                vectorstore.add_documents(batch)
                print(f"Added batch {i // batch_size + 1} with {len(batch)} chunks.")
                break
            except Exception as e:
                retries += 1
                print(f"Error adding batch {i // batch_size + 1} (attempt {retries}): {e}")
                time.sleep(2 ** retries)
        else:
            print(f"Failed to add batch {i // batch_size + 1} after {max_retries} retries.")

def notify_indexed_document(template_id, filename, organization_id, workspace_id):
    """Notify an external service that a document was indexed."""
    url = "http://localhost:8081/add-document/"
    payload = {
        "file_id": str(template_id),
        "filename": str(filename),
        "organization_id": str(organization_id),
        "workspace_id": str(workspace_id)
    }
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            print(f"Notified indexing for document (template_id: {template_id}).")
        else:
            print(f"Failed to notify indexing for document (template_id: {template_id}). "
                  f"Status: {response.status_code}, Response: {response.text}")
    except Exception as e:
        print(f"Exception notifying indexing for document (template_id: {template_id}): {e}")

def process_file(file_tuple):
    """Process a single TXT file: load, split, index, and notify."""
    file_path, metadata = file_tuple
    document = load_document_from_txt(file_path, metadata)
    chunks = split_document(document, chunk_size=1000, chunk_overlap=50)
    print(f"File {os.path.basename(file_path)} split into {len(chunks)} chunks.")
    add_chunks_to_qdrant(chunks, metadata["organization_id"], metadata["workspace_id"])
    # After indexing, notify the external service.
    notify_indexed_document(
        template_id=metadata["template_id"],
        filename=metadata["filename"],
        organization_id=metadata["organization_id"],
        workspace_id=metadata["workspace_id"]
    )

def main():
    # --- Extract ---
    docs = extract_documents()
    print(f"Extracted {len(docs)} document(s) from the database.")

    # --- Write to TXT Files ---
    file_data = write_txt_files(docs)
    print(f"Wrote {len(file_data)} TXT file(s) to disk.")

    # --- Process Files One by One ---
    for file_tuple in file_data:
        process_file(file_tuple)

    print("Completed loading all document chunks into Qdrant.")

if __name__ == "__main__":
    main()
