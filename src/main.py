from src.ingestion import IngestionPipeline
from src.retrieval import RetrievalSystem

def main():
    ingestion_pipeline = IngestionPipeline()
    retrieval_system = RetrievalSystem()

    # Example usage
    document = "Hello, this is a sample document."
    ingestion_pipeline.ingest_document(document)

    query = "sample"
    results = retrieval_system.retrieve_documents(query)
    print("Retrieved results:", results)

if __name__ == "__main__":
    main()
