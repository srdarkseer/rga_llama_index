from src.embeddings import EmbeddingManager
from src.database import DatabaseManager

class IngestionPipeline:
    def __init__(self):
        self.embedding_manager = EmbeddingManager()
        self.db_manager = DatabaseManager()

    def ingest_document(self, document):
        embedding = self.embedding_manager.get_embeddings(document)
        # Here you would have logic to insert the embedding into your database
        print("Document processed and stored:", document)
