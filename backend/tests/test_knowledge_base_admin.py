import unittest
from unittest.mock import patch

from app.rag.knowledge_base_admin import reindex_knowledge_base


class ReindexKnowledgeBaseTests(unittest.TestCase):
    @patch("app.rag.knowledge_base_admin.ingest_danske_spil_rules")
    def test_returns_reindex_summary(self, ingest_rules):
        ingest_rules.return_value = 329

        result = reindex_knowledge_base()

        ingest_rules.assert_called_once_with(recreate=True)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["chunks_indexed"], 329)
        self.assertIn("started_at", result)
        self.assertIn("completed_at", result)
        self.assertGreaterEqual(result["duration_seconds"], 0)

    @patch("app.rag.knowledge_base_admin.ingest_danske_spil_rules")
    def test_releases_lock_after_ingestion_error(self, ingest_rules):
        ingest_rules.side_effect = [RuntimeError("embedding failed"), 329]

        with self.assertRaisesRegex(RuntimeError, "embedding failed"):
            reindex_knowledge_base()

        result = reindex_knowledge_base()

        self.assertEqual(result["chunks_indexed"], 329)


if __name__ == "__main__":
    unittest.main()
