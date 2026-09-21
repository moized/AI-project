import unittest
from rag.rag_pipeline import SimpleRAGPipeline

class TestRAGPipeline(unittest.TestCase):
    def setUp(self):
        self.rag = SimpleRAGPipeline(samples_dir="data/samples")

    def test_chunking_and_metadata(self):
        chunks = self.rag._load_and_chunk_documents()
        self.assertIsInstance(chunks, list)
        if chunks:
            self.assertIn("source", chunks[0])
            self.assertIn("text", chunks[0])
            self.assertIn("id", chunks[0])

    def test_semantic_retrieve(self):
        results = self.rag.retrieve("architecture", top_k=2)
        self.assertIsInstance(results, list)
        for res in results:
            self.assertIn("id", res)
            self.assertIn("text", res)
            self.assertIn("source", res)
            self.assertIn("score", res)
            self.assertIsInstance(res["score"], float)

    def test_empty_query(self):
        results = self.rag.retrieve("", top_k=2)
        self.assertEqual(len(results), 0)

    def test_irrelevant_query(self):
        results = self.rag.retrieve("xyzabcqwe123987654321", top_k=2)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 0)

if __name__ == "__main__":
    unittest.main()
