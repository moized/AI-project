import unittest
from rag.rag_pipeline import SimpleRAGPipeline

class TestRAGPipeline(unittest.TestCase):
    def setUp(self):
        self.rag = SimpleRAGPipeline(samples_dir="data/samples")

    def test_chunking(self):
        self.assertIsInstance(self.rag.chunks, list)

    def test_retrieve(self):
        results = self.rag.retrieve("architecture", top_k=2)
        self.assertIsInstance(results, list)

if __name__ == "__main__":
    unittest.main()
