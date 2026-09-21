import unittest
from agent.agent_core import ResearchAgent

class TestResearchAgent(unittest.TestCase):
    def setUp(self):
        self.agent = ResearchAgent()

    def test_agent_date_tool(self):
        result = self.agent.run("Bugün hangi tarih?")
        self.assertIsNotNone(result["tool_output"])
        self.assertIn("Mevcut tarih", result["tool_output"])

    def test_agent_rag_retrieval(self):
        result = self.agent.run("architecture")
        self.assertIsInstance(result["sources"], list)
        self.assertIn("answer", result)

if __name__ == "__main__":
    unittest.main()
