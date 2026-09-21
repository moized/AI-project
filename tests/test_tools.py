import unittest
from tools.tool_implementation import calculator, get_current_date

class TestTools(unittest.TestCase):
    def test_calculator_valid(self):
        self.assertEqual(calculator("2 + 2"), "4")
        self.assertEqual(calculator("10 * 5"), "50")

    def test_calculator_invalid_chars(self):
        result = calculator("import os")
        self.assertIn("Hata", result)

    def test_get_current_date(self):
        date_str = get_current_date()
        self.assertEqual(len(date_str), 10)
        self.assertEqual(date_str.count("-"), 2)

if __name__ == '__main__':
    unittest.main()
