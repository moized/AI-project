from tools.tool_implementation import calculator, get_current_date


def test_calculator_valid():
    assert calculator("2 + 2") == "4"
    assert calculator("10 * 5") == "50"


def test_calculator_rejects_code():
    assert "Error" in calculator("__import__('os').system('whoami')")


def test_calculator_rejects_attributes():
    assert "Error" in calculator("().__class__.__base__")


def test_get_current_date():
    value = get_current_date()
    assert len(value) == 10
    assert value.count("-") == 2
