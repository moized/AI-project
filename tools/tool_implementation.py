from datetime import datetime

def calculator(expression: str) -> str:
    """Temel matematiksel ifadeleri güvenli bir şekilde değerlendirir."""
    try:
        allowed_chars = "0123456789+-*/()., \t"
        if not all(c in allowed_chars for c in expression):
            return "Hata: İfadenizde geçersiz karakterler var."
        
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Hata hesaplama sırasında oluştu: {str(e)}"

def get_current_date() -> str:
    """Mevcut tarihi döndürür."""
    return datetime.now().strftime("%Y-%m-%d")

TOOL_FUNCTIONS = {
    "calculator": calculator,
    "get_current_date": get_current_date
}
