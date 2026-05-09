from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply the two given integers
    a: integer
    b: integer
    
    Result in integer
    """
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Add the two given integers
    a: integer
    b: integer
    
    Result in integer
    """
    
    return a + b