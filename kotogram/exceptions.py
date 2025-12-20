class KotogramError(Exception):
    """Base exception for Kotogram errors."""
    pass

class MissingMappingError(KotogramError, KeyError):
    """Raised when a feature mapping is missing during parsing.
    
    Inherits from KeyError for backward compatibility.
    """
    def __init__(self, map_name: str, key: str, context: str = ""):
        self.map_name = map_name
        self.key = key
        self.context = context
        super().__init__(f"Missing mapping in {map_name}: key='{key}' not found. {context}")
