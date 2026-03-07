"""
Load API configuration from api_key.txt.
Format: api_key on line 1, base_url on line 2 (optional).
Lines starting with # and empty lines are ignored.
"""
import os


def _get_api_key_path() -> str:
    """Get the path to api_key.txt in the concept_gen directory."""
    return os.path.join(os.path.dirname(__file__), "api_key.txt")


def load_api_config(api_key_id: int = 0) -> dict:
    """
    Load API config from api_key.txt.
    Format: api_key on line 1, base_url on line 2 (optional).
    
    Returns:
        dict with 'api_key' (required) and optionally 'base_url'.
        base_url is omitted if not specified (uses OpenAI default).
    """
    path = _get_api_key_path()
    with open(path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip() and not l.strip().startswith("#")]
    
    if not lines:
        raise ValueError("api_key.txt is empty. Add your API key on line 1, optional base_url on line 2.")
    
    # Keys are lines that don't look like URLs; base_url is the first URL-like line
    api_keys = [l for l in lines if not l.startswith("http")]
    base_url_line = next((l for l in lines if l.startswith("http")), None)
    
    config = {"api_key": api_keys[api_key_id].strip()}
    if base_url_line:
        config["base_url"] = base_url_line.strip()
    
    return config


def load_api_keys() -> list:
    """Load API keys only (for llm_call.py compatibility)."""
    path = _get_api_key_path()
    with open(path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip() and not l.strip().startswith("#")]
    return [l for l in lines if not l.startswith("http")]
