"""Test-session environment defaults.

Only sets environment defaults (before any backend module is imported) so
tests never start background network work or call a real LLM. No shared
fixtures live here on purpose: each test module owns its fixtures.
"""
import os

os.environ.setdefault("AI_COACH_AUTO_LEARN", "0")
os.environ.setdefault("GAMEDATA_AUTO_UPDATE", "0")
# An empty value also stops python-dotenv from injecting a developer's real
# key from .env (load_dotenv never overrides variables that already exist).
os.environ.setdefault("DEEPSEEK_API_KEY", "")
