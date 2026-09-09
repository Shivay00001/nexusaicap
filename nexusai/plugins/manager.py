"""Plugin Manager — Discover, load, and manage NexusAI plugins."""
from __future__ import annotations
import importlib, json
from pathlib import Path
from nexusai.core.config import PLUGINS_DIR
from nexusai.core.exceptions import NexusAIError

class PluginBase:
    """Base class for NexusAI plugins."""
    name: str = "base_plugin"
    description: str = ""
    version: str = "1.0.0"
    def setup(self, config=None): pass
    def teardown(self): pass

class PluginManager:
    """Discovers, loads, and manages plugins."""
    def __init__(self):
        self._plugins: dict[str, PluginBase] = {}

    def load_builtin(self):
        """Load built-in plugins."""
        builtins = ["nexusai.plugins.builtin.web_search", "nexusai.plugins.builtin.code_interpreter", "nexusai.plugins.builtin.file_browser"]
        for mod_name in builtins:
            try:
                mod = importlib.import_module(mod_name)
                if hasattr(mod, "plugin"):
                    self._plugins[mod.plugin.name] = mod.plugin
            except ImportError:
                pass

    def load_from_directory(self, path: str = None):
        """Load plugins from a directory."""
        p = Path(path) if path else PLUGINS_DIR
        if not p.exists(): return
        for plugin_dir in p.iterdir():
            if plugin_dir.is_dir() and (plugin_dir / "plugin.json").exists():
                try:
                    meta = json.loads((plugin_dir / "plugin.json").read_text())
                    if (plugin_dir / "main.py").exists():
                        import sys; sys.path.insert(0, str(plugin_dir))
                        mod = importlib.import_module("main")
                        if hasattr(mod, "plugin"):
                            self._plugins[meta.get("name", plugin_dir.name)] = mod.plugin
                except Exception:
                    pass

    def get(self, name: str) -> PluginBase | None: return self._plugins.get(name)
    def list_plugins(self) -> list[dict]: return [{"name": p.name, "description": p.description, "version": p.version} for p in self._plugins.values()]
    def available(self) -> list[str]: return list(self._plugins.keys())
