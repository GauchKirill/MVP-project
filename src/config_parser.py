import json
from types import SimpleNamespace

class ConfigParser:
    def __init__(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self._data = self._to_namespace(data)

    @staticmethod
    def _to_namespace(obj):
        if isinstance(obj, dict):
            return SimpleNamespace(**{k: ConfigParser._to_namespace(v) for k, v in obj.items()})
        elif isinstance(obj, list):
            return [ConfigParser._to_namespace(item) for item in obj]
        return obj

    def get(self, key, default=None):
        return getattr(self._data, key, default)

    def __getattr__(self, name):
        return getattr(self._data, name)
