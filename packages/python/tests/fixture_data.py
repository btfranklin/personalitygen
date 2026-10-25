from __future__ import annotations

import json
from pathlib import Path


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("Expected a JSON object")
    entries: dict[object, object] = value
    result: dict[str, object] = {}
    for key, item in entries.items():
        if not isinstance(key, str):
            raise TypeError("Expected a string key")
        result[key] = item
    return result


def _list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise TypeError("Expected a JSON array")
    items: list[object] = value
    return items


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Expected a JSON number")
    return float(value)


class Fixture:
    def __init__(self, value: object) -> None:
        self._data = _mapping(value)

    def value(self, key: str) -> object:
        return self._data[key]

    def mapping(self, key: str) -> Fixture:
        return Fixture(self.value(key))

    def objects(self, key: str) -> list[Fixture]:
        return [Fixture(value) for value in _list(self.value(key))]

    def number(self, key: str) -> float:
        return _number(self.value(key))

    def numbers(self, key: str) -> list[float]:
        return [_number(value) for value in _list(self.value(key))]

    def string(self, key: str) -> str:
        value = self.value(key)
        if not isinstance(value, str):
            raise TypeError("Expected a JSON string")
        return value

    def number_map(self, key: str) -> dict[str, float]:
        return self.mapping(key).number_values()

    def number_values(self) -> dict[str, float]:
        return {key: _number(value) for key, value in self._data.items()}

    def keys(self) -> list[str]:
        return list(self._data)

    def object_values(self) -> list[Fixture]:
        return [Fixture(value) for value in self._data.values()]


def load_fixture(path: Path) -> Fixture:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    return Fixture(value)
