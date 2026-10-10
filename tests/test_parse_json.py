from __future__ import annotations

from app.llm.client import parse_json


def test_plain_json():
    assert parse_json('{"a": 1}') == {"a": 1}


def test_fenced_json():
    assert parse_json('```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}


def test_json_with_prose():
    assert parse_json('Sure! Here it is: {"a": true} hope that helps') == {"a": True}


def test_no_json_returns_none():
    assert parse_json("no json here at all") is None
