"""Tests for db.py — serialization and helper functions.

Tests the pure-function parts of db.py that don't require
a database connection.
"""
import pytest
from datetime import datetime, date
from decimal import Decimal
from uuid import UUID, uuid4

from cuba_memorys.db import _json_default, serialize, _DB_NAME_PATTERN, has_pgvector
import cuba_memorys.db


class TestHasPgVector:
    """Tests for has_pgvector() flag."""

    def test_has_pgvector_toggle(self) -> None:
        """🟢 has_pgvector() correctly reports the state of the global flag."""
        # Save original state
        original = cuba_memorys.db._pgvector_available
        try:
            cuba_memorys.db._pgvector_available = True
            assert has_pgvector() is True

            cuba_memorys.db._pgvector_available = False
            assert has_pgvector() is False
        finally:
            # Restore original state
            cuba_memorys.db._pgvector_available = original


class TestJsonDefault:
    """Tests for the custom JSON serializer."""

    def test_uuid_serialization(self) -> None:
        """🟢 UUIDs serialize to string."""
        uid = uuid4()
        result = _json_default(uid)
        assert result == str(uid)

    def test_datetime_serialization(self) -> None:
        """🟢 datetime serializes to ISO format."""
        dt = datetime(2026, 3, 8, 12, 0, 0)
        result = _json_default(dt)
        assert result == "2026-03-08T12:00:00"

    def test_date_serialization(self) -> None:
        """🟢 date serializes to ISO format."""
        d = date(2026, 3, 8)
        result = _json_default(d)
        assert result == "2026-03-08"

    def test_decimal_serialization(self) -> None:
        """🟢 Decimal serializes to float."""
        result = _json_default(Decimal("3.14"))
        assert result == pytest.approx(3.14)

    def test_unsupported_type_raises(self) -> None:
        """🟡 Unsupported types raise TypeError."""
        with pytest.raises(TypeError, match="not JSON serializable"):
            _json_default(set())

    def test_uuid_roundtrip(self) -> None:
        """🟢 UUID → string → UUID roundtrip."""
        uid = uuid4()
        serialized = _json_default(uid)
        assert UUID(serialized) == uid


class TestSerialize:
    """Tests for the serialize() function (orjson)."""

    def test_basic_dict(self) -> None:
        """🟢 Simple dict serializes to JSON string."""
        result = serialize({"key": "value"})
        assert '"key"' in result
        assert '"value"' in result

    def test_nested_dict(self) -> None:
        """🟢 Nested structures work."""
        result = serialize({"a": {"b": [1, 2, 3]}})
        assert '"a"' in result
        assert "[1,2,3]" in result

    def test_uuid_in_dict(self) -> None:
        """🟢 UUID values in dicts serialize."""
        uid = uuid4()
        result = serialize({"id": uid})
        assert str(uid) in result

    def test_datetime_in_dict(self) -> None:
        """🟢 datetime values serialize."""
        dt = datetime(2026, 1, 1)
        result = serialize({"ts": dt})
        assert "2026" in result

    def test_none_value(self) -> None:
        """🟡 None serializes to null."""
        result = serialize({"val": None})
        assert "null" in result

    def test_empty_dict(self) -> None:
        """🟡 Empty dict."""
        assert serialize({}) == "{}"

    def test_non_str_keys(self) -> None:
        """🟢 Non-string keys are handled (OPT_NON_STR_KEYS)."""
        result = serialize({1: "one", 2: "two"})
        assert '"1"' in result or "1" in result

    def test_returns_string(self) -> None:
        """🟢 Always returns a str, not bytes."""
        result = serialize({"test": True})
        assert isinstance(result, str)


class TestDBNamePattern:
    """Tests for the database name validation regex."""

    def test_valid_names(self) -> None:
        """🟢 Valid PostgreSQL identifiers match."""
        valid = ["brain", "test_db", "my_database_2", "_private", "A_mixed"]
        for name in valid:
            assert _DB_NAME_PATTERN.match(name) is not None, f"'{name}' should match"

    def test_invalid_names(self) -> None:
        """🔴 SQL injection attempts don't match."""
        invalid = [
            "0starts_with_digit",
            "has spaces",
            "'; DROP TABLE --",
            "../../../etc/passwd",
            "",
            "x" * 100,  # Too long (max 63)
        ]
        for name in invalid:
            assert _DB_NAME_PATTERN.match(name) is None, f"'{name}' should NOT match"

    def test_max_length_63(self) -> None:
        """🟢 PostgreSQL identifier max is 63 chars."""
        name_63 = "a" * 63
        name_64 = "a" * 64
        assert _DB_NAME_PATTERN.match(name_63) is not None
        assert _DB_NAME_PATTERN.match(name_64) is None
