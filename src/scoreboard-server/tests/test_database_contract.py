from __future__ import annotations

import pytest

from scoreboard_server.db.connection import _postgres_placeholders


def test_postgres_placeholder_translation_ignores_literals_and_comments() -> None:
    sql = "SELECT '?' AS literal, value FROM runs WHERE a=? -- ?\nAND b=? /* ? */"
    assert _postgres_placeholders(sql) == (
        "SELECT '?' AS literal, value FROM runs WHERE a=$1 -- ?\nAND b=$2 /* ? */"
    )


@pytest.mark.parametrize("sql", ["SELECT 'unterminated", "SELECT 1 /* open"])
def test_postgres_placeholder_translation_rejects_unterminated_sql(sql: str) -> None:
    with pytest.raises(ValueError):
        _postgres_placeholders(sql)
