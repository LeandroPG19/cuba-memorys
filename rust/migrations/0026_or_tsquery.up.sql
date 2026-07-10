-- 0026: OR-based tsquery for hybrid text search.
--
-- Problem: every text branch used `plainto_tsquery('simple', $1)`, which joins
-- lexemes with `&` (AND). A document had to contain *every* term to match, so
-- recall collapsed to zero as query length grew:
--
--     'launcher electron wine proton crash instalador NSIS'  -> 0 rows (AND)
--                                                            -> 49 rows (OR)
--
-- The vector branch could not rescue it either: `vector_search` returns an
-- empty list whenever the ONNX model is not loaded, so on a fallback embedder
-- the AND branches were the only live signal.
--
-- Fix: keep `plainto_tsquery` for normalization/lexing (it quotes every lexeme
-- and never emits phrase operators), then swap the `&` operators for `|`.
-- Ranking still rewards documents matching more/denser terms via ts_rank_cd,
-- so AND-strong documents keep floating to the top.
--
-- Injection safety: `plainto_tsquery` fully neutralizes the input before we
-- touch the text. Verified: the input
--     AT&T 'foo' | bar & baz <-> qux
-- serializes to `'at' & 't' & 'foo' & 'bar' & 'baz' & 'qux'` -- only the
-- operators we introduce are ever present, so ` & ` -> ` | ` cannot escape.
--
-- An input with no lexemes yields the empty tsquery, which matches nothing and
-- scores 0 under ts_rank -- the desired behavior for an empty query.

CREATE OR REPLACE FUNCTION cuba_or_tsquery(txt text)
RETURNS tsquery
LANGUAGE sql
IMMUTABLE
PARALLEL SAFE
RETURNS NULL ON NULL INPUT
AS $$
    SELECT replace(plainto_tsquery('simple', txt)::text, ' & ', ' | ')::tsquery
$$;

COMMENT ON FUNCTION cuba_or_tsquery(text) IS
    'Lexes text via plainto_tsquery then ORs the terms. Used by every full-text branch of cuba_faro so recall does not collapse on multi-term queries. See migration 0026.';
