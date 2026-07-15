
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
