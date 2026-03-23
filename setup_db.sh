#!/usr/bin/env bash
# setup_db.sh — Create the 'brain' database for Cuba-Memorys
# Run this once on the PostgreSQL server.
# Uses the same credentials as postgres_mcp_launcher.sh
set -euo pipefail

DB_USER="${DB_USER:-usuario}"
DB_PASS="${DB_PASS:-}"
DB_HOST="${1:-${DB_HOST:-localhost}}"
DB_PORT="${2:-${DB_PORT:-6433}}"

if [[ -z "$DB_PASS" ]]; then
  echo "❌ Error: DB_PASS environment variable is not set."
  echo "Usage: DB_PASS=yourpassword $0 [host] [port]"
  exit 1
fi

echo "🧠 Creating 'brain' database on ${DB_HOST}:${DB_PORT}..."

PGPASSWORD="$DB_PASS" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -c \
  "SELECT 'exists' FROM pg_database WHERE datname = 'brain'" | grep -q exists && {
    echo "✅ Database 'brain' already exists"
} || {
    PGPASSWORD="$DB_PASS" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -c \
      "CREATE DATABASE brain"
    echo "✅ Database 'brain' created"
}

echo "🔧 Initializing schema..."
PGPASSWORD="$DB_PASS" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d brain -f \
  "$(dirname "$0")/src/cuba_memorys/schema.sql"

echo "✅ Cuba-Memorys database ready!"
echo "   Connection: postgresql://${DB_USER}:****@${DB_HOST}:${DB_PORT}/brain"
