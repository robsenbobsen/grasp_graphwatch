#!/bin/sh
# Creates every database listed in POSTGRES_DATABASES (comma- or space-separated) that does
# not exist yet in doc-comp_postgres. Run by the one-off doc-comp_postgres_init service on
# each `docker compose --profile db up`; existing databases are left untouched.
set -eu

export PGHOST="${PGHOST:-doc-comp_postgres}" PGUSER="$POSTGRES_USER" PGPASSWORD="$POSTGRES_PASSWORD"

for db in $(echo "${POSTGRES_DATABASES:-}" | tr ',' ' '); do
  case "$db" in
    *[!A-Za-z0-9_]*)
      echo "invalid database name '$db' (letters, digits and _ only)" >&2
      exit 1
      ;;
  esac
  if [ "$(psql -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname = '$db'")" = "1" ]; then
    echo "exists:  $db"
  else
    psql -d postgres -q -c "CREATE DATABASE \"$db\""
    echo "created: $db"
  fi
done
