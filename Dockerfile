FROM python:3.14-slim AS builder

WORKDIR /app

# System deps for scipy/scikit-learn wheel builds
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-cache-dir .

# --- Production ---
FROM python:3.14-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.14/site-packages /usr/local/lib/python3.14/site-packages
COPY --from=builder /usr/local/bin/cuba-memorys /usr/local/bin/cuba-memorys
COPY --from=builder /app/src ./src

# Non-root user
RUN addgroup --system appgroup && adduser --system --group appuser
USER appuser

# DATABASE_URL must be provided at runtime
ENV DATABASE_URL=""

ENTRYPOINT ["cuba-memorys"]
