# Context Profiler: API + UI with the JobBridge demo pre-profiled.
#   docker build -t ctxprof .
#   docker run --rm -p 8000:8000 ctxprof            # http://localhost:8000
# Mount your own database instead: -v "$PWD/ctxprof.db:/data/ctxprof.db"

FROM node:20-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --ignore-scripts
COPY web/ ./
RUN npm run build

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CTXPROF_WEB_DIST=/app/web
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir . && useradd --create-home ctxprof && mkdir /data && chown ctxprof /data
COPY --from=web /web/dist /app/web
USER ctxprof
WORKDIR /data
RUN ctxprof demo --out /data/demo --db /data/ctxprof.db
EXPOSE 8000
CMD ["ctxprof", "serve", "--host", "0.0.0.0", "--port", "8000", "--db", "/data/ctxprof.db"]
