# GestureFlow web app: Next.js standalone server.
# Build context is the repository root (see docker-compose.yml); what it may read is listed in
# frontend.Dockerfile.dockerignore.

FROM node:22-alpine AS build
RUN corepack enable
WORKDIR /app

# The Edge-mode model comes from ../models in the repo; the setup script copies it into public/.
COPY models/gesture_cnn.onnx models/gesture_cnn.json /models/
ENV GESTUREFLOW_MODELS_DIR=/models

COPY frontend/package.json frontend/pnpm-lock.yaml frontend/pnpm-workspace.yaml ./
COPY frontend/scripts/ scripts/
# postinstall puts MediaPipe, ONNX Runtime Web and the model into public/
RUN pnpm install --frozen-lockfile

COPY frontend/ .
# Inlined into the browser bundle at build time: where the *browser* reaches the API.
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL \
    NEXT_TELEMETRY_DISABLED=1
RUN pnpm build

FROM node:22-alpine
WORKDIR /app
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    HOSTNAME=0.0.0.0 \
    PORT=3000

# The standalone bundle excludes static assets and public/; they are copied alongside it.
COPY --from=build --chown=node:node /app/.next/standalone ./
COPY --from=build --chown=node:node /app/.next/static ./.next/static
COPY --from=build --chown=node:node /app/public ./public

USER node
EXPOSE 3000
CMD ["node", "server.js"]
