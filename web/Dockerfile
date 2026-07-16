# The web GUI service (spec_v2 §4): Next.js standalone output behind :3000.
# To build: docker build -t varagity-web ./web

# bun is the package manager only (install + running the build script);
# Node stays the runtime — the run stage below remains node:22.
FROM oven/bun:1.3.14-alpine AS base

# Dependencies resolve from the lockfile only (cache-friendly layer)
FROM base AS deps
WORKDIR /app
COPY package.json bun.lock ./
RUN bun install --frozen-lockfile

# Build with the browser-facing API origin inlined (NEXT_PUBLIC_* is a
# build-time constant — changing it means rebuilding this image).
FROM base AS build
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
RUN bun run build

# Runtime: only the standalone server + static assets, run as the
# unprivileged node user (chown so Next can write its runtime cache).
FROM node:22-alpine AS run
WORKDIR /app
ENV NODE_ENV=production
ENV HOSTNAME=0.0.0.0
ENV PORT=3000
COPY --from=build --chown=node:node /app/.next/standalone ./
COPY --from=build --chown=node:node /app/.next/static ./.next/static
COPY --from=build --chown=node:node /app/public ./public
USER node
EXPOSE 3000
CMD ["node", "server.js"]
