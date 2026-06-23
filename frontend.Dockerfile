FROM node:20-slim

WORKDIR /app

COPY frontend/package*.json ./
RUN npm ci || npm install

COPY frontend/ .

# NEXT_PUBLIC_* values are inlined at build time, so the backend URL must be
# provided here (not just at runtime). Defaults to localhost for local compose;
# pass --build-arg NEXT_PUBLIC_API_URL=https://api.example.com for a real deploy.
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

EXPOSE 3000

CMD ["npm", "run", "start"]
