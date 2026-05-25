FROM node:20
WORKDIR /app

ARG VITE_API_BASE=/apis
ENV VITE_API_BASE=$VITE_API_BASE

# Copy and install dependencies
COPY package*.json ./
RUN npm install

# Copy source and build
COPY . .
RUN npm run build

# Install a simple static server
RUN npm install -g serve

EXPOSE 5173
CMD ["serve", "-s", "dist", "-l", "5173"]
