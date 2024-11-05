#!/bin/bash

# Instalar o Docker
apt-get update
apt-get -y install git vim
git clone https://github.com/joaomarceloalencar/devops
cd devops/Laboratorios/02_Docker/
sed -i 's/\$SUDO_USER/ubuntu/g' install_docker.sh
./install_docker.sh

# Instalar Ollama
apt-get update
apt-get -y dist-upgrade
mkdir /ollama
mkdir /openwebui
cat <<EOF > /docker-composer.yml
services:
  ollama:
    volumes:
      - ollama:/ollama
    container_name: ollama
    pull_policy: always
    tty: true
    restart: unless-stopped
    image: ollama/ollama:latest

  open-webui:
    image: ghcr.io/open-webui/open-webui:latest
    container_name: open-webui
    volumes:
      - open-webui:/app/backend/data
    depends_on:
      - ollama
    ports:
      - 80:8080
    environment:
      - 'OLLAMA_BASE_URL=http://ollama:11434'
      - 'WEBUI_SECRET_KEY=15ufc66quixada'
    extra_hosts:
      - host.docker.internal:host-gateway
    restart: unless-stopped

volumes:
  ollama:
    driver: local
    driver_opts:
      type: none
      device: /ollama
      o: bind
  open-webui:
    driver: local
    driver_opts:
      type: none
      device: /openwebui
      o: bind
EOF
docker compose up -d
