# OmniAgent Teardown (remove everything this project put on the host)

Everything project-scoped lives under `/opt/omniagent`. System-level additions
are listed explicitly below. Nothing else was installed by this project.

## 1. Stop services

```bash
cd /opt/omniagent/OmniAgent
docker compose down            # removes container + network (keeps volume data)
systemctl disable --now ollama # stop the Ollama service
```

## 2. Remove project state (one command)

```bash
rm -rf /opt/omniagent
```

This removes: the repo (including `.venv`), the Postgres data volume
(`docker-data/pg`), all Ollama models (`ollama-models/`), and the HuggingFace
caches (`hf-cache/`).

## 3. Remove system-level packages (optional)

```bash
curl -fsSL https://ollama.com/install.sh | sh   # (install was via this script)
apt-get remove --purge -y ollama                 # removes /usr/local/bin/ollama + service
rm -rf /etc/systemd/system/ollama.service.d      # our OLLAMA_MODELS override
rm -rf /usr/share/ollama                         # key dir created during setup
docker rmi pgvector/pgvector:pg16               # the DB image (~250 MB)
```

Docker Engine and containerd were already present or installed for this
project's stack; remove with `apt-get remove docker-* containerd*` only if
nothing else on the host uses them.

## 4. Remove the safety swap added at setup (optional)

```bash
swapoff /swapfile && rm /swapfile
sed -i '\#^/swapfile#d' /etc/fstab
```

## 5. Verify

```bash
ls /opt/omniagent 2>/dev/null || echo "project dir gone"
pgrep -a ollama || echo "ollama not running"
swapon --show
docker ps -a | grep omniagent || echo "no containers"
```
