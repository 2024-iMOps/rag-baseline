cd /workspace
apt update
apt list --upgradable
apt install sudo
sudo apt install tree pkg-config

cd /workspace
curl -sSL https://install.python-poetry.org | python3 -
cd /workspace/rag-baseline
export PATH='/root/.local/bin'
poetry --version
poetry install