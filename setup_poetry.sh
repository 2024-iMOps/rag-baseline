## Terminal
# 실행 권한 부여: chmod +x setup_poetry.sh
# 실행: ./setup_poetry.sh

## git
echo "Input github repo:"
read GIT_URL
git clone $GIT_URL

## .env에 API Key 작성
cd /workspace/rag-baseline
if [ ! -f ".env" ]; then
  touch .env
fi

echo "Input API name:"
read API_NAME

echo "Input API key:"
read API_KEY

echo "${API_NAME}=\"${API_KEY}\"" > .env

## install sudo and front
cd /workspace
apt update
apt list --upgradable
apt install sudo
sudo apt install tree pkg-config
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs
npm install @mui/material @emotion/react @emotion/styled @mui/icons-material react-router-dom axios react-pdf

## Poetry 설치 및 환경 구성
cd /workspace
curl -sSL https://install.python-poetry.org | python3 -
cd /workspace/rag-baseline
export PATH='/root/.local/bin'
poetry --version
poetry install

## python interpreter는 poetry로 변경