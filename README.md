# price-notify

mysql
CREATE DATABASE price_notifier;
CREATE USER 'price_notifier_user'@'%' IDENTIFIED BY 'e4HG3KF2S3f3_fd';
GRANT ALL PRIVILEGES ON price_notifier.* TO 'price_notifier_user'@'%';
FLUSH PRIVILEGES;

pip install -r requirements.txt

pip install -r requirements.txt --break-system-packages

python3 -m venv /var/www/price-notify/venv
source /var/www/price-notify/venv/bin/activate
pip install -r requirements.txt




sudo apt install tmux
tmux new -s crypto_analyzer

python3 main.py

tmux attach -t crypto_analyzer

ln -s ../render/ render



openssl req -x509 -newkey rsa:4096 -keyout server.key -out server.crt -days 365 -nodes -subj "/CN=localhost"