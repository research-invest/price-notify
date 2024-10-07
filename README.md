# price-notify

apt install atop
apt remove fwupd

sudo apt install tmux
tmux new -s crypto_analyzer
tmux attach -t crypto_analyzer

sudo apt install mysql-server

mysql
CREATE DATABASE price_notifier;
CREATE USER 'price_notifier_user'@'%' IDENTIFIED BY 'e4HG3KF2S3f3_fd';
GRANT ALL PRIVILEGES ON price_notifier.* TO 'price_notifier_user'@'%';
FLUSH PRIVILEGES;

apt install python3-pip

pip install -r requirements.txt --break-system-packages

python3 main.py

pip install scipy --break-system-packages

cd html
ln -s ../render/ render