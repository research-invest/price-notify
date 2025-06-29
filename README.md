# price-notify

apt install atop
apt remove fwupd

apt install tmux
tmux new -s crypto_analyzer
tmux attach -t crypto_analyzer

apt install mysql-server

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

supervisor


apt-get install supervisor

nano /etc/supervisor/conf.d/crypto_analyzer.conf

[program:crypto_analyzer]
directory=/var/www/price-notify
command=/usr/bin/python3 /var/www/price-notify/main.py
autostart=true
autorestart=true
stderr_logfile=/var/log/crypto_analyzer.err.log
stdout_logfile=/var/log/crypto_analyzer.out.log
user=root


supervisorctl reread
supervisorctl update
supervisorctl restart all
supervisorctl stop all

# Проверить статус
supervisorctl status crypto_analyzer

# Остановить скрипт
supervisorctl stop crypto_analyzer

# Запустить скрипт
supervisorctl start crypto_analyzer

# Перезапустить скрипт
supervisorctl restart crypto_analyzer


https://docs.ccxt.com/#/



killall core

