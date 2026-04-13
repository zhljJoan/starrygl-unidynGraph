#!/bin/bash

cd /home/zlj/StarryUniGraph/docs
make html

mkdir -p /home/zlj/StarryUniGraph/docs/logs

# 启动 http.server
tmux new -d -s docs 'cd /home/zlj/StarryUniGraph/docs/build/html && python -m http.server 8080'

# 启动 ngrok
tmux new -d -s ngrok '/home/zlj/local/ngrok http 8080'
#tmux attach -t ngrok
# 等待 ngrok 启动
# for i in $(seq 1 10); do
#     sleep 2
#     URL=$(curl -s http://127.0.0.1:4040/api/tunnels | python -c "
# import sys, json
# try:
#     data = json.load(sys.stdin)
#     print(data['tunnels'][0]['public_url'])
# except:
#     pass
# " 2>/dev/null)
#     if [ -n "$URL" ]; then
#         echo "公网地址: $URL"
#         exit 0
#     fi
# done
# echo "ngrok 启动超时，请手动查看: tmux attach -t ngrok"
