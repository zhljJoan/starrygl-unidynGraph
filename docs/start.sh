#!/bin/bash  
make html\
cd /home/zlj/StarryUniGraph/docs/build/html  \
nohup python -m http.server 8080 > /home/zlj/StarryUniGraph/docs/logs/log.out &\
nohup ~/local/ngrok http 8080' > /home/zlj/StarryUniGraph/docs/logs/grok.out &\

