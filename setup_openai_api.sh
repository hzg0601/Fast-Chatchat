# server
[ -e "./logs/" ] && "log dir exists" || mkdir "./logs/"
nohup python -m fastchat.serve.controller --host 0.0.0.0 > ./logs/controller.log 2>&1 &
wait $!
[ `ps -eo cmd|grep fastchat.serve.controller|grep -v -c grep` -ge 0 ] && echo "controller running" || echo "luanch controller failed"

# worker
nohup python -m fastchat.serve.model_worker --model-name 'chatglm-6b-int4' --model-path THUDM/chatglm-6b-int4 > ./logs/worker.log 2>&1 &
# grep -c --count不打印匹配的结果而打印匹配的行数
# https://blog.csdn.net/weixin_43772810/article/details/112059321
wait $!
[ `ps -eo cmd|grep fastchat.serve.model_worker|grep -v -c grep` -ge 0 ] && echo "worker running" || echo "luanch worker failed"

# webui
python -m fastchat.serve.openai_api_server --port 8001 --host 0.0.0.0