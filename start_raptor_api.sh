LOG_DIR="/home/pjtl2w01admin/btw/fast-api/log-raptor"
LOG_FILE="$LOG_DIR/server.log"

# 로그 디렉터리가 존재하는지 확인
mkdir -p $LOG_DIR

# Gunicorn을 사용하여 서버 시작
nohup gunicorn main_raptor:raptor\
    --bind 0.0.0.0:8002 \
    --workers 3 \
    --worker-class uvicorn.workers.UvicornWorker \
    --timeout 2592010 \
    --log-level info \
    > $LOG_FILE 2>&1 &

echo "Started 3 workers on port 8002"