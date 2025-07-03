# 기본 로그 디렉터리 설정
BASE_LOG_DIR="/home/pjtl2w01admin/btw/fast-api/log"

# 서버 로그 디렉터리 및 파일 설정
SERVER_LOG_DIR="$BASE_LOG_DIR/server"
SERVER_LOG_FILE="$SERVER_LOG_DIR/server.log"
SERVER_ERROR_LOG_FILE="$SERVER_LOG_DIR/server_error.log"

# 필요한 로그 디렉터리 생성
mkdir -p "$SERVER_LOG_DIR"
mkdir -p "$BASE_LOG_DIR/models/embedding"
mkdir -p "$BASE_LOG_DIR/models/segmentation"
mkdir -p "$BASE_LOG_DIR/models/summarize"
mkdir -p "$BASE_LOG_DIR/extraction/hwp_to_text"
mkdir -p "$BASE_LOG_DIR/extraction/image_ocr"
mkdir -p "$BASE_LOG_DIR/extraction/pdf_to_text"
mkdir -p "$BASE_LOG_DIR/converter/docx_to_pdf"
mkdir -p "$BASE_LOG_DIR/converter/excel_to_csv"
mkdir -p "$BASE_LOG_DIR/converter/excel_to_pdf"
mkdir -p "$BASE_LOG_DIR/converter/image_to_pdf"
mkdir -p "$BASE_LOG_DIR/converter/ppt_to_pdf"
mkdir -p "$BASE_LOG_DIR/converter/hwp_to_xhtml"
mkdir -p "$BASE_LOG_DIR/converter/xhtml_to_pdf"

# 로그 파일들에 대한 권한 설정
find "$BASE_LOG_DIR" -type d -exec chmod 755 {} \;
find "$BASE_LOG_DIR" -type f -exec chmod 644 {} \;

# Gunicorn 서버 시작
nohup gunicorn main:app -w 6 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8001 --timeout 1210 --error-logfile "$SERVER_ERROR_LOG_FILE" --log-level info > "$SERVER_LOG_FILE" 2>&1 &
echo "Started 8 workers on port 8001"