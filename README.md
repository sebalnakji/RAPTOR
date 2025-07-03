# datalake-fast-api

## 0. 개발 환경

[환경]

- Ubuntu 22.04

[버전]

- Python: 3.9
- CUDA: 11.8
- pytorch: 2.3

[소스코드 위치]

- /home/pjtl2w01admin/btw/fast-api

[테스트 주소]

## 1. embeding_api

[API]

- bge-m3: 
- MiniLM: 
- KR-SBERT: 

[bge-m3]
- Hyper Parameter
  1. batch size: 한 번에 모델이 학습하는 문장의 개수
  2. max_length: 처리 가능한 최대 토큰 수 (1~8192)
- API
  1. batch size: 12 (default)
  2. max_length: 500
- segmentation model의 max_length 조절 필요

[MiniLM]
- max_length: 128
- segmentation model의 max_length 조절 필요

[KR-SBERT]
- max_length: 128
- segmentation model의 max_length 조절 필요

## 2. summarize_api

[API]
- Llama3: 
- T5: 

[Llama3]
- segCount: 요약할 문장 목록을 분리할 문단 수 (-1일 경우 최적 문단 수로 분리)
- segMaxSize: 한 문단에 포함될 최대 문자열의 글자 수 (1~3000)
- segMinSize: 한 문단에 포함될 최소 문자열의 글자 수 (0~segMaxSize)

[T5]
- segMaxSize: 한 문단에 포함될 최대 문자열의 글자 수 (default:100, 1~3000)
- segMinSize: 한 문단에 포함될 최소 문자열의 글자 수 (0~segMaxSize)

## 3. segmentation_api

[API]
- Llama3: 
- nltk: 

[Llama3]
- segCount: 요약할 문장 목록을 분리할 문단 수 (-1일 경우 최적 문단 수로 분리)
- segMaxSize: 한 문단에 포함될 최대 문자열의 글자 수 (1~3000)
- segMinSize: 한 문단에 포함될 최소 문자열의 글자 수 (0~segMaxSize)

[nltk]
- chunk_length: 한 문단에 포함될 최대 토큰 수 (default:100, 1~1000)
- whether_fix_sentence: 완전한 문장으로 분할할 것인지 여부(True/False)

## 4. pdf-to-text_api

[API]
- pdf-to-text: 

[pdf-to-text]
- PDF의 이미지와 표의 프레임을 제외한 모든 텍스트 추출

## 5. RAPTOR

[API]
- RAPTOR: 

[RAPTOR]
- databaseName: Milvus 데이터베이스명(워크스페이스 관리ID)\n
- summarizeModel: 요약 모델명\n
- embeddingModel: 임베딩 모델명\n
- targetVectorTables: RAPTOR 대상 벡터테이블 목록\n
  - vectorTableName: 벡터 테이블명(컬렉션명)\n
  - vectorColumnName: 벡터 컬럼명\n
  - textColumnName: 벡터화 되기전 텍스트컬럼명\n
  - vectorTableIdx: 벡터테이블 일련번호(datalake DB 일련번호)\n
- newVectorTableName: RAPTOR 처리 후 결과 벡터 테이블명\n
- workerId: 요약 모델 할당 ID

## 배포 방법

# 1. Main

[실행 확인]
- `lsof -i :8001`

[실행 중지]
- `lsof -i TCP:8001 -t | xargs kill`

[캐시 삭제] (반드시 필요한 과정 x)
1. fast-api 파일 경로 이동
- `cd /home/pjtl2w01admin/btw/fast-api`
2. 캐시 삭제
- `find . -name "__pycache__" -exec rm -r {} +`

[서버 배포]
1. 가상환경 활성화
- `conda activate api`
2. fast-api 파일 경로 이동
- `cd /home/pjtl2w01admin/btw/fast-api`
3. 서버 배포
- `./start_api.sh`

# 2. RAPTOR

[실행 확인]
- `lsof -i :8002`

[실행 중지]
- `lsof -i TCP:8002 -t | xargs kill`

[서버 배포]
1. 가상환경 활성화
- `conda activate api`
2. fast-api 파일 경로 이동
- `cd /home/pjtl2w01admin/btw/fast-api`
3. 서버 배포
- `./start_raptor_api.sh`

## 수정사항

**[24-06-14] - btw**
- Input data를 List 형식으로 변경

**[24-06-17] - btw**
- Input Parameter를 List 형식에서 단일화
- segmentation Llama3 모델의 span 출력 오류 수정
- summarize Llama3 모델의 output_list 출력 오류 수정

**[24-06-20] - btw**
- summarize 모델 Response body 수정
- summarize 모델 Docs 규격 변경
- summarize, segmentation 모델 규격 외 파라미터 입력 시 오류 반환

**[24-06-21] - btw**
- summarize T5 모델 GPU 사용
- summarize, segmentation 모델 파라미터 변경 및 코드 최적화
- WSGI로 gunicorn 사용 및 worker 수 2개로 변경

**[24-06-25] - btw**
- segmentation nltk 모델 chunk_length 수정 (500 -> 1000)

**[24-06-28] - bjm**
- segmentation nltk 모델 입력 수정
- 문장의 길이보다 chunk_length가 큰 경우 오류 반환

**[24-07-03] - btw**
- embedding bge-m3 모델의 max_length 수정 (500 -> 1000)
- embedding 모델 규격 외 토큰 수 입력 시 오류 반환

**[24-07-04] - btw**
- embedding Minilm 모델의 max_length 수정 (256 -> 128)
- embedding, summarize, segmentation 모델 로깅 추가

**[24-07-05] - btw**
- Llama3-7b 모델 Databricks Llama3-70b 모델 API로 교체
- Segmentation, Summarize Llama3 모델 segCount (0 -> -1), segMaxSize (0 -> 100) default값 변경
- timeout (30 -> 300), worker (2 -> 8) 수 변경, 최대 worker 수 (4 -> 16)

**[24-07-08] - btw**
- timeout 로깅 추가
- 로그 정리 추가(daily, 7days, 00:02 restart)

**[24-07-08] - bjm**
- NLTK overlap 기능 추가

**[24-07-22] - btw**
- RAPTOR 추가

**[24-07-22] - bjm**
- HWP to text 추가

**[24-07-22] - hjs**
- image ocr 추가

**[24-07-22] - btw**
- RAPTOR Port 분리