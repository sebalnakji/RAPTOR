import os
import umap
import time
import asyncio
import logging
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from FlagEmbedding import BGEM3FlagModel
from typing import List, Tuple, Optional
from pymilvus import MilvusClient, DataType
from sklearn.mixture import GaussianMixture
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer

# httpx 로거의 레벨을 WARNING으로 설정
logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 임베딩 모델 로드
embedding_model_bge = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, device='cuda:0')
embedding_model_krsbert = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS', device='cuda:0')
embedding_model_minilm = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device='cuda:0')

# 요약 모델 클라이언트 풀
summarize_clients = {
    "Llama3-1": None,  
    "Llama3-2": None,  
    "Llama3-3": None,
}

# 각 모델에 대한 세마포어
summarize_semaphores = {
    "Llama3-1": asyncio.Semaphore(1),
    "Llama3-2": asyncio.Semaphore(1),
    "Llama3-3": asyncio.Semaphore(1),
}

# 스레드 풀
thread_pool = ThreadPoolExecutor(max_workers=3)

# 각 모델별 설정
model_configs = {
    "Llama3-1": {
        "api_key": os.getenv('DATABRICKS_TOKEN_1'),
        "base_url": os.getenv('DATABRICKS_ENDPOINT_1'),
        "model_name": "databricks-meta-llama-3-70b-instruct"
    },
    "Llama3-2": {
        "api_key": os.getenv('DATABRICKS_TOKEN_2'),
        "base_url": os.getenv('DATABRICKS_ENDPOINT_2'),
        "model_name": "databricks-meta-llama-3-70b-instruct"
    },
    "Llama3-3": {
        "api_key": os.getenv('DATABRICKS_TOKEN_3'),
        "base_url": os.getenv('DATABRICKS_ENDPOINT_3'),
        "model_name": "databricks-meta-llama-3-70b-instruct"
    }
}

# 워커 ID에 따라 모델 선택
async def get_summarize_client(worker_id):
    model_name = f"Llama3-{worker_id}"
    semaphore = summarize_semaphores[model_name]
                     
    async with semaphore:
        if summarize_clients[model_name] is None:
            # 클라이언트가 없으면 새로 생성
            summarize_clients[model_name] = create_summarize_client(model_name)
        return summarize_clients[model_name], model_configs[model_name]['model_name']

# 요약 모델 클라이언트 생성
def create_summarize_client(model_name):
    config = model_configs[model_name]
    client = OpenAI(
        api_key=config['api_key'],
        base_url=config['base_url']
    )
    return client

# Llama3-70B-Instruct 요약 모델
async def Llama3_summarize(text, worker_id):
    client, model_name = await get_summarize_client(worker_id)
    
    # 스레드 풀을 사용하여 비동기적으로 요약 실행
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        thread_pool,
        lambda: client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": """
             You are a summarization model. Please follow the instructions below to create a summary.
  
             1. Provide only the summary content without including unnecessary phrases.
             2. Do not include phrases like 'Here is a summary of the content:' or line breaks ('\n').
             3. Respond consistently in the same language as the input. If there is any Korean in the input, summarize the entire content in Korean.
             4. Ensure the summary is a complete sentence ending with a period, and it must be within 500 tokens.
             5. If the sentence does not end with a period, discard the content and try again.
             """
             },
            {"role": "user", "content": f"""
             Please summarize the following content, including as many key details as possible:
             {text}
             """}
            ],
            temperature=0.1,
            max_tokens=700
        )
    )
    
    return response.choices[0].message.content

# 텍스트 병합
def combined_texts(table_info, client):
    combined_dict = {"vector": [], "text": []}
    
    table_name = table_info.vectorTableName
    vector_column_name = table_info.vectorColumnName
    text_column_name = table_info.textColumnName
    
    res_count = client.query(
        collection_name=table_name,
        output_fields=["count(*)"]
    )
    total_count = res_count[0]["count(*)"]

    offset = 0
    limit = 16384

    while offset < total_count:
        res = client.query(
            collection_name=table_name,
            offset=offset,
            limit=limit,
            output_fields=[vector_column_name, text_column_name]
        )
        
        combined_dict["vector"].extend(item[vector_column_name] for item in res)
        combined_dict["text"].extend(item[text_column_name].replace('\n', '') for item in res)
        
        offset += limit
        time.sleep(0.5)
        
    logger.info(f"Number of rows of source data: {len(combined_dict['text'])}")

    return combined_dict

# 전역 차원 축소
def global_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: Optional[int] = None,
    metric: str = "cosine",
) -> np.ndarray:
    """
    UMAP을 사용하여 임베딩의 전역 차원 축소를 수행합니다.

    매개변수:
    - embeddings: numpy 배열로 된 입력 임베딩.
    - dim: 축소된 공간의 목표 차원.
    - n_neighbors: 선택 사항; 각 점을 고려할 이웃의 수.
                   제공되지 않으면 임베딩 수의 제곱근으로 기본 설정됩니다.
    - metric: UMAP에 사용할 거리 측정 기준.

    반환값:
    - 지정된 차원으로 축소된 임베딩의 numpy 배열.
    """
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    return umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)

# 지역 차원 축소
def local_cluster_embeddings(
    embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine"
) -> np.ndarray:
    """
    임베딩에 대해 지역 차원 축소를 수행합니다. 이는 일반적으로 전역 클러스터링 이후에 사용됩니다.

    매개변수:
    - embeddings: numpy 배열로서의 입력 임베딩.
    - dim: 축소된 공간의 목표 차원 수.
    - num_neighbors: 각 점에 대해 고려할 이웃의 수.
    - metric: UMAP에 사용할 거리 측정 기준.

    반환값:
    - 지정된 차원으로 축소된 임베딩의 numpy 배열.
    """
    return umap.UMAP(
        n_neighbors=num_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)

# 최적의 클러스터 수 결정
def get_optimal_clusters(
    embeddings: np.ndarray, max_clusters: int = 50
) -> int:
    """
    가우시안 혼합 모델(Gaussian Mixture Model)을 사용하여 베이지안 정보 기준(BIC)을 통해 최적의 클러스터 수를 결정합니다.

    매개변수:
    - embeddings: numpy 배열로서의 입력 임베딩.
    - max_clusters: 고려할 최대 클러스터 수.

    반환값:
    - 발견된 최적의 클러스터 수를 나타내는 정수.
    """
    max_clusters = min(
        max_clusters, len(embeddings)
    )  # 최대 클러스터 수와 임베딩의 길이 중 작은 값을 최대 클러스터 수로 설정
    n_clusters = np.arange(1, max_clusters)  # 1부터 최대 클러스터 수까지의 범위를 생성
    bics = []  # BIC 점수를 저장할 리스트
    for n in n_clusters:  # 각 클러스터 수에 대해 반복
        gm = GaussianMixture(
            n_components=n
        )  # 가우시안 혼합 모델 초기화
        gm.fit(embeddings)  # 임베딩에 대해 모델 학습
        bics.append(gm.bic(embeddings))  # 학습된 모델의 BIC 점수를 리스트에 추가
    return n_clusters[np.argmin(bics)]  # BIC 점수가 가장 낮은 클러스터 수를 반환

# 클러스터링
def GMM_cluster(embeddings: np.ndarray, threshold: float):
    """
    확률 임계값을 기반으로 가우시안 혼합 모델(GMM)을 사용하여 임베딩을 클러스터링합니다.

    매개변수:
    - embeddings: numpy 배열로서의 입력 임베딩.
    - threshold: 임베딩을 클러스터에 할당하기 위한 확률 임계값.

    반환값:
    - 클러스터 레이블과 결정된 클러스터 수, GMM 모델을 포함하는 튜플.
    """
    n_clusters = get_optimal_clusters(embeddings)  # 최적의 클러스터 수를 구합니다.
    # 가우시안 혼합 모델을 초기화합니다.
    gm = GaussianMixture(n_components=n_clusters)
    gm.fit(embeddings)  # 임베딩에 대해 모델을 학습합니다.
    probs = gm.predict_proba(embeddings)  # 임베딩이 각 클러스터에 속할 확률을 예측합니다.
    # 임계값을 초과하는 확률을 가진 클러스터를 레이블로 선택합니다.
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters, gm  # 레이블, 클러스터 수, GMM 모델을 반환합니다.

# 차원 축소, 클러스터링, 각 전역 클러스터 내에서의 지역 클러스터링을 순서대로 수행
def perform_clustering(
    embeddings: np.ndarray,
    dim: int,
    threshold: float,
    n_top: int = 10
) -> List[np.ndarray]:
    """
    임베딩에 대해 차원 축소, 가우시안 혼합 모델을 사용한 클러스터링, 각 글로벌 클러스터 내에서의 로컬 클러스터링을 순서대로 수행합니다.

    매개변수:
    - embeddings: numpy 배열로 된 입력 임베딩입니다.
    - dim: UMAP 축소를 위한 목표 차원입니다.
    - threshold: GMM에서 임베딩을 클러스터에 할당하기 위한 확률 임계값입니다.
    - n_top: 각 로컬 클러스터에서 고려할 상위 임베딩의 수입니다.

    반환값:
    - 각 임베딩의 클러스터 ID를 포함하는 numpy 배열의 리스트입니다.
    """
    if len(embeddings) <= dim + 1:
        # 데이터가 충분하지 않을 때 클러스터링을 피합니다.
        return [np.array([0]) for _ in range(len(embeddings))]

    # 글로벌 차원 축소
    reduced_embeddings_global = global_cluster_embeddings(embeddings, dim)
    # 글로벌 클러스터링
    global_clusters, n_global_clusters, gm_global = GMM_cluster(
        reduced_embeddings_global, threshold
    )

    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0

    # 각 글로벌 클러스터를 순회하며 로컬 클러스터링 수행
    for i in range(n_global_clusters):
        # 현재 글로벌 클러스터에 속하는 임베딩 추출
        global_cluster_embeddings_ = embeddings[
            np.array([i in gc for gc in global_clusters])
        ]

        if len(global_cluster_embeddings_) == 0:
            continue
        if len(global_cluster_embeddings_) <= dim + 1:
            # 작은 클러스터는 직접 할당으로 처리
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
            gm_local = None
        else:
            # 로컬 차원 축소 및 클러스터링
            reduced_embeddings_local = local_cluster_embeddings(
                global_cluster_embeddings_, dim
            )
            local_clusters, n_local_clusters, gm_local = GMM_cluster(
                reduced_embeddings_local, threshold
            )

        # 로컬 클러스터 ID 할당, 이미 처리된 총 클러스터 수를 조정
        for j in range(n_local_clusters):
            if gm_local is not None:
                local_cluster_probs = gm_local.predict_proba(reduced_embeddings_local)[:, j]
            else:
                local_cluster_probs = np.ones(len(global_cluster_embeddings_))
            
            sorted_indices = np.argsort(local_cluster_probs)[::-1]
            top_indices = sorted_indices[:n_top]
            
            for idx in top_indices:
                if local_cluster_probs[idx] > threshold:
                    global_idx = np.where((embeddings == global_cluster_embeddings_[idx]).all(axis=1))[0][0]
                    all_local_clusters[global_idx] = np.append(
                        all_local_clusters[global_idx], j + total_clusters
                    )

        total_clusters += n_local_clusters

    return all_local_clusters

# 임베딩 모델 정의
async def embed(texts, EmbeddingModel):
    if EmbeddingModel == 'bge':
        embedding = await asyncio.to_thread(embedding_model_bge.encode, texts)
        embeddings = embedding['dense_vecs'].tolist()
        
    elif EmbeddingModel == 'minilm':
        embeddings = await asyncio.to_thread(embedding_model_minilm.encode, texts)
        
    elif EmbeddingModel == 'krsbert':
        embeddings = await asyncio.to_thread(embedding_model_krsbert.encode, texts)
    
    return embeddings

# 임베딩을 numpy 배열로 변환 후 클러스터링
def embed_cluster_texts(data_dict):
    text_embeddings_np = np.array(data_dict['vector'])
    cluster_labels = perform_clustering(text_embeddings_np, 10, 0.5)
    data_dict["cluster"] = cluster_labels
    return data_dict

# 텍스트 요약
async def embed_cluster_summarize_texts(data_dict: dict, level: int, SummarizeModel: str, worker_id: int) -> Tuple[dict, dict]:
    data_dict = embed_cluster_texts(data_dict)
    data_dict['level'] = [level] * len(data_dict['text'])

    expanded_dict = {
        "text": [],
        "vector": [],
        "cluster": [],
        "level": []
    }
    for i in range(len(data_dict['text'])):
        for cluster in data_dict["cluster"][i]:
            expanded_dict["text"].append(data_dict["text"][i])
            expanded_dict["vector"].append(data_dict["vector"][i])
            expanded_dict["cluster"].append(cluster)
            expanded_dict["level"].append(data_dict["level"][i])

    all_clusters = set(expanded_dict["cluster"])
    logger.info(f"--Generated {len(all_clusters)} clusters--")

    summaries = []
    for i in all_clusters:
        cluster_texts = [expanded_dict["text"][j] for j in range(len(expanded_dict["text"])) if expanded_dict["cluster"][j] == i]
        formatted_txt = "--- --- \n --- --- ".join(cluster_texts)
        
        if SummarizeModel == "Llama3":
            summary = await Llama3_summarize(formatted_txt, worker_id)
        else:
            raise ValueError(f"Unsupported summarization model: {SummarizeModel}")
        
        summaries.append(summary)

    summary_dict = {
        "summaries": summaries,
        "level": [level] * len(summaries),
        "cluster": list(all_clusters),
    }

    return data_dict, summary_dict

# 트리 구축 및 Milvus DB 적재
async def recursive_embed_cluster_summarize_and_store(data_dict, EmbeddingModel, SummarizeModel, client, collection_name, parent_idx, worker_id, level=1):
    logger.info(f"Level {level} -> start!")
    data_dict, summary_dict = await embed_cluster_summarize_texts(data_dict, level, SummarizeModel, worker_id)
    
    # 현재 레벨의 데이터를 Milvus에 저장
    data_dict['parent_idx'] = [parent_idx] * len(data_dict['text'])
    data_dict['summarize'] = ['X' if level == 1 else 'O'] * len(data_dict['text'])
    await insert_data_to_milvus(client, collection_name, data_dict)
    
    unique_clusters = len(set(summary_dict["cluster"]))
    if unique_clusters > 1:
        new_texts = summary_dict["summaries"]
        new_embeddings = await embed(new_texts, EmbeddingModel)
        new_dict = {
            'text': new_texts,
            'vector': new_embeddings
        }
        await recursive_embed_cluster_summarize_and_store(new_dict, EmbeddingModel, SummarizeModel, client, collection_name, parent_idx, worker_id, level + 1)
    else:
        # 마지막 레벨 (단일 클러스터)
        final_summary = summary_dict["summaries"][0]
        final_embedding = (await embed([final_summary], EmbeddingModel))[0]
        final_dict = {
            'text': [final_summary], 
            'vector': [final_embedding], 
            'level': [level+1],
            'parent_idx': [parent_idx],
            'summarize': ['O']
        }
        await insert_data_to_milvus(client, collection_name, final_dict)

# 임베딩 모델 Demension 정의
def get_vector_dimensions(embedding_model):
    dimensions = {
        "bge": 1024,
        "krsbert": 768,
        "minilm": 384
    }
    return dimensions.get(embedding_model, 0)

# Milvus Scema 정의
def create_milvus_schema(dim):
    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field(field_name="idx", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65355)
    schema.add_field(field_name="level", datatype=DataType.INT64)
    schema.add_field(field_name="summarize", datatype=DataType.VARCHAR, max_length=1)
    schema.add_field(field_name="parent_idx", datatype=DataType.INT64)
    return schema

# Milvus Index Parameters 정의
def create_index_params(client):
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector", 
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )
    return index_params

# 동명의 Collection이 있는 지 확인
def check_collection_exists(client, collection_name):
    collections = client.list_collections()
    return collection_name in collections
    
# Collection 생성
def create_milvus_collection(client, collection_name, schema, index_params):
    if not check_collection_exists(client, collection_name):
        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
        )
        
# Collection이 없으면 생성
def create_collection_if_not_exists(client, collection_name, EmbeddingModel):
    if not check_collection_exists(client, collection_name):
        dim = get_vector_dimensions(EmbeddingModel)
        schema = create_milvus_schema(dim)
        index_params = create_index_params(client)
        create_milvus_collection(client, collection_name, schema, index_params)
        logger.info(f"Collection '{collection_name}' created.")
    else:
        logger.info(f"Collection '{collection_name}' already exists.")

# Collection에 데이터 삽입
async def insert_data_to_milvus(client, collection_name, data_dict, batch_size=512, delay=0.1):
    total_records = len(data_dict['text'])
    for i in range(0, total_records, batch_size):
        batch_data = [
            {
                "vector": data_dict['vector'][j],
                "text": data_dict['text'][j],
                "level": int(data_dict['level'][j]),
                "summarize": data_dict['summarize'][j],
                "parent_idx": int(data_dict['parent_idx'][j])
            }
            for j in range(i, min(i+batch_size, total_records))
        ]
        client.insert(collection_name=collection_name, data=batch_data)
        
        if i + batch_size < total_records:
            await asyncio.sleep(delay)

# Collection Load 대기
async def wait_for_collection_load(client, collection_name, wait_time=5):
    await asyncio.sleep(wait_time)
    return client.get_load_state(collection_name=collection_name)
    
# api raptor
async def raptor(data, callback_func):
    databaseName = data.databaseName
    summarizeModel = data.summarizeModel
    embeddingModel = data.embeddingModel
    targetVectorTables = data.targetVectorTables
    newVectorTableName = data.newVectorTableName
    worker_id = data.workerId
    logger.info(f"Wocker ID: {worker_id}")
    
    # Worker ID에 따라 딜레이 적용
    if worker_id == 2:
        await asyncio.sleep(60)
    elif worker_id == 3:
        await asyncio.sleep(120)
    
    load_dotenv()

    MILVUS_ENDPOINT = os.getenv('MILVUS_ENDPOINT')
    MILVUS_TOKEN = os.getenv('MILVUS_TOKEN')

    client = MilvusClient(
            uri=MILVUS_ENDPOINT,
            token=MILVUS_TOKEN,
            db_name=databaseName
        )
    
    create_collection_if_not_exists(client, newVectorTableName, embeddingModel)
    logger.info(f"Number of tables: {len(targetVectorTables)}")
    
    successfulTables = []
    failedTables = [table.vectorTableName for table in targetVectorTables]
    total_tables = len(targetVectorTables)
    sequence_tables = 0
    
    try:
        for table_info in targetVectorTables:
            logger.info(f"{table_info.vectorTableName} -> start!")
            vectorTableIdx = table_info.vectorTableIdx
            
            start_time = time.time()
            sequence_tables += 1
            
            try:
                combined_data = combined_texts(table_info, client)
                
                await recursive_embed_cluster_summarize_and_store(combined_data, embeddingModel, summarizeModel, client, newVectorTableName, vectorTableIdx, worker_id)
                successfulTables.append(table_info.vectorTableName)
                failedTables.remove(table_info.vectorTableName)
                logger.info(f"{table_info.vectorTableName} -> successful!")
                
                await callback_func(table_info.vectorTableName, True)
            
            except Exception as e:
                logger.error(f"{table_info.vectorTableName} -> failed: {str(e)}")
                
                await callback_func(table_info.vectorTableName, False)
                
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            hours, rem = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(rem, 60)
            logger.info(f"{table_info.vectorTableName} -> Process Time: {int(hours):02} %T {int(minutes):02} %M {int(seconds):02} %S")
            
            progress = (sequence_tables) / total_tables * 100
            logger.info(f"Progress: {sequence_tables}/{total_tables} ({progress:.2f}%)")
                
    except Exception as e:
        logger.error(f"RAPTOR Error Occurred: {str(e)}")
                
    finally:
        await wait_for_collection_load(client, newVectorTableName)
        client.close()
        result = {
            "successfulTables": successfulTables,
            "failedTables": failedTables
        }
    
    logger.info(f"RAPTOR complete: {newVectorTableName}")
    return result