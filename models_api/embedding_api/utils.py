# import torch
from sentence_transformers import SentenceTransformer
from FlagEmbedding import BGEM3FlagModel


embedding_model_bge = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, device='cuda:0')
embedding_model_minilm = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device='cuda:0')
embedding_model_krsbert = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS', device='cuda:0')


# # GPU 디바이스 설정
# device_0 = torch.device('cuda:0')
# device_1 = torch.device('cuda:1')

# 모델 로드 및 각 GPU에 할당
# embedding_model_bge = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, device=device_0)
# embedding_model_minilm = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device=device_1)
# embedding_model_krsbert = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS', device=device_1)

# 모델을 병렬 처리 모드로 설정
# if torch.cuda.device_count() > 1:
#     embedding_model_bge = torch.nn.DataParallel(embedding_model_bge)
#     embedding_model_minilm = torch.nn.DataParallel(embedding_model_minilm)
#     embedding_model_krsbert = torch.nn.DataParallel(embedding_model_krsbert)


def BGE(data):
    text_list = data.texts
    embedding_list , inputToken_list = [], []
    
    # for text in text_list:
    #     tokens = embedding_model_bge.encode(text, max_length=1000, return_sparse=True)
    #     lexical_weights = embedding_model_bge.convert_id_to_token(tokens['lexical_weights'])
    #     total_tokens = sum(len(token) for token in lexical_weights)
        
    #     if total_tokens > 1000:
    #         raise ValueError("유효하지 않은 인자 값입니다. 토큰 수가 1000을 초과합니다.")
    
    for text in text_list:
        embedding = embedding_model_bge.encode(text, max_length = 1000)
        embeddings = embedding['dense_vecs'].tolist()

        tokens = embedding_model_bge.encode(text, max_length = 1000, return_sparse=True)
        lexical_weights = embedding_model_bge.convert_id_to_token(tokens['lexical_weights'])
        total_tokens = sum(len(tokens) for tokens in lexical_weights)

        embedding_list.append(embeddings)
        inputToken_list.append(total_tokens)

    return {
        "embedding_list": embedding_list,
        "inputToken_list": inputToken_list
    }


def MiniLM(data):
    text_list = data.texts
    embedding_list , inputToken_list = [], []
    
    # for text in text_list:
    #     tokens = embedding_model_minilm.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    #     total_tokens = tokens['input_ids'].size(1)
        
        # if total_tokens > 128:
        #     raise ValueError("유효하지 않은 인자 값입니다. 토큰 수가 128을 초과합니다.")
    
    for text in text_list:
        embedding = embedding_model_minilm.encode(text).tolist()
        tokens = embedding_model_minilm.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        total_tokens = tokens['input_ids'].size(1)

        embedding_list.append(embedding)
        inputToken_list.append(total_tokens)

    return {
        "embedding_list": embedding_list,
        "inputToken_list": inputToken_list
    }


def KRsBert(data):
    text_list = data.texts
    embedding_list , inputToken_list = [], []
    
    # for text in text_list:
    #     tokens = embedding_model_krsbert.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    #     total_tokens = tokens['input_ids'].size(1)
        
    #     if total_tokens > 128:
    #         raise ValueError("유효하지 않은 인자 값입니다. 토큰 수가 128을 초과합니다.")
    
    for text in text_list:
        embedding = embedding_model_krsbert.encode(text).tolist()
        tokens = embedding_model_krsbert.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        total_tokens = tokens['input_ids'].size(1)
        embedding_list.append(embedding)
        inputToken_list.append(total_tokens)

    return {
        "embedding_list": embedding_list,
        "inputToken_list": inputToken_list
    }
