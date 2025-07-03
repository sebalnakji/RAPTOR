import re
import os
import nltk
from dotenv import load_dotenv
from openai import OpenAI


def segment_llama(data):
    texts = data.texts
    segCount = data.segCount
    segMaxSize = data.segMaxSize
    segMinSize = data.segMinSize
    
    if (segCount <= 0 and segCount != -1) or segMaxSize <= 0 or segMaxSize > 3000 or segMinSize > segMaxSize or segMinSize < 0:
        raise ValueError("유효하지 않은 인자 값입니다")
    
    load_dotenv()
    
    DATABRICKS_TOKEN = os.getenv('DATABRICKS_TOKEN_4')
    endpoints = os.getenv('DATABRICKS_ENDPOINT_4')
    model_id = "databricks-meta-llama-3-70b-instruct"
    
    client = OpenAI(
        api_key=DATABRICKS_TOKEN,
        base_url=endpoints
    )
    
    topicSeg_list , span_list , inputTokens_list = [], [], []
    
    for text in texts:
        response = client.chat.completions.create(
        model= model_id,
        messages = [
            { "role": "system", "content":
        """
        반드시 한국어로 대답해주세요.
        해당 내용을 문장 간 유사도를 파악하여 단락을 구분하되
        원본 텍스트는 변형하지 마세요. 텍스트 추가도 안됩니다.
        다음 조건을 따라주세요.

        조건:
        1. {}개의 문단으로 구성해주세요.
        2. 만약 1에서 '-1개의 문단으로 구성을 요청'받으면 최적의 단락 수로 구성해주세요.
        3. 한 문단의 최대 글자 수 {}자
        4. 한 문단의 최소 글자 수 {}자
        5. 아무 텍스트도 받지 않았을 시 '텍스트가 없습니다.'라고 하세요.
        
        """.format(segCount, segMaxSize, segMinSize)
        },
        { "role": "user", "content": f"{text}" }
        ],
        temperature=0.1,
        top_p=0.95
        )
        
        output = response.choices[0].message.content
        tokens = response.usage.prompt_tokens - 186
        paragraphs = re.split(r'\n\s*\n', output)
        topicSeg = [nltk.sent_tokenize(paragraph) for paragraph in paragraphs]
        flattened_topicSeg = [sentence for sublist in topicSeg for sentence in sublist]
        span = list(range(len(flattened_topicSeg)))
        
        topicSeg_list.append(flattened_topicSeg)
        span_list.append(span)
        inputTokens_list.append(tokens)
        
    return {
        "topicSeg": topicSeg_list,
        "span": span_list,
        "inputTokens": inputTokens_list
    }