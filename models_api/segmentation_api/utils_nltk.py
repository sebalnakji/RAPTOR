import nltk
from transformers import AutoTokenizer

# 토크나이저 초기화
tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")

def segment_nltk(data):
        
    # def whether_fix_sentence_False(texts):
    #     tokens = tokenizer(texts)['input_ids']
    #     paragraphs = []
    #     temp_para = []
    #     for token in tokens:
    #         temp_para.append(token)
    #         if len(temp_para) == chunk_length:
    #             paragraphs.append(temp_para)
    #             temp_para =[]
    #     if temp_para:
    #         paragraphs.append(temp_para)

    #     topicSeg , span = [] , []
    #     span_ind = 0    
    #     for paragraph in paragraphs:
    #         temp = tokenizer.decode(paragraph, skip_special_tokens=True)
    #         topicSeg.append(temp)
    #         span.append(span_ind)
    #         span_ind+=1
        
    #     return topicSeg , span

    def whether_fix_sentence_True(texts):
        results = []
        paragraph_length = 0
        temp_paragraph = []

        # 문장별로 나누기
        sentence_list = nltk.sent_tokenize(texts)
        for sentence in sentence_list:
            # 문장 토크나이징
            tokens = tokenizer(sentence)['input_ids']
            if len(tokens) > chunk_length:
                continue
            # elif :
            #     raise ValueError("주어진 문장의 길이보다 chunk_length가 작습니다. chunk_length 값을 올려주세요.")
            else:
                if paragraph_length + len(tokens) > chunk_length:
                    results.append(temp_paragraph)
                    temp_paragraph = []
                    temp_paragraph.append(tokens)
                    paragraph_length = len(tokens)
                else:
                    temp_paragraph.append(tokens)
                    paragraph_length += len(tokens)

        if temp_paragraph:
            results.append(temp_paragraph)

        topicSeg , span = [] , []
        span_ind = 0


        for result in results:
            temp_list = []
            temp_span = []
            for tokens in result:
                temp = tokenizer.decode(tokens, skip_special_tokens=True)
                temp_list.append(temp)
                temp_span.append(span_ind)
                span_ind += 1
            span.append(', '.join(map(str,temp_span)))
            topicSeg.append(' '.join(map(str, temp_list)))
        return topicSeg , span
    
    def split_with_overlap(texts, chunk_length, overlap_length):
        tokens = tokenizer(texts)['input_ids']
        total_length = len(tokens)
        # 결과를 저장할 리스트
        tokenized_texts = []

        # 시작 인덱스를 0으로 초기화합니다.
        start_idx = 0

        while start_idx < total_length:
            end_idx = min(start_idx + chunk_length, total_length)
            # 현재 조각을 가져옵니다.
            tokenized_texts.append(tokens[start_idx:end_idx])
            print(start_idx, end_idx)
            
            # 다음 조각의 시작 인덱스를 업데이트합니다.
            start_idx += chunk_length - overlap_length
            
            # 마지막 조각이 최대 길이에 미치지 못할 경우, 종료합니다.
            if end_idx == total_length:
                break

        topicSeg , span = [] , []
        span_ind = 0    

        for paragraph in tokenized_texts:
            temp = tokenizer.decode(paragraph, skip_special_tokens=True)
            topicSeg.append(temp)
            span.append(span_ind)
            span_ind+=1

        return topicSeg , span



    chunk_length = data.chunk_length
    whether_fix_sentence = data.whether_fix_sentence
    texts = data.texts
    overlap_length = data.overlap_length

    if chunk_length <= overlap_length:
        raise ValueError("유효하지 않은 인자 값입니다")
    
    # 텍스트 처리
    topicSeg_list, span_list, inputTokens_list =[], [], []
        
    for text in texts:
        if whether_fix_sentence:
            topicSeg , span = whether_fix_sentence_True(text)   
        else:
            topicSeg , span = split_with_overlap(text, chunk_length, overlap_length)
            # topicSeg , span = whether_fix_sentence_False(text)  
        # inputTokens 계산
        inputTokens = len(tokenizer(text)['input_ids'])
        totalTokens = inputTokens - 2
        topicSeg_list.append(topicSeg)
        span_list.append(span)
        inputTokens_list.append(totalTokens)

    return {
    "topicSeg": topicSeg_list,
    "span": span_list,
    "inputTokens": inputTokens_list
    }