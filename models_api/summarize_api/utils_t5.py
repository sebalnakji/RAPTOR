import torch
from models.t5 import T5Model

t5_instance = T5Model.get_instance()
tokenizer, model = t5_instance.get_tokenizer_and_model()

def summarize_t5(data):

    segMaxSize = data.segMaxSize
    segMinSize = data.segMinSize
    texts = data.texts
    
    if segMaxSize <= 0 or segMaxSize > 3000 or segMinSize > segMaxSize or segMinSize < 0:
        raise ValueError("유효하지 않은 인자 값입니다")
    
    decoded_output_list , token_size_list = [], []
    for text in texts:
        chat = ["summarize: " + str(text)]

        inputs = tokenizer(chat, return_tensors='pt').to('cuda')

        with torch.no_grad():
            output = model.generate(
                **inputs, 
                num_beams=1, 
                do_sample=False, 
                min_length=segMinSize, 
                max_length=segMaxSize
            )
            
        decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        
        token_size = inputs['input_ids'].shape[1] - 7

        decoded_output_list.append(decoded_output)
        token_size_list.append(token_size)

    return {
        "texts": decoded_output_list, 
        "inputTokens":token_size_list
        }