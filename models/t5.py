from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class T5Model:
    _instance = None

    @staticmethod
    def get_instance():
        if T5Model._instance is None:
            T5Model()
        return T5Model._instance

    def __init__(self):
        if T5Model._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            model_id = "lcw99/t5-large-korean-text-summary"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_id,
                torch_dtype='auto',
                device_map='cuda:0'
            )
            T5Model._instance = self

    def get_tokenizer_and_model(self):
        return self.tokenizer, self.model