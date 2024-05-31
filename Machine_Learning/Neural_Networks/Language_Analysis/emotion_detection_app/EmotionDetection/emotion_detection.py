from scipy.special import softmax
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForSequenceClassification.from_pretrained(model)

def emotion_detector(text_to_analyse):
    encode_text = tokenizer(text_to_analyse, return_tensors='pt') # pytorch tensor
    output_score = model(**encode_text)
    array_results = output_score[0][0].detach().numpy() # convert the result in a numpy array
    score = softmax(array_results) # get the score in percentage (negative - neutral - positive)
    score = list(score)

    response = {'negative': score[0], 'neutral': score[1], 'positive': score[2]}
    max_value = max(response, key=response.get)
    response['emotion'] = max_value
    #response = json.loads(response)
        
    return response

if __name__ == "__main__":
    response = emotion_detector('hi, i am happy')
    print(response)