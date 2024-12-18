import re
from nltk import WordNetLemmatizer
import json
from openai import OpenAI
from pydantic import BaseModel, Field
def cleanHtml(sentence):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', str(sentence))
    return cleantext

def cleanPunc(sentence):
    cleaned = re.sub(r'[?|!|\'|"|#]', r'', sentence)
    cleaned = re.sub(r'[.|,|)|(|\\|/]', r' ', cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n", " ")
    return cleaned

def keepAlpha(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent

def removeStopWords(sentence):
    global re_stop_words
    return re_stop_words.sub(" ", sentence)

def lemmatize(sentence):
    lemmatizer = WordNetLemmatizer()
    lemSentence = ""
    for word in sentence.split():
        lem = lemmatizer.lemmatize(word)
        lemSentence += lem
        lemSentence += " "
    lemSentence = lemSentence.strip()
    return lemSentence

def place(icon,text):
    class Converted(BaseModel):
        converted_sentence: str = Field(description="your emoji added sentence here. If you can't complete the request, then this must be `none`")
    if not text:
        return text
    TOGETHER_API_KEY = "54f853e31dcf2a608cdd6a11a05ab5bd3c5212add172fc1a3a9eb557d6da4ab3"
    client = OpenAI(
        api_key=TOGETHER_API_KEY,
        base_url='https://api.together.xyz/v1',
    )
    chat = [
        {
            "role": "system",
            "content": f"""
                Emoji list: {icon}
                User question is an sentence. Place these emoji provided  to the appropriate place in sentence, if you dont find an icon appropriate to the text meaning, you can place your own emoji.
                If this request violates your policy or you can't fufill the request, then your converted_sentence should be : `NONE`.
                Only reply the original sentence which has been augmented emoji, like json format below, do not reply anything else. REPLY STRICTLY TO THIS REPSONSE FORMAT:
              """
        },
        {
            "role": "user",
            "content": text
        }
    ]
    # print(f'called {model}')
    model ="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    chat_response = client.chat.completions.create(
            model = model,
            response_format={
                "type":"json_object",
                "schema": Converted.model_json_schema()
            },
            messages=chat,
            top_p = 0.2,
            # max_tokens = 6000,
            stream = False,
            
       
    )
    print(chat_response.choices[0].message.content)
    try:
        response = json.loads(chat_response.choices[0].message.content)["converted_sentence"]
    except:
        response = 'none'
    if 'none' in response.lower():
        return text + ' '+ icon[0][1]
    else:
        return response

def preprocess_text(text):
    text = text.lower()
    text = cleanHtml(text)
    text = cleanPunc(text)
    text = keepAlpha(text)
    text = removeStopWords(text)
    text = lemmatize(text)
    return text