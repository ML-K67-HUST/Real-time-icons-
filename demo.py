import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np

class SentimentEmojiPredictor:
    def __init__(self):
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english"
        )
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english"
        )
        
        self.emoji_map = {
            'joy': ['😄', '😊', '🌞', '🎉', '✨'],
            'sadness': ['😢', '😔', '💔', '🌧️', '😞'],
            'anger': ['😠', '🤬', '😡', '💢', '🔥'],
            'fear': ['😱', '😨', '🙀', '😰', '👻'],
            'surprise': ['😮', '🤯', '😲', '🎈', '🌈'],
            'neutral': ['😐', '🤨', '🫤', '😶', '🤔'],
            
            'education': ['📚', '✏️', '🎓', '📝', '🧠'],
            'time': ['⏰', '⌛', '🕰️', '⏳', '📅'],
            'family': ['👨‍👩‍👧', '❤️', '🤗', '👪', '🏡'],
            'success': ['🏆', '🌟', '💪', '🎯', '🚀'],
            'reflection': ['🤔', '💭', '🌈', '🌻', '🌄']
        }

        try:
            self.context_model = AutoModelForSequenceClassification.from_pretrained(
                "nlptown/bert-base-multilingual-uncased-sentiment"
            )
            self.context_tokenizer = AutoTokenizer.from_pretrained(
                "nlptown/bert-base-multilingual-uncased-sentiment"
            )
        except:
            self.context_model = None
    
    def predict_sentiment(self, text):
        """Predict overall sentiment of the text."""
        inputs = self.sentiment_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = self.sentiment_model(**inputs)
        
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        sentiment_label = torch.argmax(probs).item()
        sentiment_score = probs[0][sentiment_label].item()
        
        return 'positive' if sentiment_label == 1 else 'negative', sentiment_score
    
    def suggest_emojis(self, text):
        """Suggest contextual and emotional emojis."""
        sentiment_type, sentiment_score = self.predict_sentiment(text)
        
        base_emojis = self.emoji_map.get(sentiment_type, self.emoji_map['neutral'])
        
        context_emojis = []
        context_keywords = {
            'education': ['study', 'exam', 'school', 'learn', 'homework'],
            'time': ['time', 'clock', 'moment', 'deadline'],
            'family': ['parent', 'dad', 'mom', 'family', 'love'],
            'success': ['achieve', 'win', 'goal', 'success'],
            'reflection': ['think', 'reflect', 'consider', 'understand']
        }
        for context, keywords in context_keywords.items():
            if any(keyword in text.lower() for keyword in keywords):
                context_emojis.extend(self.emoji_map.get(context, []))
        
        all_emojis = list(dict.fromkeys(base_emojis + context_emojis))
        
        if sentiment_score > 0.8:
            selected_emojis = all_emojis[:2] 
        elif sentiment_score < 0.3:
            selected_emojis = all_emojis[-2:] 
        else:
            selected_emojis = all_emojis[:1] 
        return selected_emojis
    
    def process_text(self, text):
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        processed_sentences = []
        for sentence in sentences:
            emojis = self.suggest_emojis(sentence)
            emoji_str = ' '.join(emojis)
            processed_sentence = f"{emoji_str} {sentence}"
            processed_sentences.append(processed_sentence)
        
        return ' '.join(processed_sentences)

predictor = SentimentEmojiPredictor()

def emoji_interface(text):
    return predictor.process_text(text)

iface = gr.Interface(
    fn=emoji_interface,
    inputs=gr.Textbox(lines=5, placeholder="Enter your text here..."),
    outputs=gr.Textbox(lines=5),
    title="AI Sentiment & Emoji Predictor",
    description="Automatically add contextual and sentiment-based emojis to your text!"
)
iface.launch()