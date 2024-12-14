import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import random
import re
import textblob

class SentimentEmojiPredictor:
    def __init__(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english"
        )
        
        self.emojis = {
            'positive': ['🌟', '😊', '👍', '❤️', '🎉', '🌈', '✨', '🏆', '💖', '🚀'],
            'negative': ['😞', '😔', '😢', '👎', '💔', '🌧️', '🥀', '💥', '🚫', '😱'],
            'neutral': ['😐', '🤨', '😶', '🫤', '🤔', '🧐', '🤷', '😑', '🫥', '🤨']
        }
    
    def advanced_sentiment_analysis(self, text):
        blob = textblob.TextBlob(text)
        
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        sentiment_label = torch.argmax(probs).item()
        
        return {
            'ml_sentiment': 'positive' if sentiment_label == 1 else 'negative',
            'textblob_polarity': blob.sentiment.polarity,
            'textblob_subjectivity': blob.sentiment.subjectivity
        }
    
    def process_text(self, text):
        sentiment_analysis = self.advanced_sentiment_analysis(text)
        sentiment_type = sentiment_analysis['ml_sentiment']
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        processed_sentences = []
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            emoji_choices = self.emojis[sentiment_type]
            emoji = emoji_choices[random.randint(0, len(emoji_choices) - 1)]
            
            processed_sentence = f"{sentence.strip()} {emoji}"
            processed_sentences.append(processed_sentence)
        
        analysis_text = (
            f"Sentiment Analysis:\n"
            f"Machine Learning Sentiment: {sentiment_analysis['ml_sentiment'].capitalize()}\n"
            f"Polarity: {sentiment_analysis['textblob_polarity']:.2f}\n"
            f"Subjectivity: {sentiment_analysis['textblob_subjectivity']:.2f}"
        )
        
        return ' '.join(processed_sentences), analysis_text
    
    def generate_random_text(self, mood='random'):
        texts = {
            'happy': [
                "Today was an amazing day! I accomplished all my goals and felt incredibly proud.",
                "My friends surprised me with a wonderful celebration that made me so happy!",
                "I received great news about my project, and I'm thrilled beyond words!"
            ],
            'sad': [
                "I've been struggling with my emotions and feeling quite down lately.",
                "Things didn't go as planned, and I'm finding it hard to stay positive.",
                "I miss the way things used to be and feel lost right now."
            ],
            'neutral': [
                "Just another day at work, doing my routine tasks.",
                "I'm thinking about my plans for the weekend.",
                "The weather seems okay today, neither good nor bad."
            ],
            'random': [
                "Life is full of unexpected moments that keep us guessing.",
                "Sometimes the journey is more important than the destination.",
                "Learning and growing are constant parts of our existence."
            ]
        }
        
        return random.choice(texts.get(mood, texts['random']))

predictor = SentimentEmojiPredictor()

def create_interface():
    with gr.Blocks(theme='soft') as demo:
        gr.Markdown("# 🎭 Sentiment & Emoji Analyzer")
        
        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    lines=5, 
                    placeholder="Enter your text here...",
                    label="Input Text"
                )
                
                with gr.Row():
                    process_btn = gr.Button("Process Text", variant="primary")
                    generate_btn = gr.Button("Generate Sample Text", variant="secondary")
                
            with gr.Column(scale=2):
                processed_text = gr.Textbox(
                    lines=5,
                    label="Processed Text",
                    interactive=False
                )
                
                analysis_text = gr.Textbox(
                    lines=5,
                    label="Analysis",
                    interactive=False
                )
        
        generate_mood = gr.Dropdown(
            ['random', 'happy', 'sad', 'neutral'], 
            value='random', 
            label="Mood for Generated Text"
        )
        
        process_btn.click(
            predictor.process_text, 
            inputs=text_input, 
            outputs=[processed_text, analysis_text]
        )
        
        generate_btn.click(
            predictor.generate_random_text, 
            inputs=generate_mood, 
            outputs=text_input
        )
        
        gr.Examples(
            examples=[
                "I had a tough day at work, but I'm staying positive.",
                "My team just won the championship, and I'm ecstatic!",
                "Life has its ups and downs, but I'm learning to embrace them."
            ],
            inputs=text_input
        )
    
    return demo

demo = create_interface()
demo.launch(
    share=True,  
    debug=True
)