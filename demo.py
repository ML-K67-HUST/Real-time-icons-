import gradio as gr
import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import numpy as np

model_name = "j-hartmann/emotion-english-distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

classifier = pipeline("text-classification", 
                     model=model, 
                     tokenizer=tokenizer,
                     top_k=3) 

emotion_to_emoji = {
    'joy': '😊',
    'surprise': '😲',
    'neutral': '😐',
    'sadness': '😢',
    'fear': '😨',
    'anger': '😠',
    'disgust': '🤢',
    'love': '🥰'
}

def predict_emojis(text):
    """
    Predict emojis based on emotion classification of input text.
    Returns top emotions with their emojis.
    """
    if not text or len(text.strip()) == 0:
        return "😐"
    
    try:
        predictions = classifier(text)
        emojis = []
        for pred in predictions[0]:
            emotion = pred['label']
            score = pred['score']
            if score > 0.2:
                emoji = emotion_to_emoji.get(emotion, '')
                if emoji:
                    emojis.append(emoji)
        
        if not emojis:
            return "😐"
        
        return "".join(emojis)
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        return "😐"

css = """
.output-emoji { font-size: 2em; }
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("# Real-time Emotion-based Emoji Predictor")
    
    with gr.Row():
        text_input = gr.Textbox(
            label="Type something...",
            placeholder="Express how you feel...",
            show_label=True,
            lines=2
        )
        emoji_output = gr.Textbox(
            label="Predicted Emojis",
            show_label=True,
            interactive=False,
            elem_classes=["output-emoji"]
        )
    
    gr.Markdown("""
    This demo uses a deep learning model (DistilRoBERTa) fine-tuned for emotion recognition.
    It analyzes your text in real-time and predicts appropriate emojis based on detected emotions.
    """)
    
    # Update on every keypress with debouncing
    text_input.change(
        fn=predict_emojis,
        inputs=[text_input],
        outputs=[emoji_output],
        show_progress=False
    )
if __name__ == "__main__":
    demo.queue().launch()