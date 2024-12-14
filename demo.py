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
                     top_k=5)

emoji_mapping = {
    # Emotions
    'joy': '😊',
    'happy': '😃',
    'laugh': '😄',
    'grin': '😁',
    'excitement': '🤩',
    'surprise': '😲',
    'neutral': '😐',
    'sad': '😢',
    'cry': '😭',
    'fear': '😨',
    'scared': '😱',
    'anger': '😠',
    'mad': '🤬',
    'disgust': '🤢',
    'love': '🥰',
    'heart': '❤️',
    'tired': '😴',
    'sleepy': '🥱',
    'sick': '🤒',
    'nerdy': '🤓',
    'cool': '😎',
    'wink': '😉',
    'silly': '🤪',
    'worried': '😟',
    'confused': '😕',
    'shocked': '😳',

    # Animals
    'dog': '🐕',
    'cat': '🐈',
    'mouse': '🐁',
    'hamster': '🐹',
    'rabbit': '🐇',
    'fox': '🦊',
    'bear': '🐻',
    'panda': '🐼',
    'koala': '🐨',
    'tiger': '🐯',
    'lion': '🦁',
    'cow': '🐄',
    'pig': '🐷',
    'frog': '🐸',
    'monkey': '🐒',
    'chicken': '🐔',
    'penguin': '🐧',
    'bird': '🐦',
    'eagle': '🦅',
    'duck': '🦆',
    'swan': '🦢',
    'dove': '🕊️',
    'butterfly': '🦋',
    'bee': '🐝',

    # Food and Drinks
    'pizza': '🍕',
    'burger': '🍔',
    'sandwich': '🥪',
    'hotdog': '🌭',
    'taco': '🌮',
    'sushi': '🍣',
    'rice': '🍚',
    'noodles': '🍜',
    'bread': '🍞',
    'cheese': '🧀',
    'egg': '🥚',
    'coffee': '☕',
    'tea': '🫖',
    'milk': '🥛',
    'wine': '🍷',
    'beer': '🍺',

    # Transportation
    'car': '🚗',
    'taxi': '🚕',
    'bus': '🚌',
    'truck': '🚛',
    'bicycle': '🚲',
    'motorcycle': '🏍️',
    'train': '🚂',
    'airplane': '✈️',
    'helicopter': '🚁',
    'boat': '⛵',
    'ship': '🚢',

    # Weather & Nature
    'sun': '☀️',
    'moon': '🌙',
    'star': '⭐',
    'cloud': '☁️',
    'rain': '🌧️',
    'snow': '❄️',
    'thunder': '⚡',
    'rainbow': '🌈',
    'flower': '🌸',
    'tree': '🌳',
    'leaf': '🍁',

    # Sports & Activities
    'football': '⚽',
    'basketball': '🏀',
    'baseball': '⚾',
    'tennis': '🎾',
    'volleyball': '🏐',
    'swimming': '🏊',
    'running': '🏃',
    'dancing': '💃',
    'skiing': '⛷️',
    'surfing': '🏄',

    # Objects & Tools
    'phone': '📱',
    'computer': '💻',
    'camera': '📷',
    'book': '📚',
    'pen': '✒️',
    'pencil': '✏️',
    'scissors': '✂️',
    'key': '🔑',
    'lock': '🔒',
    'clock': '⏰',
    'gift': '🎁',
    'money': '💰',
    'shopping': '🛍️',

    # Clothing & Fashion
    'dress': '👗',
    'shirt': '👕',
    'pants': '👖',
    'shoes': '👟',
    'boot': '👢',
    'hat': '🎩',
    'crown': '👑',
    'glasses': '👓',
    'handbag': '👜',

    # Places & Buildings
    'house': '🏠',
    'office': '🏢',
    'school': '🏫',
    'hospital': '🏥',
    'castle': '🏰',
    'church': '⛪',
    'hotel': '🏨',
    'store': '🏪',
    'bank': '🏦',

    # Symbols
    'heart_symbol': '♥️',
    'peace': '✌️',
    'check': '✅',
    'cross': '❌',
    'warning': '⚠️',
    'question': '❓',
    'music': '🎵',
    'fire': '🔥',
    'sparkle': '✨'
}

def predict_emojis(text):
    """
    Predict emojis based on text analysis.
    Returns relevant emojis for detected words.
    """
    if not text or len(text.strip()) == 0:
        return "😐"
    
    try:
        # Convert text to lowercase and split into words
        words = text.lower().split()
        
        # Find matching emojis
        found_emojis = []
        for word in words:
            if word in emoji_mapping:
                found_emojis.append(emoji_mapping[word])
        
        # If no emojis found, try emotion classification
        if not found_emojis:
            predictions = classifier(text)
            for pred in predictions[0]:
                emotion = pred['label']
                if pred['score'] > 0.3 and emotion in emoji_mapping:
                    found_emojis.append(emoji_mapping[emotion])
        
        return " ".join(found_emojis) if found_emojis else "😐"
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        return "😐"

css = """
.output-emoji { 
    font-size: 2.5em; 
    line-height: 1.5;
    word-wrap: break-word;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("# Smart Emoji Predictor")
    
    with gr.Row():
        text_input = gr.Textbox(
            label="Type something...",
            placeholder="Express yourself...",
            show_label=True,
            lines=3
        )
        emoji_output = gr.Textbox(
            label="Predicted Emojis",
            show_label=True,
            interactive=False,
            elem_classes=["output-emoji"]
        )
    
    gr.Markdown("""
    This enhanced demo combines emotion recognition with word-to-emoji mapping.
    It analyzes your text in real-time and predicts appropriate emojis based on both
    emotional content and specific words or phrases.
    """)
    
    text_input.change(
        fn=predict_emojis,
        inputs=[text_input],
        outputs=[emoji_output],
        show_progress=False
    )

if __name__ == "__main__":
    demo.queue().launch()