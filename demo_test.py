import gradio as gr
import json
from utils import place

with open('EMOJI.json', 'r') as f:
    emoji_map = json.load(f)

def preprocess_input(test_text):
    return test_text.split('.')

def infer(test_text):
    from infer_goemotion import Sentiment
    from infer_reuter import infer_reuter
    sentiment1 = []
    sentiment2 = []
    res1 = Sentiment(test_text)
    res2 = infer_reuter(test_text)
    
    for labels in res1['predicted']:
        sentiment1.append((labels['label'], emoji_map[labels['label']]))
    
    for item in res2['magnet']:
        sentiment2.append((item['label'], emoji_map[item['label']]))
    for item in res2['dnn']:
        sentiment2.append((item['label'], emoji_map[item['label']]))
    for item in res2['lstm']:
        sentiment2.append((item['label'], emoji_map[item['label']]))
    
    return sentiment1 + sentiment2

def infer_sentiment_only(test_text):
    from infer_goemotion import Sentiment
    sentiment1 = []
    res1 = Sentiment(test_text)
    
    for labels in res1['predicted']:
        sentiment1.append((labels['label'], emoji_map[labels['label']]))
    
    return sentiment1

def infer_stuff_only(test_text):
    from infer_reuter import infer_reuter
    sentiment2 = []
    res2 = infer_reuter(test_text)
    
    for item in res2['dnn']:
        sentiment2.append((item['label'], emoji_map[item['label']]))
    for item in res2['lstm']:
        sentiment2.append((item['label'], emoji_map[item['label']]))
    
    return sentiment2

def get_response(test_text, mode):
    texts = preprocess_input(test_text)
    result = ""
    
    for text in texts:
        if mode == 'sentiment':
            icons = infer_sentiment_only(text)
        elif mode == "stuff":
            icons = infer_stuff_only(text)
        elif mode == "all":
            icons = infer(text)
        
        result += place(icons, text) + '.'
    
    return result[:-1]

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
                
                mode_dropdown = gr.Dropdown(
                    ['sentiment', 'stuff', 'all'], 
                    value='all', 
                    label="Analysis Mode"
                )
                
                process_btn = gr.Button("Analyze Text", variant="primary")
                
            with gr.Column(scale=2):
                processed_text = gr.Textbox(
                    lines=5,
                    label="Processed Text with Emojis",
                    interactive=False
                )
        
        gr.Examples(
            examples=[
                "I had a tough day at work, but I'm staying positive.",
                "My team just won the championship, and I'm ecstatic!",
                "Life has its ups and downs, but I'm learning to embrace them."
            ],
            inputs=text_input
        )
        
        process_btn.click(
            get_response, 
            inputs=[text_input, mode_dropdown], 
            outputs=processed_text
        )
    
    return demo

demo = create_interface()
demo.launch(share=True, debug=True)