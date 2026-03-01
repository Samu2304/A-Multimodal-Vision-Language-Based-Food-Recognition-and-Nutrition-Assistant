import os
import warnings
import numpy as np
import pandas as pd
import torch
import faiss
import gradio as gr
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer

warnings.filterwarnings('ignore')

# ── Device ────────────────────────────────────────────────────────────────
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')

# ── Load Assets ───────────────────────────────────────────────────────────
print('Loading nutrition table...')
nutrition_wide = pd.read_csv('data/master_nutrition.csv')

print('Loading CLIP...')
clip_model     = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
clip_model.eval()

print('Loading FAISS index...')
faiss_index = faiss.read_index('data/recipe_faiss.index')
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
recipes_df  = pd.read_csv('data/recipes_subset.csv')

print('Loading LLM...')
LLM_NAME  = 'google/flan-t5-large'
tokenizer = T5Tokenizer.from_pretrained(LLM_NAME)
llm       = T5ForConditionalGeneration.from_pretrained(
    LLM_NAME,
    device_map='auto'
)

# ── CLIP Food Labels ──────────────────────────────────────────────────────
FOOD_LABELS = [
    'pizza', 'burger', 'hamburger', 'sandwich', 'hot dog', 'sushi',
    'fried chicken', 'grilled chicken', 'chicken curry', 'butter chicken',
    'biryani', 'fried rice', 'noodles', 'pasta', 'spaghetti', 'lasagna',
    'tacos', 'burrito', 'quesadilla', 'nachos', 'fish and chips',
    'steak', 'grilled fish', 'shrimp', 'soup', 'stew', 'omelette',
    'pancakes', 'waffles', 'french toast', 'scrambled eggs', 'boiled eggs',
    'idli', 'dosa', 'samosa', 'dal', 'palak paneer', 'paneer tikka',
    'roti', 'naan', 'paratha', 'upma', 'poha',
    'french fries', 'onion rings', 'popcorn', 'chips', 'spring rolls',
    'dim sum', 'dumplings', 'kebab', 'cookies', 'chocolate chip cookies',
    'biscuits', 'brownie',
    'apple', 'banana', 'orange', 'mango', 'grapes', 'strawberry',
    'watermelon', 'pineapple', 'avocado', 'blueberries', 'peach',
    'salad', 'broccoli', 'carrot', 'tomato', 'cucumber', 'corn',
    'spinach', 'mushroom', 'bell pepper', 'cauliflower', 'potato',
    'ice cream', 'cake', 'cupcake', 'donut', 'croissant', 'muffin',
    'bread', 'toast', 'cheese', 'yogurt', 'butter',
    'coffee', 'tea', 'juice', 'smoothie', 'milkshake',
    'rice', 'oatmeal', 'cereal', 'lentils', 'chickpeas', 'beans'
]
LABEL_PROMPTS = [f'a photo of {label}' for label in FOOD_LABELS]

# Pre-compute text embeddings
with torch.no_grad():
    text_inputs = clip_processor(text=LABEL_PROMPTS, return_tensors='pt',
                                 padding=True, truncation=True).to(DEVICE)
    raw_out     = clip_model.text_model(**text_inputs)
    text_embeds = clip_model.text_projection(raw_out.pooler_output)
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

print('✅ All assets loaded')


# ── Pipeline Functions ────────────────────────────────────────────────────
def predict_food(image: Image.Image, top_n: int = 5):
    with torch.no_grad():
        inputs       = clip_processor(images=image, return_tensors='pt').to(DEVICE)
        raw_out      = clip_model.vision_model(**inputs)
        image_embeds = clip_model.visual_projection(raw_out.pooler_output)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        similarity   = (image_embeds @ text_embeds.T).squeeze(0) * 100.0
        probs        = similarity.softmax(dim=-1).cpu().numpy()
    top_idx = probs.argsort()[::-1][:top_n]
    return [{'food': FOOD_LABELS[i], 'confidence': float(probs[i])} for i in top_idx]


def get_nutrition_summary(food_query: str) -> str:
    query    = food_query.lower().strip()
    keywords = query.split()
    mask     = nutrition_wide['food_name_clean'].str.contains(keywords[0], na=False)
    for kw in keywords[1:]:
        mask &= nutrition_wide['food_name_clean'].str.contains(kw, na=False)
    results = nutrition_wide[mask]
    if results.empty:
        mask    = nutrition_wide['food_name_clean'].str.contains(keywords[0], na=False)
        results = nutrition_wide[mask]
    if results.empty:
        return f'No USDA nutrition data found for "{food_query}".'
    row = results.iloc[0]
    def v(col):
        val = row.get(col)
        return f'{val:.1f}' if pd.notna(val) else 'N/A'
    return f"""NUTRITION (per 100g): {row['food_name']}
Category: {row.get('category_name', 'Unknown')}
Calories: {v('calories_kcal')} kcal | Protein: {v('protein_g')}g | Fat: {v('fat_g')}g
Carbs: {v('carbs_g')}g | Fiber: {v('fiber_g')}g | Sugar: {v('sugar_g')}g
Sodium: {v('sodium_mg')}mg | Cholesterol: {v('cholesterol_mg')}mg
Calcium: {v('calcium_mg')}mg | Iron: {v('iron_mg')}mg
Vitamin C: {v('vitamin_c_mg')}mg | Vitamin A: {v('vitamin_a_mcg')}mcg"""


def get_recipe_context(food_name: str, top_k: int = 3) -> str:
    query_embed = embed_model.encode([food_name], convert_to_numpy=True)
    faiss.normalize_L2(query_embed)
    scores, indices = faiss_index.search(query_embed, top_k)
    results  = recipes_df.iloc[indices[0]].copy()
    context  = f'RELEVANT RECIPES FOR "{food_name}":\n'
    for _, row in results.iterrows():
        context += f'- {row["title"]}: {str(row["ingredients_text"])[:150]}\n'
    return context


def ask_llm(question: str, food_name: str, nutrition_ctx: str, recipe_ctx: str) -> str:
    prompt = f"""You are a helpful nutrition assistant. Use the context below to answer the question.

IDENTIFIED FOOD: {food_name}

{nutrition_ctx}

{recipe_ctx}

USER QUESTION: {question}

Give a clear, helpful, and specific answer based on the nutrition data and recipes above.
Answer:"""
    inputs = tokenizer(prompt, return_tensors='pt',
                       max_length=512, truncation=True).to(DEVICE)
    with torch.no_grad():
        output_ids = llm.generate(
            **inputs,
            max_new_tokens=250,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


# ── App State ─────────────────────────────────────────────────────────────
app_state = {'food_name': None, 'nutrition': None, 'recipes': None}


def analyze_image(image):
    if image is None:
        return '## ⚠️ Please upload a food image first.', '', '', []
    try:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert('RGB')
        else:
            image = image.convert('RGB')

        predictions = predict_food(image, top_n=5)
        food_name   = predictions[0]['food']
        confidence  = predictions[0]['confidence'] * 100
        nutrition   = get_nutrition_summary(food_name)
        recipes     = get_recipe_context(food_name, top_k=3)

        app_state['food_name'] = food_name
        app_state['nutrition'] = nutrition
        app_state['recipes']   = recipes

        pred_table = [[p['food'].title(), f"{p['confidence']*100:.1f}%"]
                      for p in predictions]
        header = f'## 🍽️ Identified: **{food_name.title()}** ({confidence:.1f}% confidence)'
        return header, nutrition, recipes, pred_table
    except Exception as e:
        return f'## ❌ Error: {str(e)}', '', '', []


def chat(user_message, history):
    if not user_message or not user_message.strip():
        return history
    if app_state['food_name'] is None:
        response = '⚠️ Please upload and analyze a food image first!'
    else:
        try:
            response = ask_llm(
                question      = user_message,
                food_name     = app_state['food_name'],
                nutrition_ctx = app_state['nutrition'],
                recipe_ctx    = app_state['recipes']
            )
        except Exception as e:
            response = f'Error: {str(e)}'
    history.append({'role': 'user',      'content': user_message})
    history.append({'role': 'assistant', 'content': response})
    return history


# ── Gradio UI ─────────────────────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft(), title='Food Vision Nutrition Assistant') as demo:

    gr.Markdown("""
    # 🍽️ Multimodal Food Recognition & Nutrition Assistant
    **Upload a food photo → Get instant nutrition info + recipes + AI Q&A**
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label='📸 Upload Food Photo', type='pil', height=300)
            analyze_btn = gr.Button('🔍 Analyze Food', variant='primary', size='lg')
            food_header = gr.Markdown('Upload an image and click Analyze Food')
            pred_table  = gr.Dataframe(
                headers=['Food', 'Confidence'],
                label='Top 5 CLIP Predictions',
                interactive=False
            )
        with gr.Column(scale=1):
            with gr.Tab('🥗 Nutrition Info'):
                nutrition_box = gr.Textbox(
                    label='USDA Nutrition Data (per 100g)',
                    lines=14, interactive=False
                )
            with gr.Tab('📖 Matching Recipes'):
                recipes_box = gr.Textbox(
                    label='Top Matching Recipes from Dataset',
                    lines=14, interactive=False
                )

    gr.Markdown('---')
    gr.Markdown('## 💬 Ask the Nutrition Assistant')

    chatbot = gr.Chatbot(label='Nutrition Q&A', height=350, type='messages')
    with gr.Row():
        chat_input = gr.Textbox(
            placeholder='Ask: "Is this healthy?", "Calories in 250g?", "Healthier alternative?"...',
            label='Your Question', scale=5
        )
        chat_btn = gr.Button('Send', variant='primary', scale=1)

    gr.Examples(
        examples=[
            ['Is this food healthy for weight loss?'],
            ['How many calories if I eat 250g of this?'],
            ['Is this safe for a diabetic person?'],
            ['What nutrients am I missing if I eat only this?'],
            ['Give me a healthier recipe alternative.'],
        ],
        inputs=chat_input
    )

    analyze_btn.click(
        fn=analyze_image, inputs=[image_input],
        outputs=[food_header, nutrition_box, recipes_box, pred_table]
    )
    chat_btn.click(
        fn=chat, inputs=[chat_input, chatbot], outputs=[chatbot]
    ).then(lambda: gr.update(value=''), outputs=[chat_input])
    chat_input.submit(
        fn=chat, inputs=[chat_input, chatbot], outputs=[chatbot]
    ).then(lambda: gr.update(value=''), outputs=[chat_input])


if __name__ == '__main__':
    demo.launch()
