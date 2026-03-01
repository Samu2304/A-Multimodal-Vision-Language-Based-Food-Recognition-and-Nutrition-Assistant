# Multimodal Food Recognition & Nutrition Assistant


<img width="1280" height="420" alt="image" src="https://github.com/user-attachments/assets/5590514c-66b5-4ca9-bee9-bb8c604ee242" />


[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-yellow?style=for-the-badge)](https://huggingface.co/spaces/samyukthasreenivasan/food-nutrition-assistant)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Google Colab](https://img.shields.io/badge/Open%20in-Colab-orange?style=for-the-badge&logo=google-colab)](https://colab.research.google.com)



## Output

<img width="1433" height="604" alt="image" src="https://github.com/user-attachments/assets/5ba43b4f-e7c4-44c7-abba-bd6a67ef51c3" />

<img width="1490" height="603" alt="image" src="https://github.com/user-attachments/assets/a2624f26-9528-4548-87dd-8b35b4c3cf19" />

<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/47e9d275-3e02-4898-b478-21ff79ea8832" />

<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/c6761524-e958-4513-bf53-e73092e7d8c3" />
 
## Key Features

- **Zero-Shot Food Recognition** — CLIP identifies food from any photo without task-specific training
- **USDA Nutrition Lookup** — Full nutritional breakdown (14 nutrients) from 7,700+ foods
- **Semantic Recipe Search** — FAISS-powered RAG retrieves relevant recipes from 2.2M recipe dataset
- **Conversational Q&A** — Flan-T5 LLM answers health, diet, and recipe questions grounded in real data
- **Permanent Public Demo** — Deployed on HuggingFace Spaces, accessible anytime

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   User Interface (Gradio)                │
└──────────────────────────┬──────────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │   Food Photo Upload     │
              └────────────┬────────────┘
                           │
         ┌─────────────────▼──────────────────┐
         │     CLIP (openai/clip-vit-base)     │
         │     Zero-shot Food Recognition      │
         └─────────────────┬──────────────────┘
                           │ food_name + confidence
          ┌────────────────┼────────────────┐
          │                │                │
┌─────────▼──────┐ ┌───────▼───────┐ ┌─────▼────────────┐
│  USDA Nutrition │ │  FAISS Index  │ │   Flan-T5 LLM    │
│  Database Lookup│ │  Recipe RAG   │ │  Q&A Chatbot     │
│  (7,700+ foods) │ │ (2.2M recipes)│ │  (grounded)      │
└─────────┬───────┘ └───────┬───────┘ └─────┬────────────┘
          └────────────────►│◄──────────────┘
                    ┌───────▼───────┐
                    │  Final Output │
                    │  Nutrition +  │
                    │  Recipes + AI │
                    └───────────────┘
```

---

## Datasets

| Dataset | Source | Size | Purpose |
|---|---|---|---|
| `food.csv` | USDA FoodData Central (SR Legacy) | 7,793 foods | Food name & category lookup |
| `nutrient.csv` | USDA FoodData Central | 474 nutrients | Nutrient definitions & units |
| `food_nutrient.csv` | USDA FoodData Central | ~500K rows | Nutrient values per food |
| `food_category.csv` | USDA FoodData Central | 25 categories | Category labels |
| `full_dataset.csv` | Recipe Dataset | 2.2M recipes | Semantic recipe matching |

> Download USDA data free from [fdc.nal.usda.gov](https://fdc.nal.usda.gov/download-datasets) → SR Legacy CSV

---

## Models

| Model | Role | Parameters |
|---|---|---|
| `openai/clip-vit-base-patch32` | Food image recognition | 151M |
| `google/flan-t5-large` | Nutrition Q&A chatbot | 780M |
| `all-MiniLM-L6-v2` | Recipe semantic search | 22M |

---

## Notebooks

Run these in order in **Google Colab**:

| Step | Notebook | Description |
|---|---|---|
| 1 | [`Step1_Data_Pipeline.ipynb`](notebooks/Step1_Data_Pipeline.ipynb) | Merge USDA datasets → master nutrition table |
| 2 | [`Step2_Food_Image_Recognition.ipynb`](notebooks/Step2_Food_Image_Recognition.ipynb) | CLIP food recognition pipeline |
| 3 | [`Step3_Recipe_Matching_RAG.ipynb`](notebooks/Step3_Recipe_Matching_RAG.ipynb) | FAISS recipe search engine |
| 4 | [`Step4_Chatbot_and_App.ipynb`](notebooks/Step4_Chatbot_and_App.ipynb) | LLM chatbot + Gradio demo |

---

## Setup

### Option A — Use the Live Demo (Recommended)
[huggingface.co/spaces/samyukthasreenivasan/food-nutrition-assistant](https://huggingface.co/spaces/samyukthasreenivasan/food-nutrition-assistant)

No setup required!

### Option B — Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/SamyukthaSreenivasan/food-nutrition-assistant.git
cd food-nutrition-assistant

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your data files to data/
#    master_nutrition.csv
#    recipe_faiss.index
#    recipes_subset.csv

# 4. Run the app
python src/app.py
```

### Option C — Run on Google Colab
Open each notebook in the `notebooks/` folder using the Colab badge at the top of this README.

---

## Repository Structure

```
food-nutrition-assistant/
│
├── notebooks/
│   ├── Step1_Data_Pipeline.ipynb
│   ├── Step2_Food_Image_Recognition.ipynb
│   ├── Step3_Recipe_Matching_RAG.ipynb
│   └── Step4_Chatbot_and_App.ipynb
│
├── src/
│   └── app.py                     ← main Gradio application
│
├── data/
│   └── sample/                    ← sample food images for testing
│
├── assets/
│   └── screenshots/               ← app screenshots for README
│
├── docs/
│   └── ARCHITECTURE.md            ← detailed system design
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🔬 How It Works

### Step 1 — Data Pipeline
All USDA datasets are merged into a single wide-format master table with one row per food and one column per nutrient. The recipe dataset is cleaned and ingredients are parsed into structured lists.

### Step 2 — Food Recognition (CLIP)
CLIP encodes the uploaded image and 90+ food label prompts into a shared embedding space. Cosine similarity with temperature scaling identifies the best matching food label.

### Step 3 — Recipe Matching (FAISS RAG)
All 50,000 recipe texts are encoded using `all-MiniLM-L6-v2` and stored in a FAISS flat index. At query time, the food name is encoded and the top-K most semantically similar recipes are retrieved in milliseconds.

### Step 4 — Nutrition Q&A (Flan-T5)
The identified food name, USDA nutrition data, and matched recipes are assembled into a structured prompt. Flan-T5 generates grounded, factual answers to any nutrition or recipe question.

---

## Results

| Metric | Value |
|---|---|
| Food categories supported | 90+ |
| USDA foods in database | 7,793 |
| Recipes indexed | 50,000 |
| Nutrients tracked per food | 14 |
| Average inference time (CPU) | ~3s |

---

## Future Work

- [ ] Fine-tune CLIP on a dedicated food image dataset (Food-101, Food-500)
- [ ] Add meal tracking across a full day
- [ ] Personalised health scoring based on user profile
- [ ] Ingredient-level breakdown from dish photos
- [ ] Multilingual support for regional foods

---

## Author

**Samyuktha S**
- GitHub: [@SamyukthaSreenivasan](https://github.com/Samu04)
- HuggingFace: [@samyukthasreenivasan](https://huggingface.co/samyukthasreenivasan)

---

##  License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [USDA FoodData Central](https://fdc.nal.usda.gov/) for the nutrition database
- [OpenAI CLIP](https://openai.com/research/clip) for the vision-language model
- [Google Flan-T5](https://huggingface.co/google/flan-t5-large) for the language model
- [HuggingFace](https://huggingface.co) for model hosting and Spaces deployment
