import os
import torch


try:
    from google.colab import userdata
    HF_TOKEN = userdata.get('HF_TOKEN')
    TYPESAFE_API_KEY = userdata.get('TYPESAFE_API_KEY')
except ImportError:
    
    from dotenv import load_dotenv
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")
    TYPESAFE_API_KEY = os.getenv("TYPESAFE_API_KEY")


MODELS = {
    "v1_135M": "HuggingFaceTB/SmolLM-135M-Instruct",
    "v2_135M": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "v2_1.7B": "HuggingFaceTB/SmolLM2-1.7B-Instruct"
}


REPO_ID = "Anthropic/model-written-evals"

DATASET_MAP = {
    "nlp": "sycophancy/sycophancy_on_nlp_survey.jsonl",
    "political": "sycophancy/sycophancy_on_political_typology_quiz.jsonl",
    "phil": "sycophancy/sycophancy_on_philpapers2020.jsonl"
}


SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


GEN_CONFIG = {
    "max_new_tokens": 10,
    "temperature": 0.0,  
    "do_sample": False
}


BASE_DIR = os.getcwd()
RESULTS_DIR = os.path.join(BASE_DIR, "results")
VISUALS_DIR = os.path.join(BASE_DIR, "visuals")


def get_output_path(model_label, dataset_label, file_type="ires"):
    """ Генерира име како: ires_SmolLM2-1.7B_political.json """
    filename = f"{file_type}_{model_label}_{dataset_label}.json"
    return os.path.join(RESULTS_DIR, filename)
