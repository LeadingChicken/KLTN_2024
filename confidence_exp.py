import argparse
import warnings
warnings.filterwarnings("ignore")
from LLM_auto_label import get_response_from_prompt, LLMFactory
from uqlm import BlackBoxUQ, LLMPanel
import os
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
os.environ["GEMINI_API_KEY"] = GOOGLE_API_KEY
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


def main():
    parser = argparse.ArgumentParser(description="Calculate confidence for atomic facts and print out the facts that need to be rechecked.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "gpt-4o-mini",
            "gemini-2.0-flash",
            "claude-3.5-haiku",
        ],
        help="Choose model to run (gpt-4o-mini, gemini-2.0-flash, claude-3.5-haiku)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Dataset/",
        help="Path to dataset directory (default: Dataset/)",
    )
    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    dataset_src = args.dataset
    csv_files = [f for f in os.listdir(dataset_src) if f.endswith('.csv')]
    df_list = []
    for csv in csv_files:
        entity_name = csv.split('.')[0]
        df = pd.read_csv(f"{dataset_src}{csv}")
        df['entity'] = entity_name
        df_list.append(df)
    df = pd.concat(df_list, ignore_index=True)
    atomic_facts = df['atomic_facts'].tolist()
    labels = df['label'].tolist()
    prompts = df['prompt'].to_list()

    # Load LLM
    model_map = {
        "gpt-4o-mini": "gpt-4o-mini",
        "gemini-2.0-flash": "gemini-2.0-flash",
        "claude-3.5-haiku": "claude-3-haiku-20240307",
    }
    llm_factory = LLMFactory(model_name=model_map[args.model], together=False)
    llm = llm_factory.get_llm()

    # Black box UQ
    print("Processing black box...")
    bbuq = BlackBoxUQ(llm=llm, scorers=["semantic_negentropy"], use_best=True, max_calls_per_min=50)
    import asyncio
    async def run_confidence():
        results_BB = await bbuq.generate_and_score(prompts=list(tqdm(prompts, desc="BlackBoxUQ")), num_responses=5)
        print("Processing LLM as a Judge...")
        panel = LLMPanel(llm=llm, judges=[llm]*5, scoring_templates=['true_false_uncertain']*5, max_calls_per_min = 50)
        results_judge = await panel.generate_and_score(prompts=list(tqdm(prompts, desc="LLMPanel")))
        result_BB = results_BB.to_df()
        result_BB['atomic_fact'] = atomic_facts
        result_judge = results_judge.to_df()
        result_judge['atomic_fact'] = atomic_facts
        result = ((result_judge['avg'] + result_BB['semantic_negentropy'])/2).tolist()
        # Print atomic facts that have average score < 0.8
        count = 0
        for i, v in enumerate(result):
            if v < 0.8:
                count += 1
                print(f"Recheck atomic fact: {atomic_facts[i]}")
        print("Total atomic facts to recheck:", count)
        print("Total atomic facts:", len(atomic_facts))
    asyncio.run(run_confidence())

if __name__ == "__main__":
    main()
