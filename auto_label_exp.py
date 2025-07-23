import argparse
from LLM_auto_label import run_dataset, LLMFactory

def main():
    parser = argparse.ArgumentParser(description="Run automatic labeling with LLM models.")
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

    model_map = {
        "gpt-4o-mini": "gpt-4o-mini",
        "gemini-2.0-flash": "gemini-2.0-flash",
        "claude-3.5-haiku": "claude-3-haiku-20240307",
    }

    factory = LLMFactory(model_name=model_map[args.model], together=False)
    llm = factory.get_llm()
    print(f"Running with {args.model}...")
    result = run_dataset(args.dataset, llm=llm)
    print(f"{args.model} results:")
    print(f"Total accuracy: {result['total_accuracy']}")
    print(f"Micro F1-score: {result['micro_f1']}")

if __name__ == "__main__":
    main()
