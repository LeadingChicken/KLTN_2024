from LLM_auto_label import run_dataset, LLMFactory

def main():
    dataset_src = 'New Dataset/'
    # gpt-4o-mini
    # gpt_factory = LLMFactory(model_name="gpt-4o-mini", together=False)
    # gpt_llm = gpt_factory.get_llm()
    # print("Running with gpt-4o-mini...")
    # gpt_result = run_dataset(dataset_src, llm=gpt_llm)
    # print("gpt-4o-mini results:")
    # print(f"Total accuracy: {gpt_result['total_accuracy']}")
    # print(f"Micro F1-score: {gpt_result['micro_f1']}")

    # # gemini-2.0-flash
    # gemini_factory = LLMFactory(model_name="gemini-2.0-flash", together=False)
    # gemini_llm = gemini_factory.get_llm()
    # print("Running with gemini-2.0-flash...")
    # gemini_result = run_dataset(dataset_src, llm=gemini_llm)
    # print("gemini-2.0-flash results:")
    # print(f"Total accuracy: {gemini_result['total_accuracy']}")
    # print(f"Micro F1-score: {gemini_result['micro_f1']}")

    # claude-3-haiku
    claude_factory = LLMFactory(model_name="claude-3-haiku-20240307", together=False)
    claude_llm = claude_factory.get_llm()
    print("Running with claude-3.5-haiku...")
    claude_result = run_dataset(dataset_src, llm=claude_llm)
    print("claude-3-haiku results")
    print(f"Total accuracy: {claude_result['total_accuracy']}")
    print(f"Micro F1-score: {claude_result['micro_f1']}")

if __name__ == "__main__":
    main()
