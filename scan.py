from LLM_auto_label import count_facts_and_labels

def main():
    dataset_src = 'New Dataset/'
    counts = count_facts_and_labels(dataset_src)
    print(f"Total atomic facts: {counts['total_facts']}")
    print(f"Supported: {counts['supported_count']}")
    print(f"Unsupported: {counts['unsupported_count']}")
    print(f"Irrelevant: {counts['irrelevant_count']}")

if __name__ == "__main__":
    main()