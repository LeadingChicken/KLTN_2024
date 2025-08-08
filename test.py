from split_atomic_facts import split_into_atomic_facts
from generate_biographies import generate_biography

def test_atomic_fact_splitting():
    """
    Test the atomic fact splitting function by generating biography using TinyLLaMA
    and writing atomic facts to a text file
    """
    
    # File paths
    output_file = "atomic_facts_output.txt"
    
    try:
        # Generate biography using TinyLLaMA
        print("Generating biography using TinyLLaMA...")
        person_name = "Taylor Swift"  # You can change this to any person
        biography_text = generate_biography(person_name)
        
        print("=== Generated Biography Content ===")
        print(biography_text)
        print("\n" + "="*50 + "\n")
        
    #     # Call the function to split into atomic facts
    #     print("Processing biography to extract atomic facts...")
    #     atomic_facts = split_into_atomic_facts(biography_text)
        
    #     # Write atomic facts to output file
    #     print(f"Writing atomic facts to: {output_file}")
    #     with open(output_file, 'w', encoding='utf-8') as f:
    #         for fact in atomic_facts:
    #             f.write(f"{fact}\n")
        
    #     print("Extracted Atomic Facts:")
    #     print(f"Total facts found: {len(atomic_facts)}")
        
    #     # Also write the original biography to a file for reference
    #     with open("generated_biography.txt", 'w', encoding='utf-8') as f:
    #         f.write(f"Biography of {person_name}:\n")
    #         f.write("="*50 + "\n")
    #         f.write(biography_text)
        
    #     print(f"✅ Biography written to: generated_biography.txt")
    #     print(f"✅ Atomic facts written to: {output_file}")
            
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_atomic_fact_splitting()
