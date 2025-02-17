from generate_biographies import generate_biography
from split_atomic_facts import split_into_atomic_facts
import warnings
from LLM_auto_label import factscore
warnings.filterwarnings('ignore')
if __name__=="__main__":
	# Read entities from file
	with open('entities.txt', 'r', encoding='utf-8') as f:
		entities = [line.strip() for line in f if line.strip()]
	
	# Process each entity
	for entity_name in entities:
		print(f"\nProcessing entity: {entity_name}")
		try:
			generated_text = generate_biography(entity_name)
			atomic_facts = split_into_atomic_facts(generated_text)
			
			# Write atomic facts to a file named after the entity
			output_file = 'atomic_facts.txt'
			with open(output_file, 'w', encoding='utf-8') as f:
				for fact in atomic_facts:
					f.write(fact + '\n')
			
			result = factscore(entity_name)
			print(f"Results for {entity_name}:")
			print(result)
			
		except Exception as e:
			print(f"Error processing {entity_name}: {str(e)}")
	