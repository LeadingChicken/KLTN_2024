from LLM_auto_label_wiki import run_dataset,factscore
# Snuff all warnings
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

if __name__ == '__main__':
				entity_name = "Sơn Tùng M-TP"
				print("Running dataset with Gemini 1.5 Flash")	
				accuracy_dict = factscore(entity_name)
				print(f"Accuracy dict: {accuracy_dict}")
				print("Done with Gemini 1.5 Flash")