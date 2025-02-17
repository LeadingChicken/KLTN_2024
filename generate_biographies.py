from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

def generate_biography(person_name: str) -> str:
    
	# Initialize the LLM
    llm = ChatOpenAI(api_key=API_KEY, model="gpt-4o-mini")
    
    # Create the prompt template
    prompt_template = PromptTemplate(
        input_variables=["entity"],
        template="Cho tôi tiểu sử của {entity}"
    )
    
    # Generate the formatted prompt
    prompt = prompt_template.format(entity=person_name)
    
    # Get the response from the model
    response = llm.predict(prompt)
    
    return response

# # Example usage
# if __name__ == "__main__":
#     biography = generate_biography("Hồ Chí Minh")
#     print(biography)
