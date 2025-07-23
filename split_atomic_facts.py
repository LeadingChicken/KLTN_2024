from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from typing import List
from dotenv import load_dotenv
import os
from generate_biographies import  generate_biography
# Load api key from .env file
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")


def split_into_atomic_facts(biography_text: str) -> List[str]:
    """
    Split a biography text into atomic facts using LangChain and GPT-4.
    
    Args:
        biography_text (str): The biography text to split
        
    Returns:
        List[str]: List of atomic facts
    """
    # Initialize the ChatOpenAI model
    llm = ChatOpenAI(api_key=API_KEY, model="gpt-4o-mini")
    
    # Create the message with your custom prompt
    messages = [
        HumanMessage(content=f"""
        {biography_text}
        
        Hãy tách đoạn thông tin trên thành những atomic facts. Ví dụ bạn có đoạn văn sau:
        "Trong sự nghiệp chuyên nghiệp của mình, McCoy đã chơi cho đội Broncos, San Diego Chargers, Minnesota Vikings và Jacksonville Jaguars."
        Các atomic facts sau khi tách ra sẽ là:
        McCoy đã chơi cho đội Broncos.
		McCoy đã chơi cho đội Broncos trong sự nghiệp chuyên nghiệp của mình.
		McCoy đã chơi cho đội San Diego Chargers.
		McCoy đã chơi cho đội San Diego Chargers trong sự nghiệp chuyên nghiệp của mình.
		McCoy đã chơi cho đội Minnesota Vikings.
		McCoy đã chơi cho đội Minnesota Vikings trong sự nghiệp chuyên nghiệp của mình.
		McCoy đã chơi cho đội Jacksonville Jaguars.
		McCoy đã chơi cho đội Jacksonville Jaguars trong sự nghiệp chuyên nghiệp của mình.
        
        Hãy đưa ra format giống như ví dụ trên. Đừng thêm gạch hay chấm đầu dòng và đừng sửa thông tin.
        """)
    ]
    
    # Get the response from the model
    response = llm.invoke(messages)
    # Split the response into individual facts
    atomic_facts = [fact.strip() for fact in response.content.split('\n') if fact.strip()]
    
    return atomic_facts

