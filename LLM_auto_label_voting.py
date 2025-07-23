from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
import random
from collections import Counter

# Load API key from .env file
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

class LLMFactory:
    def __init__(self, model_name, api_key=None, together=False):
        self.model_name = model_name
        self.api_key = api_key
        self.together = together
        self.llm = self._initialize_llm()

    def _initialize_llm(self):
        if self.together:
            base_url = "https://api.together.xyz/v1"
            api_key = self.api_key or TOGETHER_API_KEY
            if not api_key:
                raise ValueError("Together.ai API key not found.")
            return ChatOpenAI(api_key=api_key, base_url=base_url, model=self.model_name)

        if "gpt" in self.model_name:
            if not self.api_key:
                self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OpenAI API key not found.")
            return ChatOpenAI(api_key=self.api_key, model=self.model_name)
        
        elif "gemini" in self.model_name:
            if not self.api_key:
                self.api_key = os.getenv("GEMINI_API_KEY")
            if not self.api_key:
                raise ValueError("Google API key not found.")
            return ChatGoogleGenerativeAI(google_api_key=self.api_key, model=self.model_name)
        
        elif "claude" in self.model_name:
            if not self.api_key:
                self.api_key = os.getenv("ANTHROPIC_API_KEY")
            if not self.api_key:
                raise ValueError("Anthropic API key not found.")
            return ChatAnthropic(anthropic_api_key=self.api_key, model_name=self.model_name)

        elif "Qwen" in self.model_name:
            base_url = "https://api.together.xyz/v1"
            api_key = self.api_key or TOGETHER_API_KEY
            if not api_key:
                raise ValueError("Together.ai API key not found for Qwen model.")
            return ChatOpenAI(api_key=api_key, base_url=base_url, model=self.model_name)
        elif "Llama" in self.model_name:
            base_url = "https://api.together.xyz/v1"
            api_key = self.api_key or TOGETHER_API_KEY
            if not api_key:
                raise ValueError("Together.ai API key not found for Llama model.")
            return ChatOpenAI(api_key=api_key, base_url=base_url, model=self.model_name)
        elif "deepseek" in self.model_name:
            base_url = "https://api.together.xyz/v1"
            api_key = self.api_key or TOGETHER_API_KEY
            if not api_key:
                raise ValueError("Together.ai API key not found for DeepSeek model.")
            return ChatOpenAI(api_key=api_key, base_url=base_url, model=self.model_name)
        else:
            raise ValueError(f"Unsupported LLM model: {self.model_name}")

    def get_llm(self):
        return self.llm

# Define model names for easy switching
model_gpt = "gpt-4o-mini"
model_gemini = "gemini-1.5-flash"
model_claude = "claude-3-haiku-20240307"
model_qwen = "Qwen/Qwen3-235B-A22B-fp8-tput"
model_deepseek = "deepseek-ai/DeepSeek-R1-0528-tput"

# Initialize 3 LLMs (can be any combination)
llm_factory_1 = LLMFactory(model_name=model_gpt, together=False)
llm_factory_2 = LLMFactory(model_name=model_gemini, together=False)
llm_factory_3 = LLMFactory(model_name=model_claude, together=False)
llm1 = llm_factory_1.get_llm()
llm2 = llm_factory_2.get_llm()
llm3 = llm_factory_3.get_llm()

# Tavily search (only once per fact)
tavily_search = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=3)

def get_search_results(query, max_retries=3):
    for attempt in range(max_retries):
        try:
            time.sleep(random.uniform(1, 2))
            return tavily_search.invoke({"query": query})
        except Exception as e:
            print(f"Tavily Search error, attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt + random.uniform(1, 2))
                continue
            else:
                return f"Search results for '{query}': Information not available due to repeated errors. Error: {str(e)}"
    return f"Search results for '{query}': Information not available due to repeated errors."

rag_template = """
Bạn được cung cấp một ngữ cảnh (kết quả tìm kiếm web) kèm theo một atomic fact. Nhiệm vụ của bạn là gán nhãn cho atomic fact này là Supported, Unsupported, hoặc Irrelevant. Ngữ cảnh nói về một người hoặc một nhân vật và atomic fact là một thông tin về người đó.

Supported: Chọn nhãn này nếu atomic fact là đúng và được ngữ cảnh hoàn toàn xác thực.
Unsupported: Chọn nhãn này nếu atomic fact là sai và mâu thuẫn với ngữ cảnh.
Irrelevant: Chọn nhãn này nếu atomic fact không liên quan đến người hoặc nhân vật, hoặc atomic fact nói về một điều khác không có liên hệ với người đó.

Đây là phần bạn cần làm:
Nhân vật: {character}
Ngữ cảnh từ kết quả tìm kiếm web: {context}

atomic fact: {fact}

Trả lời theo định dạng sau:
Label: <Nhãn của bạn>
Reason: <Lý do cho nhãn này>
"""

prompt = PromptTemplate(
    input_variables=["character", "context", "fact"],
    template=rag_template
)

def get_llm_label(llm, character, context, fact):
    rag_chain = prompt | llm | StrOutputParser()
    response = rag_chain.invoke({
        "character": character,
        "context": context,
        "fact": fact,
    })
    try:
        label = response.split("Label:")[1].split("\n")[0].strip()
    except:
        label = "Error parsing label"
    return label, response

def get_voting_response(query, topic):
    search_query = f"{topic} {query}"
    try:
        context = get_search_results(search_query)
    except Exception as e:
        print(f"Search failed for '{search_query}': {e}")
        context = f"Search results for '{search_query}': Unable to retrieve information at this time. Error: {str(e)}"
    print(f"Context: {context}")
    labels = []
    responses = []
    for llm in [llm1, llm2, llm3]:
        label, response = get_llm_label(llm, topic, context, query)
        labels.append(label)
        responses.append(response)
    # Majority voting
    label_counts = Counter(labels)
    majority_label = label_counts.most_common(1)[0][0]
    return {"labels": labels, "majority_label": majority_label, "responses": responses}

def factscore(entity_name):
    dataset_src = "Dataset/"
    df = pd.read_csv(f"{dataset_src}{entity_name}.csv")
    atomic_facts = df.iloc[:, 0].tolist()
    labels = df.iloc[:, 1].tolist()
    predicted_labels = []
    all_llm_labels = []
    all_responses = []
    llm1_labels = []
    llm2_labels = []
    llm3_labels = []
    for i, atomic_fact in tqdm(enumerate(atomic_facts), total=len(atomic_facts)):
        print(f"Processing atomic fact: {atomic_fact}")
        label = labels[i]
        print(f"True Label: {label}")
        topic = entity_name
        question = atomic_fact
        result = get_voting_response(question, topic)
        predicted_labels.append(result['majority_label'])
        all_llm_labels.append(result['labels'])
        all_responses.append(result['responses'])
        # Print each model's decision
        print(f"LLM1 Decision: {result['labels'][0]}")
        print(f"LLM2 Decision: {result['labels'][1]}")
        print(f"LLM3 Decision: {result['labels'][2]}")
        print(f"Predicted label (majority): {result['majority_label']}")
        llm1_labels.append(result['labels'][0])
        llm2_labels.append(result['labels'][1])
        llm3_labels.append(result['labels'][2])
    df['Predicted Label'] = predicted_labels
    df['LLM1 Label'] = llm1_labels
    df['LLM2 Label'] = llm2_labels
    df['LLM3 Label'] = llm3_labels
    df['LLM Labels'] = all_llm_labels
    df['LLM Responses'] = all_responses
    df.rename(columns={df.columns[0]: 'Atomic Fact'}, inplace=True)
    df.rename(columns={df.columns[1]: 'True Label'}, inplace=True)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed:')]
    result_src = "Results/Voting/"
    os.makedirs(result_src, exist_ok=True)
    df.to_csv(f"{result_src}{entity_name}_auto_labeled.csv", index=False, encoding='utf-8-sig')
    correct_predictions = 0
    for i, predicted_label in enumerate(predicted_labels):
        if predicted_label == labels[i]:
            correct_predictions += 1
    accuracy = correct_predictions / len(atomic_facts)
    result_map = {
        'total_facts': len(atomic_facts),
        'supported_count': predicted_labels.count('Supported'),
        'unsupported_count': predicted_labels.count('Unsupported'),
        'irrelevant_count': predicted_labels.count('Irrelevant'),
        'accuracy': accuracy
    }
    return result_map

def run_dataset(dataset_src):
    csv_files = [f for f in os.listdir(dataset_src) if f.endswith('.csv')]
    accuracy_dict = {}
    for csv_file in csv_files:
        entity_name = csv_file.split('.')[0]
        print(f"Processing entity: {entity_name}")
        result_map = factscore(entity_name)
        print(f"Result map: {result_map}")
        accuracy_dict[entity_name] = result_map['accuracy']
    average_accuracy = sum(accuracy_dict.values()) / len(accuracy_dict)
    print(f"Average accuracy: {average_accuracy}")
    return accuracy_dict
