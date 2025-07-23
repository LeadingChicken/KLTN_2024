from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
import random
from langchain_anthropic import ChatAnthropic
from sklearn.metrics import f1_score

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
    def __init__(self, model_name="Qwen/Qwen3-235B-A22B-fp8-tput", api_key=None, together=False):
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
            return ChatOpenAI(api_key=self.api_key, model=self.model_name, temperature = 0)
        
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

# Initialize the LLM 
llm_factory = LLMFactory(model_name="gpt-4o-mini", together=False)
llm = llm_factory.get_llm()

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

def extract_main_text_from_url(url, max_chars=2000):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        for script in soup(["script", "style", "noscript"]):
            script.decompose()
        text = ' '.join(soup.stripped_strings)
        return text[:max_chars]
    except Exception as e:
        return f"[Could not extract content from {url}: {e}]"

rag_template = """
Bạn được cung cấp một ngữ cảnh (kết quả tìm kiếm web) kèm theo một atomic fact. Nhiệm vụ của bạn là gán nhãn cho atomic fact này là Supported, Unsupported, hoặc Irrelevant. Ngữ cảnh nói về một người hoặc một nhân vật và atomic fact là một thông tin về người đó.

Supported: Chọn nhãn này nếu atomic fact là đúng và được ngữ cảnh hoàn toàn xác thực.
Unsupported: Chọn nhãn này nếu atomic fact là sai và mâu thuẫn với ngữ cảnh.
Irrelevant: Chọn nhãn này nếu atomic fact không liên quan đến người hoặc nhân vật, hoặc atomic fact nói về một điều khác không có liên hệ với người đó.

LƯU Ý: thông tin tìm kiếm được trên trang web có thể chứa tin giả và tin sai sự thật. Hãy cân nhắc kĩ lưỡng trước khi ra quyết định của mình.
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

rag_chain = prompt | llm | StrOutputParser()

def get_response(query, topic):
    search_query = f"{topic} {query}"
    try:
        search_results = get_search_results(search_query)
    except Exception as e:
        print(f"Search failed for '{search_query}': {e}")
        search_results = f"Search results for '{search_query}': Unable to retrieve information at this time. Error: {str(e)}"
    context = search_results
    created_prompt = prompt.invoke({
        "character": topic,
        "context": context,
        "fact": query,
    })
    response = rag_chain.invoke({
        "character": topic,
        "context": context,
        "fact": query,
    })
    try:
        label = response.split("Label:")[1].split("\n")[0].strip()
    except:
        label = "Error parsing label"
    return {"full_response": response, "label": label, "prompt" : created_prompt, "context": context}

def get_response_from_prompt(prompt_string,llm = llm):
    """
    Takes a pre-crafted prompt string, sends it to the LLM, and parses the response.
    """
    response_obj = llm.invoke(prompt_string)
    response = response_obj.content  # Extract the raw string from the AIMessage
    try:
        label = response.split("Label:")[1].split("\n")[0].strip()
    except Exception:
        label = "Error parsing label"
    return {"full_response": response, "label": label, "prompt": prompt_string}

def factscore(entity_name):
    dataset_src = "New Dataset/"
    df = pd.read_csv(f"{dataset_src}{entity_name}.csv")
    atomic_facts = df['atomic_facts'].tolist()
    labels = df['label'].tolist()
    prompts = df['prompt'].tolist()
    predicted_labels = []
    full_responses = []
    for i, atomic_fact in tqdm(enumerate(atomic_facts), total=len(atomic_facts)):
        print(f"Processing atomic fact: {atomic_fact}")
        label = labels[i]
        print(f"Label: {label}")
        topic = entity_name
        question = atomic_fact
        result = get_response_from_prompt(prompts[i])
        predicted_labels.append(result['label'])
        print(f"Predicted label: {result['label']}")
        full_responses.append(result['full_response'])
    # df['Predicted Label'] = predicted_labels
    # df['Full Response'] = full_responses
    # df.rename(columns={df.columns[0]: 'Atomic Fact'}, inplace=True)
    # df.rename(columns={df.columns[1]: 'True Label'}, inplace=True)
    # df = df.loc[:, ~df.columns.str.contains('^Unnamed:')]
    # result_src = "Results/Deepseek-r1/"
    # df.to_csv(f"{result_src}{entity_name}_auto_labeled.csv", index=False, encoding='utf-8-sig')
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

def run_dataset(dataset_src,llm=llm):
    csv_files = [f for f in os.listdir(dataset_src) if f.endswith('.csv')]
    accuracy_dict = {}
    all_ground_truth = []
    all_predicted = []
    for csv_file in csv_files:
        entity_name = csv_file.split('.')[0]
        print(f"Processing entity: {entity_name}")
        # Load data
        df = pd.read_csv(f"{dataset_src}{entity_name}.csv")
        atomic_facts = df['atomic_facts'].tolist()
        labels = df['label'].tolist()
        prompts = df['prompt'].tolist()
        predicted_labels = []
        for i, atomic_fact in tqdm(enumerate(atomic_facts), total=len(atomic_facts)):
            print(f"Processing atomic fact: {atomic_fact}")
            result = get_response_from_prompt(prompts[i])
            predicted_labels.append(result['label'])
        # Collect for global lists
        all_ground_truth.extend(labels)
        all_predicted.extend(predicted_labels)
        # Per-entity accuracy
        correct_predictions = sum(1 for i in range(len(labels)) if predicted_labels[i] == labels[i])
        accuracy = correct_predictions / len(labels)
        accuracy_dict[entity_name] = accuracy
    # Global accuracy
    total_correct = sum(1 for i in range(len(all_ground_truth)) if all_predicted[i] == all_ground_truth[i])
    total_accuracy = total_correct / len(all_ground_truth) if all_ground_truth else 0.0
    # Micro F1-score
    micro_f1 = f1_score(all_ground_truth, all_predicted, labels=['Supported', 'Unsupported', 'Irrelevant'], average='micro')
    print(f"Micro F1-score: {micro_f1}")
    return {
        'accuracy_dict': accuracy_dict,
        'all_ground_truth': all_ground_truth,
        'all_predicted': all_predicted,
        'total_accuracy': total_accuracy,
        'micro_f1': micro_f1
    }

def count_facts_and_labels(dataset_src):
    """
    Count the total number of atomic facts and the number of Supported, Unsupported, and Irrelevant labels in the dataset.
    """
    csv_files = [f for f in os.listdir(dataset_src) if f.endswith('.csv')]
    total_facts = 0
    supported_count = 0
    unsupported_count = 0
    irrelevant_count = 0
    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(dataset_src, csv_file))
        labels = df['label'].tolist()
        total_facts += len(labels)
        supported_count += labels.count('Supported')
        unsupported_count += labels.count('Unsupported')
        irrelevant_count += labels.count('Irrelevant')
    return {
        'total_facts': total_facts,
        'supported_count': supported_count,
        'unsupported_count': unsupported_count,
        'irrelevant_count': irrelevant_count
    }

