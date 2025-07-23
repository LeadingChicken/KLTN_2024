from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
import random
import wikipedia
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
import numpy as np
from wikipedia_pull_data import get_wikipedia_summary
os.environ["CHROMA_TELEMETRY"] = "FALSE"

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
            return ChatOpenAI(api_key=self.api_key, model=self.model_name)
        
        elif "gemini" in self.model_name:
            from langchain_google_genai import ChatGoogleGenerativeAI
            if not self.api_key:
                self.api_key = os.getenv("GEMINI_API_KEY")
            if not self.api_key:
                raise ValueError("Google API key not found.")
            return ChatGoogleGenerativeAI(google_api_key=self.api_key, model=self.model_name)
        
        elif "claude" in self.model_name:
            from langchain_anthropic import ChatAnthropic
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
model1 = "claude-3-haiku-20240307"
model2 = "deepseek-ai/DeepSeek-R1-0528-tput"
model3 = "gemini-1.5-flash"
model4 = "Qwen/Qwen3-235B-A22B-fp8-tput"
model5 = "gpt-4o-mini"
llm_factory = LLMFactory(model_name=model3, together=False)
llm = llm_factory.get_llm()

def get_wikipedia_summary(title):
    url = "https://vi.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "titles": title,
        "explaintext": True
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    # Lấy nội dung từ kết quả trả về
    pages = data.get("query", {}).get("pages", {})
    for page_id, page_content in pages.items():
        if "extract" in page_content:
            return page_content["extract"]
    return "Không tìm thấy thông tin."

def get_wikipedia_content(person_name):
    try:
        # Search for the most relevant Wikipedia page
        search_results = wikipedia.search(person_name)
        if not search_results:
            return f"Wikipedia search for '{person_name}': No relevant page found."
        page_title = search_results[0]
        page = wikipedia.page(page_title, auto_suggest=False)
        content = page.content
        return content
    except Exception as e:
        print(f"Wikipedia search error for '{person_name}': {e}")
        return f"Wikipedia search for '{person_name}': Information not available due to errors. Error: {str(e)}"

def create_vector_database(person_name):
    # Get Wikipedia content for the person
    content = get_wikipedia_summary(person_name)
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(content)
    
    # Create vector database using Chroma
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_texts(chunks, embeddings)
    
    return vectorstore, content

def rerank_with_bge_m3(query, documents, top_k=5):
    # Load BGE-M3 model
    model = SentenceTransformer('BAAI/bge-m3')
    
    # Encode query and documents
    query_embedding = model.encode([query], normalize_embeddings=True)
    doc_embeddings = model.encode(documents, normalize_embeddings=True)
    
    # Calculate similarities
    similarities = np.dot(doc_embeddings, query_embedding.T).flatten()
    
    # Get top-k documents
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    return [documents[i] for i in top_indices], [similarities[i] for i in top_indices]

def get_rag_context(vectorstore, query, person_name, top_k=15):
    # Retrieve relevant documents from vector store
    docs = vectorstore.similarity_search(query, k=top_k)
    documents = [doc.page_content for doc in docs]
    
    # Re-rank with BGE-M3
    reranked_docs, scores = rerank_with_bge_m3(query, documents, top_k=3)
    
    # Combine reranked documents
    context = "\n\n".join(reranked_docs)
    return context

rag_template = """
You are given a context (Wikipedia) followed by an atomic fact. You job is to label whether this atomic fact is Supported, Unsupported, Irrelevant. The context is about a person or a character and the atomic fact is a piece of information about that person.

Supported: Choose this if the atomic fact is true and fully supported by the context.
Unsupported: Choose this if the atomic fact is false and contradict with the context.
Irrelevant: Choose this if the atomic fact is irrelevant to the person or the atomic fact talking about another thing that do not have connection with the person.

This is your part:
Character: {character}
Context from Wikipedia: {context}

atomic fact: {fact}

Return with the format:
Label: <Your label>
Reason: <Your reason for this label>
Context: {context}
"""

prompt = PromptTemplate(
    input_variables=["character", "context", "fact"],
    template=rag_template
)

rag_chain = prompt | llm | StrOutputParser()

def get_rag_response(query, topic, vectorstore):
    try:
        context = get_rag_context(vectorstore, query, topic)
        print(f"Context: {context}")
    except Exception as e:
        print(f"RAG retrieval failed for '{query}': {e}")
        context = f"RAG retrieval for '{query}': Unable to retrieve information at this time. Error: {str(e)}"
    print(f"Context: {context}")
    response = rag_chain.invoke({
        "character": topic,
        "context": context,
        "fact": query,
    })
    try:
        label = response.split("Label:")[1].split("\n")[0].strip()
    except:
        label = "Error parsing label"
    return {"full_response": response, "label": label}

def factscore(entity_name):
    dataset_src = "Dataset/"
    df = pd.read_csv(f"{dataset_src}{entity_name}.csv")
    atomic_facts = df.iloc[:, 0].tolist()
    labels = df.iloc[:, 1].tolist()
    predicted_labels = []
    full_responses = []
    
    # Create vector database for the person
    print(f"Creating vector database for {entity_name}...")
    vectorstore, full_content = create_vector_database(entity_name)
    print(f"Vector database created with {len(full_content)} characters of content")
    
    for i, atomic_fact in tqdm(enumerate(atomic_facts), total=len(atomic_facts)):
        print(f"Processing atomic fact: {atomic_fact}")
        label = labels[i]
        print(f"True Label: {label}")
        topic = entity_name
        question = atomic_fact
        result = get_rag_response(question, topic, vectorstore)
        predicted_labels.append(result['label'])
        print(f"Predicted label: {result['label']}")
        full_responses.append(result['full_response'])
    df['Predicted Label'] = predicted_labels
    df['Full Response'] = full_responses
    df.rename(columns={df.columns[0]: 'Atomic Fact'}, inplace=True)
    df.rename(columns={df.columns[1]: 'True Label'}, inplace=True)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed:')]
    result_src = "Results_wiki/VIE/Gemini 1.5 Flash/"
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
