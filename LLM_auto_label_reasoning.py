from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_community.tools import TavilySearchResults
from langchain_anthropic import ChatAnthropic
from typing import TypedDict, List, Annotated
from dotenv import load_dotenv
import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
import random
import re
import json

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

# Initialize Tavily search
tavily_search = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=3)

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
llm_factory = LLMFactory(model_name="claude-3-haiku-20240307", together=False)
llm = llm_factory.get_llm()

def get_search_results(query, max_retries=3):
    """Enhanced search function using Tavily with better error handling and retry logic"""
    for attempt in range(max_retries):
        try:
            time.sleep(random.uniform(1, 2))
            results = tavily_search.invoke({"query": query})
            return results
        except Exception as e:
            print(f"Tavily Search error, attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt + random.uniform(1, 2))
                continue
            else:
                return f"Search results for '{query}': Information not available due to repeated errors. Error: {str(e)}"
    return f"Search results for '{query}': Information not available due to repeated errors."

def extract_main_text_from_url(url, max_chars=2000):
    """Extract main text content from a URL"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # Remove script, style, and other non-content elements
        for script in soup(["script", "style", "noscript", "nav", "footer", "header"]):
            script.decompose()
        
        # Extract text from main content areas
        main_content = soup.find('main') or soup.find('article') or soup.find('body')
        if main_content:
            text = ' '.join(main_content.stripped_strings)
        else:
            text = ' '.join(soup.stripped_strings)
        
        return text[:max_chars]
    except Exception as e:
        return f"[Could not extract content from {url}: {e}]"

def search_and_extract_tool(query):
    """Tool that searches for information using Tavily and extracts content from URLs with fallback strategies"""
    try:
        # Try the original query first
        search_results = get_search_results(query)
        
        # If search results contain URLs, extract content from them
        if isinstance(search_results, list) and len(search_results) > 0:
            extracted_content = []
            for result in search_results[:3]:  # Limit to first 3 results
                if 'url' in result:
                    content = extract_main_text_from_url(result['url'])
                    extracted_content.append(f"URL: {result['url']}\nTitle: {result.get('title', 'No title')}\nContent: {content}\n")
            
            if extracted_content:
                return "\n".join(extracted_content)
        
        # If no meaningful results, try a more general search
        if not search_results or len(str(search_results)) < 100:
            print(f"⚠️ No meaningful results for '{query}', trying general search...")
            
            # Extract key terms for general search
            key_terms = extract_key_terms(query)
            general_query = f"{key_terms} information facts"
            
            print(f"🔄 Trying general search: {general_query}")
            general_results = get_search_results(general_query)
            
            if isinstance(general_results, list) and len(general_results) > 0:
                extracted_content = []
                for result in general_results[:2]:  # Limit to first 2 results for general search
                    if 'url' in result:
                        content = extract_main_text_from_url(result['url'])
                        extracted_content.append(f"URL: {result['url']}\nTitle: {result.get('title', 'No title')}\nContent: {content}\n")
                
                if extracted_content:
                    return f"General search results for '{key_terms}':\n" + "\n".join(extracted_content)
        
        # If still no results, return the raw search results
        if isinstance(search_results, list):
            formatted_results = []
            for result in search_results[:3]:
                formatted_results.append(f"Title: {result.get('title', 'No title')}\nSnippet: {result.get('content', 'No content')}\nURL: {result.get('url', 'No URL')}")
            return "\n\n".join(formatted_results)
        
        return str(search_results)
    except Exception as e:
        return f"Error in search and extract: {str(e)}"

def extract_key_terms(query):
    """Extract key terms from a search query for fallback searches"""
    # Remove common words and keep important terms
    common_words = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'was', 'are', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall']
    
    # Split query into words and filter out common words
    words = query.lower().split()
    key_words = [word for word in words if word not in common_words and len(word) > 2]
    
    # Take up to 3 most important words
    return ' '.join(key_words[:3])

# Create tools for the LangGraph agent using the @tool decorator
@tool
def web_search(query: str) -> str:
    """Search the web for information about a person, entity, or topic using Tavily. Use this to find factual information about the character or person mentioned in the atomic fact."""
    return search_and_extract_tool(query)

# Define the state for our LangGraph
class AgentState(TypedDict):
    messages: Annotated[List, "The messages in the conversation"]
    character: Annotated[str, "The character being analyzed"]
    fact: Annotated[str, "The atomic fact to check"]
    search_results: Annotated[List, "Results from web searches"]
    reasoning_steps: Annotated[List, "Steps in the reasoning process"]
    final_label: Annotated[str, "The final label (Supported/Unsupported/Irrelevant)"]
    final_reasoning: Annotated[str, "The final reasoning for the label"]

# Define the reasoning node
def reasoning_node(state: AgentState) -> AgentState:
    """Node for initial reasoning and deciding what to search"""
    messages = state["messages"]
    character = state["character"]
    fact = state["fact"]
    
    print(f"\n{'='*60}")
    print(f"🧠 REASONING NODE - Analyzing fact about {character}")
    print(f"{'='*60}")
    print(f"Atomic Fact: {fact}")
    
    # Create reasoning prompt
    reasoning_prompt = f"""You are an expert fact-checking agent. Analyze this atomic fact about {character}: "{fact}"

Your task is to determine if this fact is Supported, Unsupported, or Irrelevant.

Think step by step:
1. What information do you need to verify this fact?
2. What search queries would help you find relevant information?
3. What specific aspects of {character} should you investigate?

Provide your reasoning and suggest search queries to gather information."""
    
    # Get reasoning from LLM
    print("🤔 Generating initial reasoning...")
    reasoning_response = llm.invoke([HumanMessage(content=reasoning_prompt)])
    reasoning_content = reasoning_response.content
    
    print(f"💭 Reasoning Output:\n{reasoning_content}")
    
    # Add reasoning to state
    state["reasoning_steps"].append({
        "step": "initial_reasoning",
        "content": reasoning_content
    })
    
    # Add reasoning message to conversation
    messages.append(AIMessage(content=reasoning_content))
    
    return state

# Define the search node
def search_node(state: AgentState) -> AgentState:
    """Node for performing intelligent web searches based on current information"""
    character = state["character"]
    fact = state["fact"]
    current_search_results = state.get("search_results", [])
    
    print(f"\n{'='*60}")
    print(f"🔍 SEARCH NODE - Intelligent search for {character}")
    print(f"{'='*60}")
    print(f"Current search results: {len(current_search_results)}")
    
    # If we already have 5 searches, move to analysis
    if len(current_search_results) >= 3:
        print("⚠️ Maximum searches (3) reached. Moving to analysis.")
        return state
    
    # Create a prompt for the LLM to decide what to search for
    search_decision_prompt = f"""You are an expert fact-checking agent. You need to determine what information to search for to verify this fact about {character}: "{fact}"

Current search results: {len(current_search_results)} searches performed

Your task is to decide:
1. Do we have enough information to make a decision about this fact?
2. If not, what specific search query should we use next to get the missing information?

Consider:
- What specific aspects of {character} are relevant to this fact?
- What information is still missing?
- What would be the most effective search query?
- Use simple, clear search terms that are likely to return results
- DO NOT use the same search query
- DO NOT try to perform more search if you have enough trustful information

IMPORTANT: You must respond with exactly this format:
ENOUGH_INFO: [Yes/No]
SEARCH_QUERY: [Your specific search query or "NONE" if enough info]
REASONING: [Why you made this decision]

Do not include any other text or formatting."""
    
    print("🤔 Deciding what to search for...")
    search_decision_response = llm.invoke([HumanMessage(content=search_decision_prompt)])
    decision_content = search_decision_response.content
    
    print(f"💭 Search Decision:\n{decision_content}")
    
    # Parse the decision with fallback patterns
    enough_info = False
    search_query = ""
    reasoning = ""
    
    # Pattern 1: Exact format match
    enough_info_match = re.search(r"ENOUGH_INFO:\s*(Yes|No)", decision_content, re.IGNORECASE)
    search_query_match = re.search(r"SEARCH_QUERY:\s*(.+)", decision_content, re.IGNORECASE)
    reasoning_match = re.search(r"REASONING:\s*(.+)", decision_content, re.IGNORECASE)
    
    if enough_info_match:
        enough_info = enough_info_match.group(1).lower() == "yes"
    if search_query_match:
        search_query = search_query_match.group(1).strip()
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    
    # Fallback: if parsing failed, try to extract from the text
    if not enough_info_match or not search_query_match:
        content_lower = decision_content.lower()
        if "enough" in content_lower and "yes" in content_lower:
            enough_info = True
        elif "not enough" in content_lower or "need more" in content_lower:
            enough_info = False
        
        # Try to extract search query from the text
        if not search_query or search_query.lower() == "none":
            # Look for potential search queries in the text
            lines = decision_content.split('\n')
            for line in lines:
                if any(word in line.lower() for word in [character.lower(), 'search', 'query', 'find']):
                    potential_query = line.strip()
                    if len(potential_query) > 10:  # Reasonable query length
                        search_query = potential_query
                        break
    print("Query to search:", search_query)
    # Add decision to reasoning steps
    state["reasoning_steps"].append({
        "step": "search_decision",
        "content": f"Decision: {'Enough info' if enough_info else 'Need more info'}. Query: {search_query}. Reasoning: {reasoning}"
    })
    
    if enough_info or search_query.lower() == "none":
        print("✅ Agent determined we have enough information. Moving to analysis.")
        return state
    
    # Perform the search
    print(f"🔎 Executing search {len(current_search_results) + 1}/3: {search_query}")
    try:
        result = web_search.invoke({"query": search_query})
        
        # Check if the search result is meaningful
        if not result or result.strip() == "" or "no search results found" in result.lower():
            result = f"Search returned no meaningful results for: {search_query}"
            print(f"⚠️ Search returned no meaningful results")
        else:
            print(f"✅ Search completed successfully")
            print(f"📄 Result preview: {result[:200]}...")
            
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        result = error_msg
        print(f"❌ Search failed: {error_msg}")
    
    search_results = current_search_results + [{
        "query": search_query,
        "result": result,
        "reasoning": reasoning
    }]
    
    state["search_results"] = search_results
    
    # Add search results to reasoning steps
    state["reasoning_steps"].append({
        "step": "web_search",
        "content": f"Performed search {len(search_results)}/5: {search_query}"
    })
    
    print(f"\n📊 Total search results collected: {len(search_results)}")
    
    return state

# Define the analysis node
def analysis_node(state: AgentState) -> AgentState:
    """Node for analyzing search results and determining the label"""
    messages = state["messages"]
    character = state["character"]
    fact = state["fact"]
    search_results = state["search_results"]
    
    print(f"\n{'='*60}")
    print(f"📊 ANALYSIS NODE - Analyzing search results for {character}")
    print(f"{'='*60}")
    print(f"Atomic Fact: {fact}")
    print(f"Number of search results to analyze: {len(search_results)}")
    
    # Format search results for analysis
    search_content = "\n\n".join([
        f"Search Query: {sr['query']}\nReasoning: {sr.get('reasoning', 'No reasoning provided')}\nResult: {sr['result']}"
        for sr in search_results
    ])
    
    # Create analysis prompt
    analysis_prompt = f"""Based on the search results below, label this atomic fact about {character}: "{fact}"

Search Results:
{search_content}

Labeling rules:
- Supported: The atomic fact is true and fully supported by reliable information
- Unsupported: The atomic fact is false or contradicts reliable information
- Irrelevant: The atomic fact is not related to the person/character or discusses something completely unrelated.

IMPORTANT: You must respond with exactly this format:
Label: [Supported/Unsupported/Irrelevant]
Reasoning: [Your detailed reasoning for this label]

Do not include any other text or formatting."""
    
    # Get analysis from LLM
    print("🧠 Generating analysis and final label...")
    analysis_response = llm.invoke([HumanMessage(content=analysis_prompt)])
    analysis_content = analysis_response.content
    
    print(f"📋 Analysis Output:\n{analysis_content}")
    
    # Parse the label and reasoning with multiple fallback patterns
    final_label = "Error parsing label"
    final_reasoning = analysis_content
    
    # Pattern 1: Exact format match
    label_match = re.search(r"Label:\s*(Supported|Unsupported|Irrelevant)", analysis_content, re.IGNORECASE)
    reasoning_match = re.search(r"Reasoning:\s*(.+)", analysis_content, re.DOTALL | re.IGNORECASE)
    
    if label_match:
        final_label = label_match.group(1)
        if reasoning_match:
            final_reasoning = reasoning_match.group(1).strip()
    else:
        # Pattern 2: Look for the words anywhere in the response
        content_lower = analysis_content.lower()
        if "supported" in content_lower and "unsupported" not in content_lower and "irrelevant" not in content_lower:
            final_label = "Supported"
        elif "unsupported" in content_lower and "supported" not in content_lower and "irrelevant" not in content_lower:
            final_label = "Unsupported"
        elif "irrelevant" in content_lower and "supported" not in content_lower and "unsupported" not in content_lower:
            final_label = "Irrelevant"
        else:
            # Pattern 3: Count occurrences and pick the most frequent
            supported_count = content_lower.count("supported")
            unsupported_count = content_lower.count("unsupported")
            irrelevant_count = content_lower.count("irrelevant")
            
            max_count = max(supported_count, unsupported_count, irrelevant_count)
            if max_count > 0:
                if supported_count == max_count:
                    final_label = "Supported"
                elif unsupported_count == max_count:
                    final_label = "Unsupported"
                elif irrelevant_count == max_count:
                    final_label = "Irrelevant"
    
    print(f"\n🎯 FINAL RESULT:")
    print(f"   Label: {final_label}")
    print(f"   Reasoning: {final_reasoning[:200]}...")
    print(f"   Total searches performed: {len(search_results)}")
    
    # Update state
    state["final_label"] = final_label
    state["final_reasoning"] = final_reasoning
    state["reasoning_steps"].append({
        "step": "analysis",
        "content": analysis_content
    })
    
    # Add analysis message to conversation
    messages.append(AIMessage(content=analysis_content))
    
    return state

# Define the router function
def router(state: AgentState) -> str:
    """Router to determine the next step"""
    search_results = state.get("search_results", [])
    
    # If we have search results, check if we should continue searching or analyze
    if search_results:
        # Check the last search decision to see if we have enough info
        reasoning_steps = state.get("reasoning_steps", [])
        last_search_decision = None
        
        for step in reversed(reasoning_steps):
            if step.get("step") == "search_decision":
                last_search_decision = step.get("content", "")
                break
        
        # Robustly check ENOUGH_INFO value
        if last_search_decision:
            enough_info_match = re.search(r"ENOUGH_INFO:\s*(Yes|No)", last_search_decision, re.IGNORECASE)
            if enough_info_match and enough_info_match.group(1).strip().lower() == "yes":
                return "analysis"
            # Fallback: also check for the phrase as before
            if "Enough info" in last_search_decision or "ENOUGH_INFO: Yes" in last_search_decision:
                return "analysis"
        
        # If we've reached max searches (3), go to analysis
        if len(search_results) >= 3:
            return "analysis"
        
        # Otherwise, continue searching
        return "search"
    
    # If no search results yet, go to search
    return "search"

# Create the LangGraph
def create_fact_checking_graph():
    """Create the LangGraph for fact checking"""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("reasoning", reasoning_node)
    workflow.add_node("search", search_node)
    workflow.add_node("analysis", analysis_node)
    
    # Add edges
    workflow.add_edge("reasoning", "search")
    workflow.add_conditional_edges(
        "search",
        router,
        {
            "analysis": "analysis",
            "search": "search"
        }
    )
    workflow.add_edge("analysis", END)
    
    # Set entry point
    workflow.set_entry_point("reasoning")
    
    return workflow.compile()

# Create the graph
fact_checking_graph = create_fact_checking_graph()

def get_langgraph_response(atomic_fact, character):
    """Get response using LangGraph reasoning"""
    try:
        # Initialize state
        initial_state = {
            "messages": [HumanMessage(content=f"Analyze this fact about {character}: {atomic_fact}")],
            "character": character,
            "fact": atomic_fact,
            "search_results": [],
            "reasoning_steps": [],
            "final_label": "",
            "final_reasoning": ""
        }
        
        # Run the graph
        result = fact_checking_graph.invoke(initial_state)
        
        # Extract results
        final_label = result.get("final_label", "Error")
        final_reasoning = result.get("final_reasoning", "")
        reasoning_steps = result.get("reasoning_steps", [])
        
        # Create full response
        full_response = f"Final Label: {final_label}\n\nFinal Reasoning: {final_reasoning}\n\nReasoning Steps: {json.dumps(reasoning_steps, indent=2)}"
        
        return {
            "full_response": full_response,
            "label": final_label,
            "reasoning_steps": reasoning_steps
        }
        
    except Exception as e:
        print(f"Error in LangGraph: {e}")
        return {
            "full_response": f"Error: {str(e)}",
            "label": "Error",
            "reasoning_steps": []
        }

def factscore_reasoning(entity_name):
    """Enhanced fact scoring using LangGraph reasoning"""
    dataset_src = "Dataset/"
    df = pd.read_csv(f"{dataset_src}{entity_name}.csv")
    atomic_facts = df.iloc[:, 0].tolist()
    labels = df.iloc[:, 1].tolist()
    
    print(f"\n{'='*80}")
    print(f"🚀 STARTING FACT CHECKING FOR: {entity_name.upper()}")
    print(f"{'='*80}")
    print(f"📊 Total facts to process: {len(atomic_facts)}")
    print(f"📁 Dataset source: {dataset_src}{entity_name}.csv")
    
    predicted_labels = []
    full_responses = []
    reasoning_steps = []
    
    for i, atomic_fact in tqdm(enumerate(atomic_facts), total=len(atomic_facts)):
        print(f"\n{'='*80}")
        print(f"📝 PROCESSING FACT {i+1}/{len(atomic_facts)}")
        print(f"{'='*80}")
        print(f"🔍 Atomic Fact: {atomic_fact}")
        print(f"✅ True Label: {labels[i]}")
        
        topic = entity_name
        result = get_langgraph_response(atomic_fact, topic)
        
        predicted_labels.append(result['label'])
        full_responses.append(result['full_response'])
        
        # Format reasoning steps for CSV storage
        formatted_reasoning = format_reasoning_steps_for_csv(result['reasoning_steps'])
        reasoning_steps.append(formatted_reasoning)
        
        print(f"\n🎯 PREDICTION RESULT:")
        print(f"   Predicted Label: {result['label']}")
        print(f"   True Label: {labels[i]}")
        print(f"   Match: {'✅' if result['label'] == labels[i] else '❌'}")
        print(f"   Reasoning Steps: {len(result['reasoning_steps'])} steps")
        
        # Show a brief summary of reasoning steps
        if result['reasoning_steps']:
            print(f"   Step Summary:")
            for step in result['reasoning_steps']:
                step_name = step.get('step', 'unknown')
                print(f"     - {step_name}")
    
    # Calculate accuracy
    correct_predictions = sum(1 for i, pred in enumerate(predicted_labels) if pred == labels[i])
    accuracy = correct_predictions / len(atomic_facts)
    
    print(f"\n{'='*80}")
    print(f"📈 FINAL RESULTS FOR {entity_name.upper()}")
    print(f"{'='*80}")
    print(f"📊 Total Facts: {len(atomic_facts)}")
    print(f"✅ Correct Predictions: {correct_predictions}")
    print(f"🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"📋 Label Distribution:")
    print(f"   - Supported: {predicted_labels.count('Supported')}")
    print(f"   - Unsupported: {predicted_labels.count('Unsupported')}")
    print(f"   - Irrelevant: {predicted_labels.count('Irrelevant')}")
    print(f"   - Errors: {predicted_labels.count('Error')}")
    
    # Update dataframe with results
    df['Predicted Label'] = predicted_labels
    df['Full Response'] = full_responses
    df['Reasoning Steps'] = reasoning_steps
    df.rename(columns={df.columns[0]: 'Atomic Fact'}, inplace=True)
    df.rename(columns={df.columns[1]: 'True Label'}, inplace=True)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed:')]
    
    # Save results
    result_src = "Results/Reasoning/"
    os.makedirs(result_src, exist_ok=True)
    output_file = f"{result_src}{entity_name}_langgraph_labeled.csv"
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"💾 Results saved to: {output_file}")
    
    result_map = {
        'total_facts': len(atomic_facts),
        'supported_count': predicted_labels.count('Supported'),
        'unsupported_count': predicted_labels.count('Unsupported'),
        'irrelevant_count': predicted_labels.count('Irrelevant'),
        'error_count': predicted_labels.count('Error'),
        'accuracy': accuracy,
        'correct_predictions': correct_predictions
    }
    
    return result_map

def format_reasoning_steps_for_csv(reasoning_steps):
    """Format reasoning steps for CSV storage in a readable format"""
    if not reasoning_steps:
        return "No reasoning steps available"
    
    formatted_steps = []
    for i, step in enumerate(reasoning_steps, 1):
        step_name = step.get('step', 'unknown_step')
        step_content = step.get('content', 'No content')
        
        # Clean and format the content
        if isinstance(step_content, str):
            # Remove excessive whitespace and newlines for CSV compatibility
            cleaned_content = ' '.join(step_content.split())
            # Truncate if too long for CSV
            if len(cleaned_content) > 1000:
                cleaned_content = cleaned_content[:1000] + "..."
        else:
            cleaned_content = str(step_content)
        
        # Add special formatting for search decisions
        if step_name == "search_decision":
            formatted_step = f"Step {i} (SEARCH_DECISION): {cleaned_content}"
        elif step_name == "web_search":
            formatted_step = f"Step {i} (WEB_SEARCH): {cleaned_content}"
        else:
            formatted_step = f"Step {i} ({step_name}): {cleaned_content}"
        
        formatted_steps.append(formatted_step)
    
    # Join all steps with a clear separator
    return " | ".join(formatted_steps)

def run_dataset_reasoning(dataset_src="Dataset/"):
    """Run the LangGraph-based fact checking on the entire dataset"""
    csv_files = [f for f in os.listdir(dataset_src) if f.endswith('.csv')]
    accuracy_dict = {}
    detailed_results = {}
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    for csv_file in csv_files:
        entity_name = csv_file.split('.')[0]
        print(f"\n{'='*50}")
        print(f"Processing entity: {entity_name}")
        print(f"{'='*50}")
        
        try:
            result_map = factscore_reasoning(entity_name)
            print(f"Results for {entity_name}:")
            print(f"  Total facts: {result_map['total_facts']}")
            print(f"  Supported: {result_map['supported_count']}")
            print(f"  Unsupported: {result_map['unsupported_count']}")
            print(f"  Irrelevant: {result_map['irrelevant_count']}")
            print(f"  Errors: {result_map['error_count']}")
            print(f"  Accuracy: {result_map['accuracy']:.4f}")
            
            accuracy_dict[entity_name] = result_map['accuracy']
            detailed_results[entity_name] = result_map
            
        except Exception as e:
            print(f"Error processing {entity_name}: {e}")
            accuracy_dict[entity_name] = 0.0
            detailed_results[entity_name] = {"error": str(e)}
    
    # Calculate overall statistics
    valid_accuracies = [acc for acc in accuracy_dict.values() if acc > 0]
    if valid_accuracies:
        average_accuracy = sum(valid_accuracies) / len(valid_accuracies)
        print(f"\n{'='*50}")
        print(f"OVERALL RESULTS")
        print(f"{'='*50}")
        print(f"Average accuracy: {average_accuracy:.4f}")
        print(f"Entities processed: {len(valid_accuracies)}/{len(csv_files)}")
        
        # Save overall results
        overall_results = {
            'entity_results': detailed_results,
            'accuracy_summary': accuracy_dict,
            'average_accuracy': average_accuracy,
            'total_entities': len(csv_files),
            'successful_entities': len(valid_accuracies)
        }
        
        result_src = "Results/"
        os.makedirs(result_src, exist_ok=True)
        
        # Save detailed results as JSON for further analysis
        with open(f"{result_src}overall_langgraph_results.json", 'w') as f:
            json.dump(overall_results, f, indent=2)
        
        return accuracy_dict, overall_results
    else:
        print("No successful processing results found.")
        return accuracy_dict, {}

if __name__ == "__main__":
    # Example usage
    print("Starting LangGraph-based fact checking...")
    accuracy_dict, overall_results = run_dataset_reasoning()
    print("Accuracy:", accuracy_dict)
    print("Overall result:", overall_results)
    print("Processing complete!")
