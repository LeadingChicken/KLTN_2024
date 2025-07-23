# FastAPI backend for atomic fact labeling using LLM voting
# POST /label_fact with JSON: {"character": ..., "atomic_fact": ...} returns label, reasoning, and sources.
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
from dotenv import load_dotenv
import time
import random
from collections import Counter
import re
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Import LLM and search dependencies
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from split_atomic_facts import split_into_atomic_facts
from uqlm import BlackBoxUQ
from uqlm import UQEnsemble
from uqlm import LLMPanel
from uqlm.judges import LLMJudge
import asyncio

# Load API keys
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

# LLM Factory (from LLM_auto_label_voting.py)
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

# Model names
model_gemini = "gemini-2.0-flash"
llm_factory = LLMFactory(model_name=model_gemini, together=False)
gemini_llm = llm_factory.get_llm()

tavily_search = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=3)

def get_search_results(query, max_retries=3):
    for attempt in range(max_retries):
        try:
            time.sleep(random.uniform(1, 2))
            return tavily_search.invoke({"query": query})
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt + random.uniform(1, 2))
                continue
            else:
                return []  # Return empty list on repeated errors
    return []

rag_template = """
You are given a context (web search) followed by an atomic fact. You job is to label whether this atomic fact is Supported, Unsupported, Irrelevant. The context is about a person or a character and the atomic fact is a piece of information about that person.

Supported: Choose this if the atomic fact is true and fully supported by the context.
Unsupported: Choose this if the atomic fact is false and contradict with the context.
Irrelevant: Choose this if the atomic fact is irrelevant to the person or the atomic fact talking about another thing that do not have connection with the person.

This is your part:
Character: {character}
Context from the web search: {context}

atomic fact: {fact}

Return with the format:
Label: <Your label>
Reason: <Your reason for this label, write in Vietnamese>
Context: <Summary of the context, write in Vietnamese>
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

class FactRequest(BaseModel):
    character: str
    atomic_fact: str
    
class SourceInfo(BaseModel):
    title: str
    url: str

class FactResponse(BaseModel):
    label: str
    reasoning: str
    sources: List[SourceInfo]
    context: str
    llm_labels: List[str]

class SplitFactsRequest(BaseModel):
    biography_text: str

class SplitFactsResponse(BaseModel):
    atomic_facts: List[str]

class LabelWithConfidenceRequest(BaseModel):
    biography_text: str
    character: str = ""

class AtomicFactWithConfidence(BaseModel):
    atomic_fact: str
    label: str
    reasoning: str
    sources: List[SourceInfo]
    prompt: str
    confidence: float

class LabelWithConfidenceResponse(BaseModel):
    results: List[AtomicFactWithConfidence]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return FileResponse("frontend/index.html")

@app.post("/label_fact", response_model=FactResponse)
def label_fact(request: FactRequest):
    try:
        result = get_gemini_response(request.atomic_fact, request.character)
        label = result['label']
        response_text = result['response']
        sources = result.get('sources', [])
        context = result.get('context', '')
        # Extract reason
        reason = ""
        for line in response_text.splitlines():
            if line.startswith("Reason:"):
                reason = line.replace("Reason:", "").strip()
        return FactResponse(label=label, reasoning=reason, sources=sources, context=context, llm_labels=[label])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/split_atomic_facts", response_model=SplitFactsResponse)
def split_atomic_facts_api(request: SplitFactsRequest):
    try:
        facts = split_into_atomic_facts(request.biography_text)
        return SplitFactsResponse(atomic_facts=facts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/label_facts_with_confidence", response_model=LabelWithConfidenceResponse)
async def label_facts_with_confidence(request: LabelWithConfidenceRequest):
    try:
        print("Đang tách atomic facts")
        facts = split_into_atomic_facts(request.biography_text)
        prompts = []
        search_contexts = []
        sources_list = []
        print("Đang search")
        for fact in facts:
            search_query = f"{request.character} {fact}".strip()
            search_results = get_search_results(search_query)
            if isinstance(search_results, list):
                sources = [
                    {"title": item.get("title", item.get("url", "No title")), "url": item["url"]}
                    for item in search_results if "url" in item
                ]
                context = "\n".join([item.get('content', '') for item in search_results])
            else:
                sources = []
                context = str(search_results)
            prompt_str = prompt.invoke({
                "character": request.character,
                "context": context,
                "fact": fact,
            })
            prompts.append(prompt_str.text)
            search_contexts.append(context)
            sources_list.append(sources)
        # Run BlackBoxUQ for confidence scores (async)
        async def get_confidences(prompts):
            # Black box UQ
            llm = gemini_llm
            bbuq = BlackBoxUQ(llm=llm, scorers=["semantic_negentropy"], use_best=True, max_calls_per_min=100)
            results_BB = await bbuq.generate_and_score(prompts=prompts, num_responses=5)
            # LLM as a Judge
            panel = LLMPanel(llm=llm, judges=[llm,llm,llm,llm,llm], scoring_templates=['true_false_uncertain','true_false_uncertain','true_false_uncertain','true_false_uncertain','true_false_uncertain'])
            results_judge = await panel.generate_and_score(prompts=prompts)
            # Calculate confidence score
            responses = results_BB.to_df()["response"].tolist()
            result_BB = results_BB.to_df()
            result_judge = results_judge.to_df()
            result =((result_judge['avg'] + result_BB['semantic_negentropy'])/2).tolist()
            return result, responses
        print("Đang sinh confidence score")
        confidences, responses = await get_confidences(prompts)
        results = []
        for i, fact in enumerate(facts):
            # Parse label and reasoning from response
            response = responses[i]
            try:
                label = response.split("Label:")[1].split("\n")[0].strip()
            except:
                label = "Error parsing label"
            reasoning = ""
            for line in response.splitlines():
                if line.startswith("Reason:"):
                    reasoning = line.replace("Reason:", "").strip()
            results.append(AtomicFactWithConfidence(
                atomic_fact=fact,
                label=label,
                reasoning=reasoning,
                sources=sources_list[i],
                prompt=prompts[i],
                confidence=confidences[i]
            ))
        return LabelWithConfidenceResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_gemini_response(query, topic):
    search_query = f"{topic} {query}"
    try:
        search_results = get_search_results(search_query)
        if isinstance(search_results, list):
            sources = [
                {"title": item.get("title", item.get("url", "No title")), "url": item["url"]}
                for item in search_results if "url" in item
            ]
            context = "\n".join([item.get('content', '') for item in search_results])
        else:
            sources = []
            context = str(search_results)
    except Exception as e:
        sources = []
        context = f"Search results for '{search_query}': Unable to retrieve information at this time. Error: {str(e)}"
    label, response = get_llm_label(gemini_llm, topic, context, query)
    return {"label": label, "response": response, "sources": sources, "context": context} 