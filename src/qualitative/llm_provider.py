import json
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from config.settings import get_settings
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

class QualitativeInterpretation(BaseModel):
    summary: str = Field(description="A concise summary of the material changes in Risk Factors, strictly based on the provided evidence.")
    risk_level_change: str = Field(description="One of: 'INCREASED', 'DECREASED', 'UNCHANGED', or 'UNKNOWN'")
    key_themes: List[str] = Field(description="List of key risk themes that emerged or disappeared.")

class LLMProvider(ABC):
    @abstractmethod
    def interpret_diff(self, added: List[str], deleted: List[str], shifted: List[str]) -> QualitativeInterpretation:
        pass

class LangchainLLMProvider(LLMProvider):
    def __init__(self):
        settings = get_settings()
        self.parser = JsonOutputParser(pydantic_object=QualitativeInterpretation)
        
        self.prompt = PromptTemplate(
            template="""You are a strict financial risk analyst.
Your task is to interpret the changes in a company's SEC 10-K Risk Factors section from one year to the next.
You MUST rely ONLY on the provided added, deleted, and shifted sentences. Do NOT invent evidence. Do NOT use outside knowledge.
If no material changes are provided, summarize that risks are unchanged.

Added sentences:
{added}

Deleted sentences:
{deleted}

Shifted/Modified sentences:
{shifted}

Output your response strictly as JSON that matches the following schema:
{format_instructions}
""",
            input_variables=["added", "deleted", "shifted"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )
        
        if settings.llm.gemini_available:
            self.model = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=settings.llm.gemini_api_key,
                temperature=0.0
            )
            self.provider_name = "gemini"
            self.model_name = "gemini-1.5-flash"
        elif settings.llm.ollama_available:
            self.model = ChatOllama(
                base_url=settings.llm.ollama_host,
                model=settings.llm.ollama_model,
                temperature=0.0
            )
            self.provider_name = "ollama"
            self.model_name = settings.llm.ollama_model
        else:
            raise ValueError("No LLM provider available (Gemini or Ollama required).")
            
        self.chain = self.prompt | self.model | self.parser

    def interpret_diff(self, added: List[str], deleted: List[str], shifted: List[str]) -> QualitativeInterpretation:
        # Truncate lists if they are extremely long to prevent context overflow
        # but the prompt injection rule means we should treat it carefully.
        added_text = "\n".join(added[:50]) if added else "None"
        deleted_text = "\n".join(deleted[:50]) if deleted else "None"
        shifted_text = "\n".join(shifted[:50]) if shifted else "None"
        
        try:
            result = self.chain.invoke({
                "added": added_text,
                "deleted": deleted_text,
                "shifted": shifted_text
            })
            return QualitativeInterpretation.model_validate(result)
        except Exception as e:
            # Fallback for structured output failures
            return QualitativeInterpretation(
                summary="Failed to parse LLM response or LLM error.",
                risk_level_change="UNKNOWN",
                key_themes=[]
            )
