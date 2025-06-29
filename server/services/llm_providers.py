"""
LLM provider integrations for Vision Agent
"""
import base64
import logging
import asyncio
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from dataclasses import dataclass
import openai
from anthropic import AsyncAnthropic
import json

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM provider"""
    success: bool
    content: str
    confidence: float
    tokens_used: int
    cost_cents: Optional[float]
    provider: str
    model: str
    metadata: Dict[str, Any]
    error_message: Optional[str] = None


@dataclass
class AccessibilityPrompt:
    """Structured prompt for accessibility analysis"""
    base_prompt: str
    context_specific: str
    verbosity_level: str
    output_format: str
    user_preferences: Dict[str, Any]


class BaseLLMProvider(ABC):
    """Base class for LLM providers"""
    
    @abstractmethod
    async def analyze_image(self, 
                          image_data: bytes, 
                          prompt: AccessibilityPrompt) -> LLMResponse:
        """Analyze image with accessibility prompt"""
        pass
    
    @abstractmethod
    def estimate_cost(self, image_size: int, prompt_length: int) -> float:
        """Estimate processing cost in cents"""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT-4V provider for image analysis"""
    
    def __init__(self, api_key: str, model: str = "gpt-4-vision-preview"):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model
        
        # Pricing (approximate, in cents per 1K tokens)
        self.input_token_cost = 1.0  # $0.01 per 1K tokens
        self.output_token_cost = 3.0  # $0.03 per 1K tokens
        self.image_cost_per_tile = 0.17  # $0.00170 per image tile
    
    async def analyze_image(self, 
                          image_data: bytes, 
                          prompt: AccessibilityPrompt) -> LLMResponse:
        """Analyze image using OpenAI GPT-4V"""
        try:
            # Encode image to base64
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            # Build comprehensive prompt
            full_prompt = self._build_prompt(prompt)
            
            # Make API call
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": full_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1500,
                temperature=0.1
            )
            
            # Extract response
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            
            # Parse structured response
            parsed_content = self._parse_response(content)
            
            # Calculate cost
            cost_cents = self._calculate_cost(tokens_used, len(image_data))
            
            # Estimate confidence based on response completeness
            confidence = self._estimate_confidence(parsed_content)
            
            return LLMResponse(
                success=True,
                content=json.dumps(parsed_content),
                confidence=confidence,
                tokens_used=tokens_used,
                cost_cents=cost_cents,
                provider="openai",
                model=self.model,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "response_id": response.id,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens
                }
            )
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return LLMResponse(
                success=False,
                content="",
                confidence=0.0,
                tokens_used=0,
                cost_cents=0.0,
                provider="openai",
                model=self.model,
                metadata={},
                error_message=str(e)
            )
    
    def _build_prompt(self, prompt: AccessibilityPrompt) -> str:
        """Build comprehensive accessibility prompt"""
        base = """You are an expert accessibility analyst. Analyze this image and provide detailed accessibility information.

REQUIRED OUTPUT FORMAT (JSON):
{
  "alt_text": "Primary alternative text description",
  "detailed_description": "Extended detailed description",
  "brief_summary": "One-line summary",
  "structural_elements": {
    "headings": [{"level": 1, "text": "Title", "position": {"x": 0, "y": 0}}],
    "paragraphs": [{"text": "Content", "position": {"x": 0, "y": 0}}],
    "lists": [{"type": "unordered", "items": ["Item 1"], "position": {"x": 0, "y": 0}}],
    "tables": [{"headers": ["Col1"], "rows": [["Data"]], "caption": "Table description"}]
  },
  "interactive_elements": {
    "buttons": [{"text": "Click me", "position": {"x": 0, "y": 0}, "enabled": true}],
    "links": [{"text": "Learn more", "url": "", "position": {"x": 0, "y": 0}}],
    "form_fields": [{"type": "input", "label": "Email", "required": true, "position": {"x": 0, "y": 0}}],
    "images": [{"alt": "Description", "position": {"x": 0, "y": 0}}]
  },
  "reading_order": ["element1", "element2", "element3"],
  "text_content": "All visible text extracted",
  "layout_description": "Overall layout and visual hierarchy",
  "color_information": {"background": "white", "text": "black", "accents": ["blue"]},
  "accessibility_issues": ["Issue 1", "Issue 2"],
  "navigation_landmarks": {"header": true, "main": true, "footer": false}
}"""
        
        # Add context-specific instructions
        if prompt.context_specific:
            base += f"\n\nCONTEXT-SPECIFIC ANALYSIS:\n{prompt.context_specific}"
        
        # Add verbosity instructions
        verbosity_instructions = {
            "brief": "Provide concise descriptions focusing on essential information only.",
            "medium": "Provide balanced descriptions with key details.",
            "detailed": "Provide comprehensive descriptions with all relevant details."
        }
        
        base += f"\n\nVERBOSITY LEVEL: {prompt.verbosity_level.upper()}\n"
        base += verbosity_instructions.get(prompt.verbosity_level, verbosity_instructions["medium"])
        
        # Add user preferences
        if prompt.user_preferences:
            base += f"\n\nUSER PREFERENCES:\n{json.dumps(prompt.user_preferences, indent=2)}"
        
        return base
    
    def _parse_response(self, content: str) -> Dict[str, Any]:
        """Parse and validate LLM response"""
        try:
            # Try to extract JSON from response
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                json_content = content[json_start:json_end].strip()
            elif "{" in content and "}" in content:
                json_start = content.find("{")
                json_end = content.rfind("}") + 1
                json_content = content[json_start:json_end]
            else:
                # Fallback: create basic structure
                return {
                    "alt_text": content[:200] + "..." if len(content) > 200 else content,
                    "detailed_description": content,
                    "structural_elements": {},
                    "interactive_elements": {},
                    "reading_order": [],
                    "text_content": "",
                    "layout_description": "",
                    "accessibility_issues": []
                }
            
            parsed = json.loads(json_content)
            
            # Validate required fields
            required_fields = ["alt_text", "structural_elements", "interactive_elements"]
            for field in required_fields:
                if field not in parsed:
                    parsed[field] = ""
            
            return parsed
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return {
                "alt_text": content[:200] + "..." if len(content) > 200 else content,
                "detailed_description": content,
                "structural_elements": {},
                "interactive_elements": {},
                "reading_order": [],
                "parsing_error": str(e)
            }
    
    def _estimate_confidence(self, parsed_content: Dict[str, Any]) -> float:
        """Estimate confidence based on response completeness"""
        confidence = 0.5  # Base confidence
        
        # Check for key components
        if parsed_content.get("alt_text"):
            confidence += 0.2
        
        if parsed_content.get("structural_elements"):
            confidence += 0.1
        
        if parsed_content.get("interactive_elements"):
            confidence += 0.1
        
        if parsed_content.get("text_content"):
            confidence += 0.1
        
        # Penalize for parsing errors
        if "parsing_error" in parsed_content:
            confidence -= 0.2
        
        return min(1.0, max(0.0, confidence))
    
    def _calculate_cost(self, tokens_used: int, image_size_bytes: int) -> float:
        """Calculate processing cost"""
        # Token costs
        token_cost = (tokens_used / 1000) * self.input_token_cost
        
        # Image processing cost (rough estimate based on size)
        # GPT-4V charges per image tile (512x512 pixels)
        estimated_pixels = image_size_bytes * 0.1  # Rough estimate
        estimated_tiles = max(1, estimated_pixels / (512 * 512))
        image_cost = estimated_tiles * self.image_cost_per_tile
        
        return token_cost + image_cost
    
    def estimate_cost(self, image_size: int, prompt_length: int) -> float:
        """Estimate processing cost"""
        estimated_tokens = (prompt_length / 4) + 500  # Rough token estimate
        return self._calculate_cost(estimated_tokens, image_size)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider for image analysis"""
    
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model
        self.input_token_cost = 0.3  # $0.003 per 1K tokens
        self.output_token_cost = 1.5  # $0.015 per 1K tokens
    
    async def analyze_image(self, 
                          image_data: bytes, 
                          prompt: AccessibilityPrompt) -> LLMResponse:
        """Analyze image using Anthropic Claude"""
        try:
            # Encode image to base64
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            # Build prompt
            full_prompt = self._build_prompt(prompt)
            
            # Make API call
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=1500,
                temperature=0.1,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": full_prompt
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_b64
                                }
                            }
                        ]
                    }
                ]
            )
            
            # Extract response
            content = response.content[0].text
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            
            # Parse response
            parsed_content = self._parse_response(content)
            
            # Calculate cost
            cost_cents = self._calculate_cost(response.usage.input_tokens, response.usage.output_tokens)
            
            # Estimate confidence
            confidence = self._estimate_confidence(parsed_content)
            
            return LLMResponse(
                success=True,
                content=json.dumps(parsed_content),
                confidence=confidence,
                tokens_used=tokens_used,
                cost_cents=cost_cents,
                provider="anthropic",
                model=self.model,
                metadata={
                    "stop_reason": response.stop_reason,
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                }
            )
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return LLMResponse(
                success=False,
                content="",
                confidence=0.0,
                tokens_used=0,
                cost_cents=0.0,
                provider="anthropic",
                model=self.model,
                metadata={},
                error_message=str(e)
            )
    
    def _build_prompt(self, prompt: AccessibilityPrompt) -> str:
        """Build prompt for Claude"""
        # Similar to OpenAI but adjusted for Claude's preferences
        return self._build_accessibility_prompt(prompt)
    
    def _parse_response(self, content: str) -> Dict[str, Any]:
        """Parse Claude response"""
        # Similar parsing logic to OpenAI
        return self._parse_json_response(content)
    
    def _estimate_confidence(self, parsed_content: Dict[str, Any]) -> float:
        """Estimate confidence for Claude response"""
        # Similar confidence estimation
        return self._calculate_response_confidence(parsed_content)
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate Anthropic processing cost"""
        input_cost = (input_tokens / 1000) * self.input_token_cost
        output_cost = (output_tokens / 1000) * self.output_token_cost
        return input_cost + output_cost
    
    def estimate_cost(self, image_size: int, prompt_length: int) -> float:
        """Estimate cost for Anthropic"""
        estimated_input_tokens = (prompt_length / 4) + 200
        estimated_output_tokens = 500
        return self._calculate_cost(estimated_input_tokens, estimated_output_tokens)


class LLMProviderManager:
    """Manages multiple LLM providers with fallback"""
    
    def __init__(self):
        self.providers: Dict[str, BaseLLMProvider] = {}
        self.provider_order = ["openai", "anthropic", "google", "local"]
    
    def add_provider(self, name: str, provider: BaseLLMProvider):
        """Add LLM provider"""
        self.providers[name] = provider
        logger.info(f"Added LLM provider: {name}")
    
    async def analyze_image_with_fallback(self, 
                                        image_data: bytes, 
                                        prompt: AccessibilityPrompt,
                                        preferred_provider: str = None) -> LLMResponse:
        """Analyze image with automatic fallback"""
        # Determine provider order
        providers_to_try = [preferred_provider] if preferred_provider else []
        providers_to_try.extend([p for p in self.provider_order if p not in providers_to_try])
        
        last_error = None
        
        for provider_name in providers_to_try:
            if provider_name not in self.providers:
                continue
            
            provider = self.providers[provider_name]
            logger.info(f"Attempting analysis with {provider_name}")
            
            try:
                response = await provider.analyze_image(image_data, prompt)
                if response.success:
                    logger.info(f"Successfully analyzed with {provider_name}")
                    return response
                else:
                    last_error = response.error_message
                    logger.warning(f"{provider_name} failed: {last_error}")
            
            except Exception as e:
                last_error = str(e)
                logger.error(f"{provider_name} error: {e}")
        
        # All providers failed
        return LLMResponse(
            success=False,
            content="",
            confidence=0.0,
            tokens_used=0,
            cost_cents=0.0,
            provider="none",
            model="none",
            metadata={},
            error_message=f"All providers failed. Last error: {last_error}"
        )
    
    def get_cheapest_provider_for_task(self, image_size: int, 
                                     prompt_length: int) -> Optional[str]:
        """Get cheapest provider for given task"""
        if not self.providers:
            return None
        
        costs = {}
        for name, provider in self.providers.items():
            try:
                costs[name] = provider.estimate_cost(image_size, prompt_length)
            except Exception:
                continue
        
        if costs:
            return min(costs, key=costs.get)
        return None