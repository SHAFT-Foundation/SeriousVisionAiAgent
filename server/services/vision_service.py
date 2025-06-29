"""
Vision processing service that coordinates image analysis
"""
import logging
import hashlib
import json
from typing import Dict, Any, Optional
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..models import User, ProcessingJob, AccessibilityAnnotation, CacheEntry
from .llm_providers import LLMProviderManager, AccessibilityPrompt, LLMResponse
from ..utils.config import get_settings, get_llm_config

logger = logging.getLogger(__name__)


class VisionProcessingService:
    """Main service for processing visual content"""
    
    def __init__(self, llm_manager: LLMProviderManager):
        self.llm_manager = llm_manager
        self.settings = get_settings()
    
    async def process_image(self, 
                          db: AsyncSession,
                          user: User,
                          job: ProcessingJob,
                          image_data: bytes) -> AccessibilityAnnotation:
        """
        Process image and create accessibility annotation
        
        Args:
            db: Database session
            user: User making the request
            job: Processing job record
            image_data: Raw image bytes
        
        Returns:
            AccessibilityAnnotation with processed results
        """
        try:
            # Check cache first
            cache_result = await self._check_cache(db, job, image_data)
            if cache_result:
                logger.info(f"Cache hit for job {job.id}")
                job.complete_processing(
                    confidence_score=cache_result["confidence_score"],
                    was_cached=True
                )
                return await self._create_annotation_from_cache(db, job, cache_result)
            
            # Start processing
            job.start_processing("determining", "determining")
            await db.commit()
            
            # Build accessibility prompt
            prompt = self._build_accessibility_prompt(user, job)
            
            # Determine best provider
            preferred_provider = self._select_provider(user, image_data, prompt)
            
            # Process with LLM
            logger.info(f"Processing image with {preferred_provider} for job {job.id}")
            llm_response = await self.llm_manager.analyze_image_with_fallback(
                image_data, prompt, preferred_provider
            )
            
            if not llm_response.success:
                job.fail_processing(
                    error_message=llm_response.error_message,
                    error_code="LLM_PROCESSING_FAILED"
                )
                await db.commit()
                raise Exception(f"LLM processing failed: {llm_response.error_message}")
            
            # Complete job with results
            job.complete_processing(
                confidence_score=llm_response.confidence,
                tokens_used=llm_response.tokens_used,
                api_cost_cents=llm_response.cost_cents,
                was_cached=False
            )
            job.llm_provider = llm_response.provider
            job.model_used = llm_response.model
            
            # Parse LLM response
            response_data = json.loads(llm_response.content)
            
            # Create accessibility annotation
            annotation = await self._create_annotation_from_llm_response(
                db, job, response_data, llm_response
            )
            
            # Cache results for future use
            await self._cache_results(db, job, image_data, response_data, llm_response)
            
            # Update user statistics
            user.update_usage_stats(success=True, context=job.processing_context)
            
            await db.commit()
            
            logger.info(f"Successfully processed job {job.id}")
            return annotation
            
        except Exception as e:
            logger.error(f"Error processing image for job {job.id}: {e}")
            
            # Mark job as failed if not already done
            if job.processing_status == "processing":
                job.fail_processing(
                    error_message=str(e),
                    error_code="PROCESSING_ERROR"
                )
                user.update_usage_stats(success=False, context=job.processing_context)
                await db.commit()
            
            raise
    
    def _build_accessibility_prompt(self, user: User, job: ProcessingJob) -> AccessibilityPrompt:
        """Build accessibility prompt based on user preferences and context"""
        
        # Base accessibility analysis prompt
        base_prompt = """Analyze this image for accessibility and provide comprehensive metadata 
        that will help visually impaired users understand and interact with the content."""
        
        # Context-specific prompts
        context_prompts = {
            "code": """This appears to be a programming/code interface. Focus on:
            - Code structure and syntax
            - IDE elements (menus, panels, error messages)
            - Variable names and function definitions
            - Debugging information""",
            
            "academic": """This appears to be academic/research content. Focus on:
            - Research data and charts
            - Academic text and citations
            - Mathematical formulas or equations
            - Diagrams and illustrations""",
            
            "business": """This appears to be business/professional content. Focus on:
            - Business metrics and dashboards
            - Professional documents and presentations
            - Charts and business graphics
            - Form fields and business processes""",
            
            "general": """Provide a general accessibility analysis covering all visible elements."""
        }
        
        context_specific = context_prompts.get(job.processing_context, context_prompts["general"])
        
        # Get user preferences
        user_prefs = user.get_preferred_settings_for_context(job.processing_context)
        
        return AccessibilityPrompt(
            base_prompt=base_prompt,
            context_specific=context_specific,
            verbosity_level=job.verbosity_level,
            output_format=job.requested_output_format,
            user_preferences=user_prefs
        )
    
    def _select_provider(self, user: User, image_data: bytes, 
                        prompt: AccessibilityPrompt) -> Optional[str]:
        """Select best LLM provider for this request"""
        
        # Check user preferences for provider
        user_prefs = user.domain_specific_settings or {}
        context_prefs = user_prefs.get(prompt.context_specific, {})
        preferred_provider = context_prefs.get("preferred_llm_provider")
        
        if preferred_provider and preferred_provider in self.llm_manager.providers:
            return preferred_provider
        
        # Check if PII detected and user requires local processing
        contains_pii = self._detect_pii_in_image(image_data)
        if contains_pii and not user.can_use_cloud_processing(contains_pii=True):
            return "local"  # Force local processing
        
        # Otherwise, get cheapest provider
        prompt_length = len(prompt.base_prompt + prompt.context_specific)
        return self.llm_manager.get_cheapest_provider_for_task(
            len(image_data), prompt_length
        )
    
    def _detect_pii_in_image(self, image_data: bytes) -> bool:
        """Simple PII detection (placeholder - would need actual implementation)"""
        # This would use OCR + pattern matching to detect:
        # - Social security numbers
        # - Credit card numbers  
        # - Email addresses
        # - Names and addresses
        # For now, return False
        return False
    
    async def _check_cache(self, db: AsyncSession, job: ProcessingJob, 
                         image_data: bytes) -> Optional[Dict[str, Any]]:
        """Check if we have cached results for this image"""
        
        # Generate cache key
        cache_key = CacheEntry.generate_cache_key(
            job.image_hash,
            job.processing_context,
            job.verbosity_level,
            "any",  # We'll accept any provider for cache hits
            None
        )
        
        # Try to find cache entry with similar key
        result = await db.execute(
            select(CacheEntry).where(
                CacheEntry.image_hash == job.image_hash,
                CacheEntry.processing_context == job.processing_context,
                CacheEntry.verbosity_level == job.verbosity_level,
                CacheEntry.invalidated == False
            )
        )
        
        cache_entry = result.scalar_one_or_none()
        
        if cache_entry and cache_entry.is_valid:
            # Update cache access
            cache_entry.access()
            await db.commit()
            
            return cache_entry.get_accessibility_result()
        
        return None
    
    async def _create_annotation_from_cache(self, db: AsyncSession, 
                                          job: ProcessingJob,
                                          cache_data: Dict[str, Any]) -> AccessibilityAnnotation:
        """Create annotation from cached data"""
        
        annotation = AccessibilityAnnotation(
            job_id=job.id,
            alt_text=cache_data.get("alt_text", ""),
            detailed_description=cache_data.get("detailed_description"),
            brief_summary=cache_data.get("brief_summary"),
            headings=cache_data.get("structural_elements", {}).get("headings"),
            lists=cache_data.get("structural_elements", {}).get("lists"),
            tables=cache_data.get("structural_elements", {}).get("tables"),
            paragraphs=cache_data.get("structural_elements", {}).get("paragraphs"),
            buttons=cache_data.get("interactive_elements", {}).get("buttons"),
            links=cache_data.get("interactive_elements", {}).get("links"),
            form_fields=cache_data.get("interactive_elements", {}).get("form_fields"),
            images=cache_data.get("interactive_elements", {}).get("images"),
            reading_order=cache_data.get("reading_order"),
            text_content=cache_data.get("text_content"),
            layout_description=cache_data.get("layout_description"),
            color_information=cache_data.get("color_information"),
            confidence_score=cache_data.get("confidence_score", 0.8),
            processing_model="cached"
        )
        
        db.add(annotation)
        return annotation
    
    async def _create_annotation_from_llm_response(self, 
                                                 db: AsyncSession,
                                                 job: ProcessingJob,
                                                 response_data: Dict[str, Any],
                                                 llm_response: LLMResponse) -> AccessibilityAnnotation:
        """Create annotation from LLM response"""
        
        annotation = AccessibilityAnnotation(
            job_id=job.id,
            alt_text=response_data.get("alt_text", ""),
            detailed_description=response_data.get("detailed_description"),
            brief_summary=response_data.get("brief_summary"),
            
            # Structural elements
            headings=response_data.get("structural_elements", {}).get("headings"),
            lists=response_data.get("structural_elements", {}).get("lists"),
            tables=response_data.get("structural_elements", {}).get("tables"),
            paragraphs=response_data.get("structural_elements", {}).get("paragraphs"),
            
            # Interactive elements
            buttons=response_data.get("interactive_elements", {}).get("buttons"),
            links=response_data.get("interactive_elements", {}).get("links"),
            form_fields=response_data.get("interactive_elements", {}).get("form_fields"),
            images=response_data.get("interactive_elements", {}).get("images"),
            
            # Navigation and content
            reading_order=response_data.get("reading_order"),
            text_content=response_data.get("text_content"),
            layout_description=response_data.get("layout_description"),
            color_information=response_data.get("color_information"),
            
            # Domain-specific analysis
            code_analysis=response_data.get("code_analysis"),
            academic_analysis=response_data.get("academic_analysis"),
            business_analysis=response_data.get("business_analysis"),
            
            # Quality metrics
            confidence_score=llm_response.confidence,
            processing_model=f"{llm_response.provider}:{llm_response.model}",
            token_count={"total": llm_response.tokens_used}
        )
        
        db.add(annotation)
        return annotation
    
    async def _cache_results(self, db: AsyncSession, job: ProcessingJob,
                           image_data: bytes, response_data: Dict[str, Any],
                           llm_response: LLMResponse):
        """Cache processing results"""
        
        cache_key = CacheEntry.generate_cache_key(
            job.image_hash,
            job.processing_context,
            job.verbosity_level,
            llm_response.provider,
            llm_response.model
        )
        
        # Prepare cache data
        cache_data = {
            **response_data,
            "confidence_score": llm_response.confidence,
            "processing_metadata": llm_response.metadata
        }
        
        cache_entry = CacheEntry(
            cache_key=cache_key,
            image_hash=job.image_hash,
            processing_context=job.processing_context,
            verbosity_level=job.verbosity_level,
            llm_provider=llm_response.provider,
            model_used=llm_response.model,
            accessibility_result=cache_data,
            cache_size_bytes=len(json.dumps(cache_data)),
            confidence_score=int(llm_response.confidence * 100)
        )
        
        # Set expiration (24 hours for high confidence, 6 hours for low)
        if llm_response.confidence > 0.8:
            cache_entry.set_expiration(hours=24)
        else:
            cache_entry.set_expiration(hours=6)
        
        db.add(cache_entry)
        job.result_cache_key = cache_key