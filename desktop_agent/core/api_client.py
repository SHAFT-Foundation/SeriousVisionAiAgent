"""
Async HTTP client for communicating with Vision Agent server
"""
import base64
import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import aiohttp
import json
from aiohttp import ClientTimeout, ClientError

logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    """Server configuration"""
    base_url: str = "http://localhost:8000"
    api_version: str = "v1"
    timeout: int = 30
    max_retries: int = 3
    
    @property
    def api_base_url(self) -> str:
        return f"{self.base_url}/api/{self.api_version}"


@dataclass
class ProcessingRequest:
    """Request for image processing"""
    image_data: bytes
    user_id: str
    context: str = "general"
    verbosity: str = "medium"
    output_format: str = "text"
    force_reprocess: bool = False


@dataclass
class ProcessingResponse:
    """Response from image processing"""
    success: bool
    job_id: Optional[str] = None
    status: Optional[str] = None
    alt_text: Optional[str] = None
    detailed_description: Optional[str] = None
    structural_elements: Optional[Dict[str, Any]] = None
    interactive_elements: Optional[List[Dict[str, Any]]] = None
    reading_order: Optional[List[str]] = None
    confidence_score: Optional[float] = None
    processing_time_ms: Optional[int] = None
    provider_used: Optional[str] = None
    was_cached: bool = False
    error_message: Optional[str] = None


class VisionAgentClient:
    """Client for communicating with Vision Agent server"""
    
    def __init__(self, config: ServerConfig = None):
        """Initialize client with server configuration"""
        self.config = config or ServerConfig()
        self.session: Optional[aiohttp.ClientSession] = None
        self._setup_session()
        
        logger.info(f"Vision Agent client initialized for {self.config.base_url}")
    
    def _setup_session(self):
        """Setup aiohttp session with proper configuration"""
        timeout = ClientTimeout(total=self.config.timeout)
        
        connector = aiohttp.TCPConnector(
            limit=10,  # Max connections
            limit_per_host=5,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={
                "User-Agent": "VisionAgent-Desktop/1.0",
                "Content-Type": "application/json"
            }
        )
    
    async def close(self):
        """Close the client session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check server health"""
        try:
            url = f"{self.config.api_base_url}/health"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {"success": True, "data": data}
                else:
                    return {
                        "success": False,
                        "error": f"Health check failed with status {response.status}"
                    }
                    
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def process_image(self, request: ProcessingRequest) -> ProcessingResponse:
        """
        Process image with server
        
        Args:
            request: Processing request with image and parameters
        
        Returns:
            ProcessingResponse with results or error
        """
        try:
            # Encode image to base64
            image_b64 = base64.b64encode(request.image_data).decode('utf-8')
            
            # Prepare request payload
            payload = {
                "image_data": image_b64,
                "user_id": request.user_id,
                "context": request.context,
                "verbosity": request.verbosity,
                "output_format": request.output_format,
                "force_reprocess": request.force_reprocess
            }
            
            url = f"{self.config.api_base_url}/process"
            
            # Make request with retries
            for attempt in range(self.config.max_retries):
                try:
                    async with self.session.post(url, json=payload) as response:
                        response_data = await response.json()
                        
                        if response.status == 200:
                            return ProcessingResponse(
                                success=True,
                                job_id=response_data.get("job_id"),
                                status=response_data.get("status"),
                                alt_text=response_data.get("alt_text"),
                                detailed_description=response_data.get("detailed_description"),
                                structural_elements=response_data.get("structural_elements"),
                                interactive_elements=response_data.get("interactive_elements"),
                                reading_order=response_data.get("reading_order"),
                                confidence_score=response_data.get("confidence_score"),
                                processing_time_ms=response_data.get("processing_time_ms"),
                                provider_used=response_data.get("provider_used"),
                                was_cached=response_data.get("was_cached", False)
                            )
                        else:
                            error_msg = response_data.get("detail", f"HTTP {response.status}")
                            if attempt == self.config.max_retries - 1:
                                return ProcessingResponse(
                                    success=False,
                                    error_message=error_msg
                                )
                            
                            # Wait before retry
                            await asyncio.sleep(2 ** attempt)
                            
                except ClientError as e:
                    if attempt == self.config.max_retries - 1:
                        return ProcessingResponse(
                            success=False,
                            error_message=f"Network error: {str(e)}"
                        )
                    
                    await asyncio.sleep(2 ** attempt)
                    continue
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return ProcessingResponse(
                success=False,
                error_message=str(e)
            )
    
    async def upload_and_process(self, image_data: bytes, user_id: str,
                               context: str = "general", verbosity: str = "medium") -> ProcessingResponse:
        """
        Upload image file and process it
        
        Args:
            image_data: Image file bytes
            user_id: User identifier
            context: Processing context
            verbosity: Detail level
        
        Returns:
            ProcessingResponse with results
        """
        try:
            # Prepare multipart form data
            data = aiohttp.FormData()
            data.add_field('file', image_data, filename='image.png', content_type='image/png')
            data.add_field('user_id', user_id)
            data.add_field('context', context)
            data.add_field('verbosity', verbosity)
            data.add_field('output_format', 'text')
            
            url = f"{self.config.api_base_url}/process/upload"
            
            async with self.session.post(url, data=data) as response:
                response_data = await response.json()
                
                if response.status == 200:
                    return ProcessingResponse(
                        success=True,
                        job_id=response_data.get("job_id"),
                        status=response_data.get("status"),
                        alt_text=response_data.get("alt_text"),
                        detailed_description=response_data.get("detailed_description"),
                        structural_elements=response_data.get("structural_elements"),
                        interactive_elements=response_data.get("interactive_elements"),
                        reading_order=response_data.get("reading_order"),
                        confidence_score=response_data.get("confidence_score"),
                        processing_time_ms=response_data.get("processing_time_ms"),
                        provider_used=response_data.get("provider_used"),
                        was_cached=response_data.get("was_cached", False)
                    )
                else:
                    error_msg = response_data.get("detail", f"HTTP {response.status}")
                    return ProcessingResponse(success=False, error_message=error_msg)
                    
        except Exception as e:
            logger.error(f"Error uploading image: {e}")
            return ProcessingResponse(success=False, error_message=str(e))
    
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get processing job status"""
        try:
            url = f"{self.config.api_base_url}/process/job/{job_id}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {"success": True, "data": data}
                else:
                    response_data = await response.json()
                    error_msg = response_data.get("detail", f"HTTP {response.status}")
                    return {"success": False, "error": error_msg}
                    
        except Exception as e:
            logger.error(f"Error getting job status: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user accessibility preferences"""
        try:
            url = f"{self.config.api_base_url}/users/{user_id}/preferences"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {"success": True, "data": data}
                else:
                    response_data = await response.json()
                    error_msg = response_data.get("detail", f"HTTP {response.status}")
                    return {"success": False, "error": error_msg}
                    
        except Exception as e:
            logger.error(f"Error getting user preferences: {e}")
            return {"success": False, "error": str(e)}
    
    async def update_user_preferences(self, user_id: str, 
                                    preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Update user accessibility preferences"""
        try:
            url = f"{self.config.api_base_url}/users/{user_id}/preferences"
            
            async with self.session.post(url, json=preferences) as response:
                if response.status == 200:
                    data = await response.json()
                    return {"success": True, "data": data}
                else:
                    response_data = await response.json()
                    error_msg = response_data.get("detail", f"HTTP {response.status}")
                    return {"success": False, "error": error_msg}
                    
        except Exception as e:
            logger.error(f"Error updating user preferences: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get user usage statistics"""
        try:
            url = f"{self.config.api_base_url}/users/{user_id}/stats"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {"success": True, "data": data}
                else:
                    response_data = await response.json()
                    error_msg = response_data.get("detail", f"HTTP {response.status}")
                    return {"success": False, "error": error_msg}
                    
        except Exception as e:
            logger.error(f"Error getting user stats: {e}")
            return {"success": False, "error": str(e)}
    
    async def submit_feedback(self, user_id: str, job_id: str, 
                            rating: float, feedback_text: str = None) -> Dict[str, Any]:
        """Submit user feedback for a processing job"""
        try:
            url = f"{self.config.api_base_url}/users/{user_id}/feedback"
            
            payload = {
                "job_id": job_id,
                "rating": rating
            }
            
            if feedback_text:
                payload["feedback_text"] = feedback_text
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return {"success": True, "data": data}
                else:
                    response_data = await response.json()
                    error_msg = response_data.get("detail", f"HTTP {response.status}")
                    return {"success": False, "error": error_msg}
                    
        except Exception as e:
            logger.error(f"Error submitting feedback: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_connection(self) -> bool:
        """Test connection to server"""
        try:
            result = await self.health_check()
            return result["success"]
        except Exception:
            return False


# Utility functions for common operations
async def quick_process_image(image_data: bytes, user_id: str, 
                            server_url: str = "http://localhost:8000",
                            context: str = "general") -> ProcessingResponse:
    """Quick image processing without managing client lifecycle"""
    config = ServerConfig(base_url=server_url)
    
    async with VisionAgentClient(config) as client:
        request = ProcessingRequest(
            image_data=image_data,
            user_id=user_id,
            context=context
        )
        return await client.process_image(request)


async def test_server_connection(server_url: str = "http://localhost:8000") -> bool:
    """Test if server is reachable"""
    config = ServerConfig(base_url=server_url)
    
    async with VisionAgentClient(config) as client:
        return await client.test_connection()