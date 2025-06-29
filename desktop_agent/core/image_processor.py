"""
Image preprocessing and optimization for Vision Agent
"""
import logging
from typing import Tuple, Optional, Dict, Any
from io import BytesIO
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import cv2
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Configuration for image processing"""
    max_size: int = 2048
    jpeg_quality: int = 85
    enhance_contrast: float = 1.2
    enhance_sharpness: float = 1.1
    denoise_strength: int = 5
    edge_enhance: bool = True
    auto_crop_whitespace: bool = True
    convert_to_grayscale: bool = False


@dataclass
class ProcessedImage:
    """Result of image processing"""
    image_data: bytes
    original_size: Tuple[int, int]
    processed_size: Tuple[int, int]
    file_size_original: int
    file_size_processed: int
    processing_metadata: Dict[str, Any]


class ImageProcessor:
    """Handles image preprocessing and optimization"""
    
    def __init__(self, config: ProcessingConfig = None):
        """Initialize with processing configuration"""
        self.config = config or ProcessingConfig()
        logger.info("Image processor initialized")
    
    def process_image(self, image_data: bytes, 
                     content_type: str = "general") -> ProcessedImage:
        """
        Process image for optimal LLM analysis
        
        Args:
            image_data: Raw image bytes
            content_type: Type of content (general, text, diagram, ui)
        
        Returns:
            ProcessedImage with optimized data
        """
        original_size_bytes = len(image_data)
        
        # Load image
        img = Image.open(BytesIO(image_data))
        original_size = img.size
        
        # Apply content-specific preprocessing
        if content_type == "text":
            img = self._preprocess_text_image(img)
        elif content_type == "diagram":
            img = self._preprocess_diagram_image(img)
        elif content_type == "ui":
            img = self._preprocess_ui_image(img)
        else:
            img = self._preprocess_general_image(img)
        
        # Convert to RGB if necessary
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Resize if needed
        img = self._resize_image(img)
        
        # Auto-crop whitespace if enabled
        if self.config.auto_crop_whitespace:
            img = self._autocrop_whitespace(img)
        
        # Save processed image
        output_buffer = BytesIO()
        img.save(output_buffer, format="JPEG", quality=self.config.jpeg_quality, optimize=True)
        processed_data = output_buffer.getvalue()
        
        metadata = {
            "content_type": content_type,
            "preprocessing_applied": self._get_applied_preprocessing(content_type),
            "size_reduction": f"{(1 - len(processed_data) / original_size_bytes) * 100:.1f}%"
        }
        
        return ProcessedImage(
            image_data=processed_data,
            original_size=original_size,
            processed_size=img.size,
            file_size_original=original_size_bytes,
            file_size_processed=len(processed_data),
            processing_metadata=metadata
        )
    
    def _preprocess_general_image(self, img: Image.Image) -> Image.Image:
        """General image preprocessing"""
        # Enhance contrast slightly
        if self.config.enhance_contrast != 1.0:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(self.config.enhance_contrast)
        
        # Enhance sharpness
        if self.config.enhance_sharpness != 1.0:
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(self.config.enhance_sharpness)
        
        return img
    
    def _preprocess_text_image(self, img: Image.Image) -> Image.Image:
        """Preprocessing optimized for text content"""
        # Convert to numpy array for OpenCV operations
        img_array = np.array(img)
        
        # Convert to grayscale for better text recognition
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray, h=self.config.denoise_strength)
        
        # Apply adaptive thresholding for better text contrast
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Convert back to PIL Image
        img = Image.fromarray(thresh)
        
        # Enhance sharpness for text
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.5)
        
        return img
    
    def _preprocess_diagram_image(self, img: Image.Image) -> Image.Image:
        """Preprocessing optimized for diagrams and charts"""
        # Enhance contrast for better line visibility
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.3)
        
        # Edge enhancement for diagram elements
        if self.config.edge_enhance:
            img_array = np.array(img)
            
            # Apply edge detection
            edges = cv2.Canny(img_array, 50, 150)
            
            # Combine edges with original
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            enhanced = cv2.addWeighted(img_array, 0.8, edges_colored, 0.2, 0)
            
            img = Image.fromarray(enhanced)
        
        return img
    
    def _preprocess_ui_image(self, img: Image.Image) -> Image.Image:
        """Preprocessing optimized for UI screenshots"""
        # Moderate contrast enhancement
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.1)
        
        # Slight sharpness enhancement
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.2)
        
        # Ensure UI elements are crisp
        img_array = np.array(img)
        
        # Apply bilateral filter to reduce noise while keeping edges sharp
        filtered = cv2.bilateralFilter(img_array, 9, 75, 75)
        
        img = Image.fromarray(filtered)
        
        return img
    
    def _resize_image(self, img: Image.Image) -> Image.Image:
        """Resize image if it exceeds maximum dimensions"""
        max_size = self.config.max_size
        
        if img.width > max_size or img.height > max_size:
            # Calculate new size maintaining aspect ratio
            ratio = min(max_size / img.width, max_size / img.height)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            
            # Use high-quality resampling
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            logger.debug(f"Resized image from {img.size} to {new_size}")
        
        return img
    
    def _autocrop_whitespace(self, img: Image.Image, 
                           threshold: int = 250) -> Image.Image:
        """Automatically crop whitespace from image edges"""
        # Convert to grayscale for analysis
        gray = img.convert('L')
        gray_array = np.array(gray)
        
        # Find non-white pixels
        non_white = gray_array < threshold
        
        # Find bounding box of non-white pixels
        rows = np.any(non_white, axis=1)
        cols = np.any(non_white, axis=0)
        
        if np.any(rows) and np.any(cols):
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            
            # Add small padding
            padding = 10
            rmin = max(0, rmin - padding)
            rmax = min(img.height, rmax + padding)
            cmin = max(0, cmin - padding)
            cmax = min(img.width, cmax + padding)
            
            # Crop image
            img = img.crop((cmin, rmin, cmax, rmax))
            logger.debug(f"Auto-cropped to remove whitespace: new size {img.size}")
        
        return img
    
    def _get_applied_preprocessing(self, content_type: str) -> list:
        """Get list of preprocessing steps applied"""
        steps = ["resize", "quality_optimization"]
        
        if content_type == "text":
            steps.extend(["grayscale", "denoise", "adaptive_threshold", "sharpen"])
        elif content_type == "diagram":
            steps.extend(["contrast_enhance", "edge_enhance"])
        elif content_type == "ui":
            steps.extend(["contrast_enhance", "bilateral_filter", "sharpen"])
        else:
            steps.extend(["contrast_enhance", "sharpen"])
        
        if self.config.auto_crop_whitespace:
            steps.append("auto_crop")
        
        return steps
    
    def extract_text_regions(self, image_data: bytes) -> Dict[str, Any]:
        """Extract regions likely to contain text"""
        img = Image.open(BytesIO(image_data))
        img_array = np.array(img)
        
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply morphological operations to find text regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
        dilated = cv2.dilate(gray, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 50 and h > 10:  # Filter small regions
                text_regions.append({
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h)
                })
        
        return {
            "text_regions": text_regions,
            "total_regions": len(text_regions),
            "image_size": img.size
        }
    
    def prepare_for_ocr(self, image_data: bytes) -> bytes:
        """Prepare image specifically for OCR processing"""
        config = ProcessingConfig(
            convert_to_grayscale=True,
            enhance_contrast=1.5,
            enhance_sharpness=1.8,
            denoise_strength=10,
            edge_enhance=False
        )
        
        processor = ImageProcessor(config)
        result = processor.process_image(image_data, content_type="text")
        return result.image_data
    
    def create_thumbnail(self, image_data: bytes, 
                        size: Tuple[int, int] = (256, 256)) -> bytes:
        """Create thumbnail for preview or caching"""
        img = Image.open(BytesIO(image_data))
        img.thumbnail(size, Image.Resampling.LANCZOS)
        
        output_buffer = BytesIO()
        img.save(output_buffer, format="JPEG", quality=75)
        return output_buffer.getvalue()


# Utility functions
def optimize_for_llm(image_data: bytes, 
                    max_size: int = 2048,
                    quality: int = 85) -> bytes:
    """Quick optimization for LLM processing"""
    processor = ImageProcessor(ProcessingConfig(max_size=max_size, jpeg_quality=quality))
    result = processor.process_image(image_data)
    return result.image_data