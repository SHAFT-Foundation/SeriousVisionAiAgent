"""
Accessibility annotation model for Vision Agent
Stores the processed accessibility metadata and descriptions
"""
from typing import Dict, Any, List, Optional
from sqlalchemy import Column, String, Text, JSON, Float, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from .base import Base


class AccessibilityAnnotation(Base):
    __tablename__ = "accessibility_annotations"
    
    # Relationship to processing job
    job_id = Column(UUID(as_uuid=True), ForeignKey("processing_jobs.id"), nullable=False, unique=True, index=True)
    processing_job = relationship("ProcessingJob", back_populates="accessibility_annotation")
    
    # Core Accessibility Content
    alt_text = Column(Text, nullable=False)  # Primary alternative text description
    detailed_description = Column(Text, nullable=True)  # Extended detailed description
    brief_summary = Column(Text, nullable=True)  # Very brief one-line summary
    
    # Structural Analysis
    headings = Column(JSON, nullable=True)  # [{"level": 1, "text": "Main Title", "position": {"x": 100, "y": 50}}]
    lists = Column(JSON, nullable=True)     # [{"type": "unordered", "items": ["Item 1", "Item 2"], "position": {...}}]
    tables = Column(JSON, nullable=True)    # [{"headers": ["Col1", "Col2"], "rows": [["A", "B"]], "caption": "..."}]
    paragraphs = Column(JSON, nullable=True) # [{"text": "Paragraph content", "position": {...}}]
    
    # Interactive Elements
    buttons = Column(JSON, nullable=True)   # [{"text": "Submit", "position": {"x": 200, "y": 300}, "enabled": true}]
    links = Column(JSON, nullable=True)     # [{"text": "Learn More", "url": "example.com", "position": {...}}]
    form_fields = Column(JSON, nullable=True) # [{"type": "input", "label": "Email", "required": true, "value": "..."}]
    images = Column(JSON, nullable=True)    # [{"alt": "Description", "position": {...}, "size": {...}}]
    
    # Navigation and Structure
    reading_order = Column(JSON, nullable=True)  # ["header", "paragraph1", "button", "footer"]
    focus_sequence = Column(JSON, nullable=True) # For keyboard navigation
    landmarks = Column(JSON, nullable=True)      # Page regions: header, main, nav, aside, footer
    
    # Content Analysis
    text_content = Column(Text, nullable=True)   # All extracted text content
    color_information = Column(JSON, nullable=True)  # Color descriptions and contrasts
    layout_description = Column(Text, nullable=True) # Overall layout and visual hierarchy
    
    # Domain-Specific Analysis
    code_analysis = Column(JSON, nullable=True)     # For programming/code content
    academic_analysis = Column(JSON, nullable=True)  # For academic/research content
    business_analysis = Column(JSON, nullable=True)  # For business/professional content
    
    # Quality and Confidence Metrics
    confidence_score = Column(Float, nullable=False)  # Overall confidence 0.0-1.0
    text_confidence = Column(Float, nullable=True)    # OCR text confidence
    structure_confidence = Column(Float, nullable=True) # UI structure detection confidence
    
    # Review and Validation
    human_reviewed = Column(Boolean, default=False)
    reviewer_notes = Column(Text, nullable=True)
    accuracy_rating = Column(Float, nullable=True)  # Human-validated accuracy 1-5
    
    # Processing Metadata
    processing_model = Column(String(100), nullable=True)  # Model used for analysis
    processing_prompt = Column(Text, nullable=True)        # Prompt used for LLM
    token_count = Column(JSON, nullable=True)              # Token usage breakdown
    
    def __repr__(self) -> str:
        return f"<AccessibilityAnnotation(job_id={self.job_id}, confidence={self.confidence_score:.2f})>"
    
    def get_description_by_verbosity(self, verbosity: str = "medium") -> str:
        """Get appropriate description based on verbosity level"""
        if verbosity == "brief" and self.brief_summary:
            return self.brief_summary
        elif verbosity == "detailed" and self.detailed_description:
            return self.detailed_description
        else:
            return self.alt_text
    
    def get_structural_elements(self) -> Dict[str, Any]:
        """Get all structural elements in a unified format"""
        return {
            "headings": self.headings or [],
            "lists": self.lists or [],
            "tables": self.tables or [],
            "paragraphs": self.paragraphs or [],
            "buttons": self.buttons or [],
            "links": self.links or [],
            "form_fields": self.form_fields or [],
            "images": self.images or []
        }
    
    def get_interactive_elements(self) -> List[Dict[str, Any]]:
        """Get all interactive elements for keyboard navigation"""
        interactive = []
        
        # Add buttons
        for button in (self.buttons or []):
            interactive.append({
                "type": "button",
                "text": button.get("text", ""),
                "position": button.get("position", {}),
                "enabled": button.get("enabled", True)
            })
        
        # Add links
        for link in (self.links or []):
            interactive.append({
                "type": "link",
                "text": link.get("text", ""),
                "url": link.get("url", ""),
                "position": link.get("position", {})
            })
        
        # Add form fields
        for field in (self.form_fields or []):
            interactive.append({
                "type": "form_field",
                "label": field.get("label", ""),
                "field_type": field.get("type", "input"),
                "required": field.get("required", False),
                "position": field.get("position", {})
            })
        
        # Sort by reading order if available
        if self.focus_sequence:
            # Create a mapping for sorting
            order_map = {elem: i for i, elem in enumerate(self.focus_sequence)}
            interactive.sort(key=lambda x: order_map.get(x.get("text", ""), 999))
        
        return interactive
    
    def get_navigation_structure(self) -> Dict[str, Any]:
        """Get navigation and structure information"""
        return {
            "reading_order": self.reading_order or [],
            "focus_sequence": self.focus_sequence or [],
            "landmarks": self.landmarks or {},
            "layout_description": self.layout_description
        }
    
    def get_domain_specific_analysis(self, domain: str) -> Optional[Dict[str, Any]]:
        """Get domain-specific analysis if available"""
        domain_map = {
            "code": self.code_analysis,
            "academic": self.academic_analysis,
            "business": self.business_analysis
        }
        return domain_map.get(domain)
    
    def add_human_review(self, accuracy_rating: float, reviewer_notes: str = None) -> None:
        """Add human review and validation"""
        self.human_reviewed = True
        self.accuracy_rating = max(1.0, min(5.0, accuracy_rating))  # Clamp to 1-5
        self.reviewer_notes = reviewer_notes
    
    def has_high_confidence(self, threshold: float = 0.8) -> bool:
        """Check if annotation has high confidence"""
        return self.confidence_score >= threshold
    
    def get_quality_metrics(self) -> Dict[str, Any]:
        """Get quality and confidence metrics"""
        return {
            "overall_confidence": self.confidence_score,
            "text_confidence": self.text_confidence,
            "structure_confidence": self.structure_confidence,
            "human_reviewed": self.human_reviewed,
            "accuracy_rating": self.accuracy_rating,
            "has_high_confidence": self.has_high_confidence()
        }