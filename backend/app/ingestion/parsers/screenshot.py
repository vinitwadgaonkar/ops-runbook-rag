"""
Screenshot parser for dashboard images using OCR and GPT-4 Vision API.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import base64
import io
from PIL import Image
import pytesseract
import openai
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DashboardElement:
    """Represents an element extracted from a dashboard screenshot."""
    element_type: str  # "metric", "chart", "alert", "status", "text"
    content: str
    position: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float
    metadata: Dict[str, Any]


@dataclass
class OCRResult:
    """Result from OCR processing."""
    text: str
    confidence: float
    bounding_boxes: List[Tuple[int, int, int, int]]
    words: List[Dict[str, Any]]


class ScreenshotParser:
    """
    Parser for dashboard screenshots using OCR and GPT-4 Vision API.
    Extracts operational metrics, alerts, and status information.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.openai_api_key = openai_api_key
        if openai_api_key:
            openai.api_key = openai_api_key
        
        # Configure Tesseract for better OCR
        self.tesseract_config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,:;!?()[]{}"\'/\\-+=*%$#@&^|~`'
    
    def parse(self, image_path: str, use_vision_api: bool = True) -> Dict[str, Any]:
        """
        Parse screenshot and extract operational information.
        
        Args:
            image_path: Path to screenshot image
            use_vision_api: Whether to use GPT-4 Vision API in addition to OCR
            
        Returns:
            Dictionary with parsed content and metadata
        """
        try:
            # Load and validate image
            image = self._load_image(image_path)
            
            # Extract basic image metadata
            image_metadata = self._extract_image_metadata(image)
            
            # Perform OCR extraction
            ocr_result = self._extract_ocr(image)
            
            # Extract dashboard elements using OCR
            dashboard_elements = self._extract_dashboard_elements(ocr_result)
            
            # Use GPT-4 Vision API if available
            vision_result = None
            if use_vision_api and self.openai_api_key:
                try:
                    vision_result = self._extract_with_vision_api(image)
                except Exception as e:
                    logger.warning(f"Vision API failed: {e}")
            
            # Combine OCR and Vision results
            combined_content = self._combine_extraction_results(ocr_result, vision_result)
            
            # Extract operational metrics
            metrics = self._extract_metrics(combined_content)
            
            # Extract alerts and status information
            alerts = self._extract_alerts(combined_content)
            status_info = self._extract_status_info(combined_content)
            
            # Generate research metadata
            metadata = self._generate_research_metadata(
                image_metadata, dashboard_elements, metrics, alerts, status_info
            )
            
            return {
                "content": combined_content,
                "ocr_result": self._ocr_result_to_dict(ocr_result),
                "vision_result": vision_result,
                "dashboard_elements": [self._element_to_dict(elem) for elem in dashboard_elements],
                "metrics": metrics,
                "alerts": alerts,
                "status_info": status_info,
                "metadata": metadata,
                "source_path": image_path
            }
            
        except Exception as e:
            logger.error(f"Failed to parse screenshot {image_path}: {e}")
            raise
    
    def _load_image(self, image_path: str) -> Image.Image:
        """Load and validate image."""
        try:
            image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize if too large (for better OCR performance)
            max_size = 2000
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            return image
        except Exception as e:
            raise ValueError(f"Failed to load image: {e}")
    
    def _extract_image_metadata(self, image: Image.Image) -> Dict[str, Any]:
        """Extract basic metadata from image."""
        return {
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "format": image.format,
            "size_bytes": len(image.tobytes())
        }
    
    def _extract_ocr(self, image: Image.Image) -> OCRResult:
        """Extract text using OCR."""
        try:
            # Get detailed OCR data
            ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            # Extract text and confidence
            text = pytesseract.image_to_string(image, config=self.tesseract_config)
            confidence_scores = [int(conf) for conf in ocr_data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
            
            # Extract bounding boxes and words
            words = []
            bounding_boxes = []
            
            for i in range(len(ocr_data['text'])):
                if int(ocr_data['conf'][i]) > 30:  # Only high-confidence text
                    word_data = {
                        'text': ocr_data['text'][i],
                        'confidence': int(ocr_data['conf'][i]),
                        'bbox': (
                            ocr_data['left'][i],
                            ocr_data['top'][i],
                            ocr_data['width'][i],
                            ocr_data['height'][i]
                        )
                    }
                    words.append(word_data)
                    bounding_boxes.append(word_data['bbox'])
            
            return OCRResult(
                text=text.strip(),
                confidence=avg_confidence / 100.0,
                bounding_boxes=bounding_boxes,
                words=words
            )
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return OCRResult(text="", confidence=0.0, bounding_boxes=[], words=[])
    
    def _extract_dashboard_elements(self, ocr_result: OCRResult) -> List[DashboardElement]:
        """Extract dashboard elements from OCR result."""
        elements = []
        
        # Look for common dashboard patterns
        dashboard_patterns = {
            "metric": r'\d+(?:\.\d+)?%?',
            "alert": r'(?:alert|warning|error|critical)',
            "status": r'(?:up|down|healthy|unhealthy|running|stopped)',
            "chart": r'(?:chart|graph|plot)',
            "time": r'\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AP]M)?'
        }
        
        for word_data in ocr_result.words:
            text = word_data['text'].lower()
            confidence = word_data['confidence'] / 100.0
            
            for element_type, pattern in dashboard_patterns.items():
                if re.search(pattern, text, re.IGNORECASE):
                    element = DashboardElement(
                        element_type=element_type,
                        content=word_data['text'],
                        position=word_data['bbox'],
                        confidence=confidence,
                        metadata={'pattern': pattern}
                    )
                    elements.append(element)
                    break
        
        return elements
    
    def _extract_with_vision_api(self, image: Image.Image) -> Optional[Dict[str, Any]]:
        """Extract information using GPT-4 Vision API."""
        try:
            # Convert image to base64
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Create vision API request
            response = openai.ChatCompletion.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Analyze this dashboard screenshot and extract:
                                1. All visible metrics and their values
                                2. Any alerts or warnings
                                3. System status information
                                4. Chart types and data trends
                                5. Time stamps or time ranges
                                6. Service names and components
                                
                                Return the information in a structured JSON format."""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000
            )
            
            content = response.choices[0].message.content
            
            # Try to parse JSON response
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {"raw_content": content}
                
        except Exception as e:
            logger.error(f"Vision API extraction failed: {e}")
            return None
    
    def _combine_extraction_results(self, ocr_result: OCRResult, vision_result: Optional[Dict[str, Any]]) -> str:
        """Combine OCR and Vision API results."""
        combined_text = ocr_result.text
        
        if vision_result:
            if "raw_content" in vision_result:
                combined_text += "\n\n" + vision_result["raw_content"]
            else:
                # Add structured data from vision API
                for key, value in vision_result.items():
                    combined_text += f"\n{key}: {value}"
        
        return combined_text
    
    def _extract_metrics(self, content: str) -> List[Dict[str, Any]]:
        """Extract operational metrics from content."""
        metrics = []
        
        # Common metric patterns
        metric_patterns = [
            r'(\w+):\s*(\d+(?:\.\d+)?%?)',  # "CPU: 85%"
            r'(\d+(?:\.\d+)?%?)\s*(\w+)',  # "85% CPU"
            r'(\w+)\s*=\s*(\d+(?:\.\d+)?%?)',  # "CPU = 85%"
        ]
        
        for pattern in metric_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if len(match) == 2:
                    name, value = match
                    metrics.append({
                        "name": name.strip(),
                        "value": value.strip(),
                        "type": "metric",
                        "confidence": 0.8
                    })
        
        return metrics
    
    def _extract_alerts(self, content: str) -> List[Dict[str, Any]]:
        """Extract alerts and warnings from content."""
        alerts = []
        
        # Alert patterns
        alert_patterns = [
            r'(alert|warning|error|critical):\s*(.+)',
            r'(down|unhealthy|failed):\s*(.+)',
            r'(\w+)\s*(alert|warning|error)',
        ]
        
        for pattern in alert_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                alerts.append({
                    "type": match[0].strip(),
                    "message": match[1].strip() if len(match) > 1 else "",
                    "severity": self._classify_alert_severity(match[0])
                })
        
        return alerts
    
    def _extract_status_info(self, content: str) -> List[Dict[str, Any]]:
        """Extract system status information."""
        status_info = []
        
        # Status patterns
        status_patterns = [
            r'(up|down|healthy|unhealthy|running|stopped)',
            r'(online|offline)',
            r'(active|inactive)',
        ]
        
        for pattern in status_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                status_info.append({
                    "status": match.strip(),
                    "type": "status"
                })
        
        return status_info
    
    def _classify_alert_severity(self, alert_text: str) -> str:
        """Classify alert severity based on text."""
        alert_text = alert_text.lower()
        
        if any(word in alert_text for word in ['critical', 'fatal', 'down']):
            return 'critical'
        elif any(word in alert_text for word in ['error', 'failed', 'unhealthy']):
            return 'high'
        elif any(word in alert_text for word in ['warning', 'caution']):
            return 'medium'
        else:
            return 'low'
    
    def _generate_research_metadata(self, image_metadata: Dict[str, Any], 
                                  dashboard_elements: List[DashboardElement],
                                  metrics: List[Dict[str, Any]], alerts: List[Dict[str, Any]],
                                  status_info: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate research-specific metadata."""
        return {
            "document_type": "screenshot",
            "image_width": image_metadata.get("width", 0),
            "image_height": image_metadata.get("height", 0),
            "element_count": len(dashboard_elements),
            "metric_count": len(metrics),
            "alert_count": len(alerts),
            "status_count": len(status_info),
            "has_alerts": len(alerts) > 0,
            "has_metrics": len(metrics) > 0,
            "has_status": len(status_info) > 0,
            "complexity_score": self._calculate_screenshot_complexity(
                dashboard_elements, metrics, alerts
            ),
            "element_types": list(set(elem.element_type for elem in dashboard_elements)),
            "alert_severities": list(set(alert.get("severity", "unknown") for alert in alerts))
        }
    
    def _calculate_screenshot_complexity(self, elements: List[DashboardElement],
                                       metrics: List[Dict[str, Any]], 
                                       alerts: List[Dict[str, Any]]) -> float:
        """Calculate complexity score for screenshot."""
        score = 0.0
        
        # Base score from element count
        score += len(elements) * 0.1
        
        # Add score for metrics
        score += len(metrics) * 0.2
        
        # Add score for alerts
        score += len(alerts) * 0.3
        
        # Add score for element diversity
        element_types = set(elem.element_type for elem in elements)
        score += len(element_types) * 0.2
        
        return min(score, 10.0)
    
    def _ocr_result_to_dict(self, ocr_result: OCRResult) -> Dict[str, Any]:
        """Convert OCR result to dictionary."""
        return {
            "text": ocr_result.text,
            "confidence": ocr_result.confidence,
            "word_count": len(ocr_result.words),
            "bounding_boxes": ocr_result.bounding_boxes
        }
    
    def _element_to_dict(self, element: DashboardElement) -> Dict[str, Any]:
        """Convert dashboard element to dictionary."""
        return {
            "element_type": element.element_type,
            "content": element.content,
            "position": element.position,
            "confidence": element.confidence,
            "metadata": element.metadata
        }
    
    def parse_file(self, image_path: str, use_vision_api: bool = True) -> Dict[str, Any]:
        """Parse a screenshot file from disk."""
        return self.parse(image_path, use_vision_api)
    
    def extract_research_features(self, parsed_content: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features useful for research analysis."""
        metadata = parsed_content.get("metadata", {})
        
        return {
            "document_type": metadata.get("document_type", "screenshot"),
            "element_count": metadata.get("element_count", 0),
            "metric_count": metadata.get("metric_count", 0),
            "alert_count": metadata.get("alert_count", 0),
            "has_alerts": metadata.get("has_alerts", False),
            "has_metrics": metadata.get("has_metrics", False),
            "complexity_score": metadata.get("complexity_score", 0),
            "element_types": metadata.get("element_types", []),
            "alert_severities": metadata.get("alert_severities", [])
        }
