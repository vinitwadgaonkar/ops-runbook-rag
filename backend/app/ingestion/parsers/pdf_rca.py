"""
PDF RCA parser for extracting structured information from incident reports.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import fitz  # PyMuPDF
import re
from dataclasses import dataclass
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class RCATimeline:
    """Represents a timeline entry in an RCA."""
    timestamp: str
    event: str
    description: str
    impact: Optional[str] = None
    action_taken: Optional[str] = None


@dataclass
class RCAAction:
    """Represents an action item from an RCA."""
    action_id: str
    description: str
    owner: Optional[str] = None
    due_date: Optional[str] = None
    status: str = "open"
    priority: str = "medium"


@dataclass
class RCAMetric:
    """Represents a metric or measurement from an RCA."""
    name: str
    value: str
    unit: Optional[str] = None
    context: Optional[str] = None


class PDFRCAParser:
    """
    Parser for PDF RCA documents with research-specific features.
    Extracts timeline, action items, metrics, and root cause analysis.
    """
    
    def __init__(self):
        self.timeline_patterns = [
            r'(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AP]M)?)\s+(.+)',
            r'(\d{4}-\d{2}-\d{2}\s+\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AP]M)?)\s+(.+)',
            r'(timeline|chronology)[:\s]*(.+)',
        ]
        
        self.action_patterns = [
            r'action\s*(\d+)[:\s]*(.+)',
            r'(\d+)\.\s*(.+)',
            r'todo[:\s]*(.+)',
            r'follow.?up[:\s]*(.+)',
        ]
        
        self.metric_patterns = [
            r'(\w+):\s*(\d+(?:\.\d+)?%?)',
            r'(\d+(?:\.\d+)?%?)\s*(\w+)',
            r'(\w+)\s*=\s*(\d+(?:\.\d+)?%?)',
        ]
    
    def parse(self, pdf_path: str) -> Dict[str, Any]:
        """
        Parse PDF RCA document and extract structured information.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with parsed content and metadata
        """
        try:
            # Open PDF document
            doc = fitz.open(pdf_path)
            
            # Extract text from all pages
            full_text = ""
            page_texts = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()
                page_texts.append({
                    "page_number": page_num + 1,
                    "text": page_text,
                    "char_count": len(page_text)
                })
                full_text += page_text + "\n"
            
            # Extract structured information
            timeline = self._extract_timeline(full_text)
            action_items = self._extract_action_items(full_text)
            metrics = self._extract_metrics(full_text)
            root_cause = self._extract_root_cause(full_text)
            impact_analysis = self._extract_impact_analysis(full_text)
            lessons_learned = self._extract_lessons_learned(full_text)
            
            # Extract metadata
            metadata = self._extract_metadata(doc, full_text, timeline, action_items, metrics)
            
            # Close document
            doc.close()
            
            return {
                "content": full_text,
                "page_texts": page_texts,
                "timeline": [self._timeline_to_dict(entry) for entry in timeline],
                "action_items": [self._action_to_dict(action) for action in action_items],
                "metrics": [self._metric_to_dict(metric) for metric in metrics],
                "root_cause": root_cause,
                "impact_analysis": impact_analysis,
                "lessons_learned": lessons_learned,
                "metadata": metadata,
                "source_path": pdf_path
            }
            
        except Exception as e:
            logger.error(f"Failed to parse PDF RCA {pdf_path}: {e}")
            raise
    
    def _extract_timeline(self, text: str) -> List[RCATimeline]:
        """Extract timeline entries from RCA text."""
        timeline = []
        
        # Split text into lines for processing
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Check for timeline patterns
            for pattern in self.timeline_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    timestamp = match.group(1)
                    event_description = match.group(2)
                    
                    # Look for additional context in following lines
                    context_lines = []
                    for j in range(i + 1, min(i + 3, len(lines))):
                        next_line = lines[j].strip()
                        if next_line and not re.search(r'^\d+[:\d\s]*[AP]M?$', next_line):
                            context_lines.append(next_line)
                        else:
                            break
                    
                    context = ' '.join(context_lines) if context_lines else ""
                    
                    timeline_entry = RCATimeline(
                        timestamp=timestamp,
                        event=event_description,
                        description=context,
                        impact=self._extract_impact_from_text(context),
                        action_taken=self._extract_action_from_text(context)
                    )
                    timeline.append(timeline_entry)
                    break
        
        return timeline
    
    def _extract_action_items(self, text: str) -> List[RCAAction]:
        """Extract action items from RCA text."""
        actions = []
        
        # Look for action item sections
        action_sections = re.findall(
            r'(?:action\s+items?|follow.?up\s+items?|recommendations?)[:\s]*(.*?)(?=\n\n|\n[A-Z]|\Z)',
            text, re.IGNORECASE | re.DOTALL
        )
        
        for section in action_sections:
            lines = section.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check for action patterns
                for pattern in self.action_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        action_id = match.group(1) if len(match.groups()) > 1 else f"action_{len(actions) + 1}"
                        description = match.group(2) if len(match.groups()) > 1 else match.group(1)
                        
                        action = RCAAction(
                            action_id=str(action_id),
                            description=description.strip(),
                            owner=self._extract_owner_from_text(description),
                            due_date=self._extract_due_date_from_text(description),
                            priority=self._classify_priority(description)
                        )
                        actions.append(action)
                        break
        
        return actions
    
    def _extract_metrics(self, text: str) -> List[RCAMetric]:
        """Extract metrics and measurements from RCA text."""
        metrics = []
        
        for pattern in self.metric_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) == 2:
                    name, value = match
                    metric = RCAMetric(
                        name=name.strip(),
                        value=value.strip(),
                        unit=self._extract_unit_from_value(value),
                        context=self._extract_context_for_metric(name, text)
                    )
                    metrics.append(metric)
        
        return metrics
    
    def _extract_root_cause(self, text: str) -> Dict[str, Any]:
        """Extract root cause analysis from text."""
        root_cause = {
            "primary_cause": "",
            "contributing_factors": [],
            "technical_details": "",
            "human_factors": [],
            "process_issues": []
        }
        
        # Look for root cause sections
        root_cause_sections = re.findall(
            r'(?:root\s+cause|root\s+analysis|cause\s+analysis)[:\s]*(.*?)(?=\n\n|\n[A-Z]|\Z)',
            text, re.IGNORECASE | re.DOTALL
        )
        
        for section in root_cause_sections:
            # Extract primary cause
            primary_match = re.search(r'(?:primary|main|root)\s+cause[:\s]*(.+)', section, re.IGNORECASE)
            if primary_match:
                root_cause["primary_cause"] = primary_match.group(1).strip()
            
            # Extract contributing factors
            factor_patterns = [
                r'contributing\s+factors?[:\s]*(.+)',
                r'additional\s+factors?[:\s]*(.+)',
                r'secondary\s+causes?[:\s]*(.+)'
            ]
            
            for pattern in factor_patterns:
                matches = re.findall(pattern, section, re.IGNORECASE)
                for match in matches:
                    factors = [f.strip() for f in match.split(',')]
                    root_cause["contributing_factors"].extend(factors)
        
        return root_cause
    
    def _extract_impact_analysis(self, text: str) -> Dict[str, Any]:
        """Extract impact analysis from text."""
        impact = {
            "affected_services": [],
            "user_impact": "",
            "business_impact": "",
            "duration": "",
            "severity": "unknown"
        }
        
        # Look for impact sections
        impact_sections = re.findall(
            r'(?:impact|affected|consequences?)[:\s]*(.*?)(?=\n\n|\n[A-Z]|\Z)',
            text, re.IGNORECASE | re.DOTALL
        )
        
        for section in impact_sections:
            # Extract affected services
            service_matches = re.findall(r'(\w+(?:-\w+)*\s+(?:service|api|system))', section, re.IGNORECASE)
            impact["affected_services"].extend([s.strip() for s in service_matches])
            
            # Extract user impact
            user_impact_match = re.search(r'user\s+impact[:\s]*(.+)', section, re.IGNORECASE)
            if user_impact_match:
                impact["user_impact"] = user_impact_match.group(1).strip()
            
            # Extract business impact
            business_impact_match = re.search(r'business\s+impact[:\s]*(.+)', section, re.IGNORECASE)
            if business_impact_match:
                impact["business_impact"] = business_impact_match.group(1).strip()
            
            # Extract duration
            duration_match = re.search(r'duration[:\s]*(\d+(?:\.\d+)?\s*(?:minutes?|hours?|days?))', section, re.IGNORECASE)
            if duration_match:
                impact["duration"] = duration_match.group(1).strip()
            
            # Extract severity
            severity_match = re.search(r'severity[:\s]*(critical|high|medium|low)', section, re.IGNORECASE)
            if severity_match:
                impact["severity"] = severity_match.group(1).lower()
        
        return impact
    
    def _extract_lessons_learned(self, text: str) -> List[str]:
        """Extract lessons learned from RCA text."""
        lessons = []
        
        # Look for lessons learned sections
        lessons_sections = re.findall(
            r'(?:lessons?\s+learned|key\s+takeaways?|recommendations?)[:\s]*(.*?)(?=\n\n|\n[A-Z]|\Z)',
            text, re.IGNORECASE | re.DOTALL
        )
        
        for section in lessons_sections:
            # Split by common delimiters
            lesson_items = re.split(r'[•\-\*\n]', section)
            for item in lesson_items:
                item = item.strip()
                if item and len(item) > 10:  # Filter out very short items
                    lessons.append(item)
        
        return lessons
    
    def _extract_metadata(self, doc, text: str, timeline: List[RCATimeline], 
                         actions: List[RCAAction], metrics: List[RCAMetric]) -> Dict[str, Any]:
        """Extract comprehensive metadata from RCA."""
        metadata = {
            "document_type": "rca",
            "page_count": len(doc),
            "char_count": len(text),
            "word_count": len(text.split()),
            "timeline_entries": len(timeline),
            "action_items": len(actions),
            "metrics_count": len(metrics),
            "has_timeline": len(timeline) > 0,
            "has_action_items": len(actions) > 0,
            "has_metrics": len(metrics) > 0,
            "complexity_score": self._calculate_rca_complexity(timeline, actions, metrics),
            "severity": self._extract_severity_from_text(text),
            "incident_type": self._classify_incident_type(text),
            "affected_components": self._extract_affected_components(text)
        }
        
        return metadata
    
    def _extract_impact_from_text(self, text: str) -> Optional[str]:
        """Extract impact information from text."""
        impact_patterns = [
            r'impact[:\s]*(.+)',
            r'affected[:\s]*(.+)',
            r'consequence[:\s]*(.+)'
        ]
        
        for pattern in impact_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_action_from_text(self, text: str) -> Optional[str]:
        """Extract action taken from text."""
        action_patterns = [
            r'action\s+taken[:\s]*(.+)',
            r'response[:\s]*(.+)',
            r'mitigation[:\s]*(.+)'
        ]
        
        for pattern in action_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_owner_from_text(self, text: str) -> Optional[str]:
        """Extract action owner from text."""
        owner_match = re.search(r'owner[:\s]*([^,\n]+)', text, re.IGNORECASE)
        if owner_match:
            return owner_match.group(1).strip()
        
        # Look for common owner patterns
        owner_patterns = [
            r'assigned\s+to[:\s]*([^,\n]+)',
            r'responsible[:\s]*([^,\n]+)',
            r'owner[:\s]*([^,\n]+)'
        ]
        
        for pattern in owner_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_due_date_from_text(self, text: str) -> Optional[str]:
        """Extract due date from text."""
        date_patterns = [
            r'due[:\s]*(\d{4}-\d{2}-\d{2})',
            r'deadline[:\s]*(\d{4}-\d{2}-\d{2})',
            r'by[:\s]*(\d{4}-\d{2}-\d{2})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _classify_priority(self, text: str) -> str:
        """Classify action priority based on text."""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['critical', 'urgent', 'immediate']):
            return 'high'
        elif any(word in text_lower for word in ['important', 'priority']):
            return 'medium'
        else:
            return 'low'
    
    def _extract_unit_from_value(self, value: str) -> Optional[str]:
        """Extract unit from metric value."""
        unit_match = re.search(r'(\d+(?:\.\d+)?)(%|ms|s|min|hour|day|GB|MB|KB)', value)
        if unit_match:
            return unit_match.group(2)
        return None
    
    def _extract_context_for_metric(self, name: str, text: str) -> Optional[str]:
        """Extract context for a metric."""
        # Look for sentences containing the metric name
        sentences = text.split('.')
        for sentence in sentences:
            if name.lower() in sentence.lower():
                return sentence.strip()
        return None
    
    def _calculate_rca_complexity(self, timeline: List[RCATimeline], 
                              actions: List[RCAAction], metrics: List[RCAMetric]) -> float:
        """Calculate complexity score for RCA."""
        score = 0.0
        
        # Base score from components
        score += len(timeline) * 0.1
        score += len(actions) * 0.2
        score += len(metrics) * 0.1
        
        # Add score for timeline complexity
        if timeline:
            avg_timeline_length = sum(len(entry.description) for entry in timeline) / len(timeline)
            score += min(avg_timeline_length / 100, 2.0)
        
        return min(score, 10.0)
    
    def _extract_severity_from_text(self, text: str) -> str:
        """Extract incident severity from text."""
        severity_patterns = [
            r'severity[:\s]*(critical|high|medium|low)',
            r'priority[:\s]*(critical|high|medium|low)',
            r'impact[:\s]*(critical|high|medium|low)'
        ]
        
        for pattern in severity_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).lower()
        
        return 'unknown'
    
    def _classify_incident_type(self, text: str) -> str:
        """Classify incident type based on content."""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['database', 'db', 'sql']):
            return 'database'
        elif any(word in text_lower for word in ['network', 'connectivity', 'dns']):
            return 'network'
        elif any(word in text_lower for word in ['deployment', 'release', 'rollout']):
            return 'deployment'
        elif any(word in text_lower for word in ['authentication', 'auth', 'login']):
            return 'authentication'
        elif any(word in text_lower for word in ['performance', 'slow', 'latency']):
            return 'performance'
        else:
            return 'general'
    
    def _extract_affected_components(self, text: str) -> List[str]:
        """Extract affected components from text."""
        components = []
        
        # Look for component patterns
        component_patterns = [
            r'affected\s+components?[:\s]*(.+)',
            r'impacted\s+services?[:\s]*(.+)',
            r'components?[:\s]*(.+)'
        ]
        
        for pattern in component_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Split by common delimiters
                items = re.split(r'[,;]', match)
                for item in items:
                    item = item.strip()
                    if item:
                        components.append(item)
        
        return components
    
    def _timeline_to_dict(self, timeline: RCATimeline) -> Dict[str, Any]:
        """Convert timeline entry to dictionary."""
        return {
            "timestamp": timeline.timestamp,
            "event": timeline.event,
            "description": timeline.description,
            "impact": timeline.impact,
            "action_taken": timeline.action_taken
        }
    
    def _action_to_dict(self, action: RCAAction) -> Dict[str, Any]:
        """Convert action to dictionary."""
        return {
            "action_id": action.action_id,
            "description": action.description,
            "owner": action.owner,
            "due_date": action.due_date,
            "status": action.status,
            "priority": action.priority
        }
    
    def _metric_to_dict(self, metric: RCAMetric) -> Dict[str, Any]:
        """Convert metric to dictionary."""
        return {
            "name": metric.name,
            "value": metric.value,
            "unit": metric.unit,
            "context": metric.context
        }
    
    def parse_file(self, pdf_path: str) -> Dict[str, Any]:
        """Parse a PDF RCA file from disk."""
        return self.parse(pdf_path)
    
    def extract_research_features(self, parsed_content: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features useful for research analysis."""
        metadata = parsed_content.get("metadata", {})
        
        return {
            "document_type": metadata.get("document_type", "rca"),
            "timeline_entries": metadata.get("timeline_entries", 0),
            "action_items": metadata.get("action_items", 0),
            "metrics_count": metadata.get("metrics_count", 0),
            "has_timeline": metadata.get("has_timeline", False),
            "has_action_items": metadata.get("has_action_items", False),
            "has_metrics": metadata.get("has_metrics", False),
            "complexity_score": metadata.get("complexity_score", 0),
            "severity": metadata.get("severity", "unknown"),
            "incident_type": metadata.get("incident_type", "general"),
            "affected_components": metadata.get("affected_components", [])
        }
