"""
JSON KB article parser for structured knowledge base content.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)


@dataclass
class KBArticle:
    """Represents a knowledge base article."""
    title: str
    content: str
    category: str
    tags: List[str]
    metadata: Dict[str, Any]
    source_path: Optional[str] = None


@dataclass
class KBStep:
    """Represents a step in a KB article."""
    step_number: int
    title: str
    description: str
    commands: List[str]
    expected_output: Optional[str] = None
    troubleshooting: Optional[str] = None


class JSONKBParser:
    """
    Parser for JSON knowledge base articles with research-specific features.
    Handles structured KB content, step-by-step procedures, and metadata extraction.
    """
    
    def __init__(self):
        self.required_fields = ["title", "content", "category"]
        self.optional_fields = ["tags", "metadata", "steps", "troubleshooting", "related_articles"]
        self.step_patterns = [
            r"step\s*(\d+)",
            r"(\d+)\.\s*",
            r"action\s*(\d+)",
            r"procedure\s*(\d+)"
        ]
    
    def parse(self, content: str, source_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Parse JSON KB content and extract structured information.
        
        Args:
            content: Raw JSON content
            source_path: Optional source file path
            
        Returns:
            Dictionary with parsed content and metadata
        """
        try:
            # Parse JSON content
            kb_data = json.loads(content)
            
            # Validate required fields
            self._validate_kb_structure(kb_data)
            
            # Extract main content
            title = kb_data.get("title", "")
            content_text = kb_data.get("content", "")
            category = kb_data.get("category", "general")
            
            # Extract steps if present
            steps = self._extract_steps(kb_data)
            
            # Extract troubleshooting information
            troubleshooting = self._extract_troubleshooting(kb_data)
            
            # Extract tags and metadata
            tags = kb_data.get("tags", [])
            metadata = kb_data.get("metadata", {})
            
            # Extract commands and code snippets
            commands = self._extract_commands(content_text)
            code_snippets = self._extract_code_snippets(content_text)
            
            # Generate research metadata
            research_metadata = self._generate_research_metadata(
                kb_data, steps, commands, code_snippets, source_path
            )
            
            return {
                "title": title,
                "content": content_text,
                "category": category,
                "tags": tags,
                "steps": [self._step_to_dict(step) for step in steps],
                "troubleshooting": troubleshooting,
                "commands": commands,
                "code_snippets": code_snippets,
                "metadata": research_metadata,
                "source_path": source_path,
                "raw_data": kb_data
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON content: {e}")
            raise ValueError(f"Invalid JSON format: {e}")
        except Exception as e:
            logger.error(f"Failed to parse KB article: {e}")
            raise
    
    def _validate_kb_structure(self, kb_data: Dict[str, Any]) -> None:
        """Validate that KB data has required fields."""
        missing_fields = [field for field in self.required_fields if field not in kb_data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
    
    def _extract_steps(self, kb_data: Dict[str, Any]) -> List[KBStep]:
        """Extract step-by-step procedures from KB data."""
        steps = []
        
        # Check for explicit steps array
        if "steps" in kb_data and isinstance(kb_data["steps"], list):
            for i, step_data in enumerate(kb_data["steps"]):
                step = KBStep(
                    step_number=i + 1,
                    title=step_data.get("title", f"Step {i + 1}"),
                    description=step_data.get("description", ""),
                    commands=step_data.get("commands", []),
                    expected_output=step_data.get("expected_output"),
                    troubleshooting=step_data.get("troubleshooting")
                )
                steps.append(step)
        
        # Extract steps from content using pattern matching
        content = kb_data.get("content", "")
        content_steps = self._extract_steps_from_content(content)
        steps.extend(content_steps)
        
        return steps
    
    def _extract_steps_from_content(self, content: str) -> List[KBStep]:
        """Extract steps from content using pattern matching."""
        steps = []
        lines = content.split('\n')
        
        current_step = None
        step_content = []
        
        for i, line in enumerate(lines):
            # Check if line starts a new step
            step_match = None
            for pattern in self.step_patterns:
                match = re.match(pattern, line.strip(), re.IGNORECASE)
                if match:
                    step_match = match
                    break
            
            if step_match:
                # Save previous step
                if current_step:
                    current_step.description = '\n'.join(step_content).strip()
                    steps.append(current_step)
                
                # Start new step
                step_number = int(step_match.group(1))
                title = line.strip()
                current_step = KBStep(
                    step_number=step_number,
                    title=title,
                    description="",
                    commands=[]
                )
                step_content = []
            else:
                if current_step:
                    step_content.append(line)
        
        # Add final step
        if current_step:
            current_step.description = '\n'.join(step_content).strip()
            steps.append(current_step)
        
        return steps
    
    def _extract_troubleshooting(self, kb_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract troubleshooting information from KB data."""
        troubleshooting = {}
        
        # Check for explicit troubleshooting section
        if "troubleshooting" in kb_data:
            troubleshooting_data = kb_data["troubleshooting"]
            if isinstance(troubleshooting_data, dict):
                troubleshooting = troubleshooting_data
            elif isinstance(troubleshooting_data, str):
                troubleshooting["content"] = troubleshooting_data
        
        # Extract troubleshooting from content
        content = kb_data.get("content", "")
        troubleshooting_patterns = [
            r"troubleshooting[:\s]*(.*?)(?=\n\n|\n[A-Z]|\Z)",
            r"common\s+issues[:\s]*(.*?)(?=\n\n|\n[A-Z]|\Z)",
            r"troubleshoot[:\s]*(.*?)(?=\n\n|\n[A-Z]|\Z)"
        ]
        
        for pattern in troubleshooting_patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                troubleshooting["extracted_content"] = match.group(1).strip()
                break
        
        return troubleshooting
    
    def _extract_commands(self, content: str) -> List[Dict[str, Any]]:
        """Extract shell commands from content."""
        commands = []
        
        # Common command patterns
        command_patterns = [
            r'`([^`]+)`',  # Inline code
            r'```(?:bash|sh|shell|zsh|fish)?\n(.*?)```',  # Code blocks
            r'^\s*\$?\s*([a-zA-Z][a-zA-Z0-9_-]*(?:\s+[^\n]+)?)$'  # Command lines
        ]
        
        for pattern in command_patterns:
            matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
            for match in matches:
                if isinstance(match, tuple):
                    cmd = match[0] if match[0] else match[1]
                else:
                    cmd = match
                
                cmd = cmd.strip()
                if cmd and not cmd.startswith('#'):  # Skip comments
                    commands.append({
                        "command": cmd,
                        "type": "extracted",
                        "pattern": pattern
                    })
        
        return commands
    
    def _extract_code_snippets(self, content: str) -> List[Dict[str, Any]]:
        """Extract code snippets from content."""
        code_snippets = []
        
        # Extract code blocks
        code_block_pattern = r'```(\w+)?\n(.*?)```'
        matches = re.findall(code_block_pattern, content, re.DOTALL)
        
        for language, code in matches:
            code_snippets.append({
                "language": language or "text",
                "content": code.strip(),
                "type": "code_block",
                "line_count": code.count('\n') + 1
            })
        
        # Extract inline code
        inline_pattern = r'`([^`]+)`'
        inline_matches = re.findall(inline_pattern, content)
        
        for code in inline_matches:
            if len(code) > 10:  # Only longer inline code
                code_snippets.append({
                    "language": "inline",
                    "content": code,
                    "type": "inline",
                    "line_count": 1
                })
        
        return code_snippets
    
    def _generate_research_metadata(self, kb_data: Dict[str, Any], steps: List[KBStep], 
                                   commands: List[Dict[str, Any]], code_snippets: List[Dict[str, Any]], 
                                   source_path: Optional[str]) -> Dict[str, Any]:
        """Generate research-specific metadata."""
        metadata = {
            "document_type": "kb_article",
            "source_path": source_path,
            "category": kb_data.get("category", "general"),
            "step_count": len(steps),
            "command_count": len(commands),
            "code_snippet_count": len(code_snippets),
            "has_troubleshooting": bool(kb_data.get("troubleshooting")),
            "complexity_score": self._calculate_complexity_score(steps, commands, code_snippets),
            "languages": list(set(snippet.get("language", "text") for snippet in code_snippets)),
            "tags": kb_data.get("tags", [])
        }
        
        # Add original metadata
        metadata.update(kb_data.get("metadata", {}))
        
        # Add research-specific features
        metadata.update({
            "has_step_by_step": len(steps) > 0,
            "has_commands": len(commands) > 0,
            "has_code_examples": len(code_snippets) > 0,
            "average_step_length": sum(len(step.description) for step in steps) / max(len(steps), 1),
            "command_diversity": len(set(cmd.get("command", "").split()[0] for cmd in commands if cmd.get("command"))),
            "troubleshooting_quality": self._assess_troubleshooting_quality(kb_data.get("troubleshooting", {}))
        })
        
        return metadata
    
    def _calculate_complexity_score(self, steps: List[KBStep], commands: List[Dict[str, Any]], 
                                  code_snippets: List[Dict[str, Any]]) -> float:
        """Calculate complexity score for KB article."""
        score = 0.0
        
        # Base score from step count
        score += len(steps) * 0.2
        
        # Add score for commands
        score += len(commands) * 0.1
        
        # Add score for code snippets
        score += len(code_snippets) * 0.15
        
        # Add score for step complexity
        for step in steps:
            if step.commands:
                score += 0.1
            if step.troubleshooting:
                score += 0.2
        
        return min(score, 10.0)  # Cap at 10.0
    
    def _assess_troubleshooting_quality(self, troubleshooting: Dict[str, Any]) -> float:
        """Assess the quality of troubleshooting information."""
        if not troubleshooting:
            return 0.0
        
        score = 0.0
        
        # Check for common troubleshooting elements
        content = str(troubleshooting).lower()
        
        if "error" in content:
            score += 0.2
        if "solution" in content or "fix" in content:
            score += 0.3
        if "prevention" in content or "avoid" in content:
            score += 0.2
        if "escalate" in content or "contact" in content:
            score += 0.1
        if "log" in content or "debug" in content:
            score += 0.2
        
        return min(score, 1.0)
    
    def _step_to_dict(self, step: KBStep) -> Dict[str, Any]:
        """Convert step to dictionary."""
        return {
            "step_number": step.step_number,
            "title": step.title,
            "description": step.description,
            "commands": step.commands,
            "expected_output": step.expected_output,
            "troubleshooting": step.troubleshooting
        }
    
    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """Parse a JSON KB file from disk."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return self.parse(content, file_path)
        except Exception as e:
            logger.error(f"Failed to parse JSON KB file {file_path}: {e}")
            raise
    
    def extract_research_features(self, parsed_content: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features useful for research analysis."""
        metadata = parsed_content.get("metadata", {})
        
        return {
            "document_type": metadata.get("document_type", "kb_article"),
            "category": metadata.get("category", "general"),
            "step_count": metadata.get("step_count", 0),
            "command_count": metadata.get("command_count", 0),
            "has_troubleshooting": metadata.get("has_troubleshooting", False),
            "complexity_score": metadata.get("complexity_score", 0),
            "languages": metadata.get("languages", []),
            "tags": metadata.get("tags", []),
            "has_step_by_step": metadata.get("has_step_by_step", False),
            "has_commands": metadata.get("has_commands", False),
            "troubleshooting_quality": metadata.get("troubleshooting_quality", 0)
        }
