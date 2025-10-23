"""
Markdown parser for runbooks with frontmatter extraction and code block preservation.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import yaml
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CodeBlock:
    """Represents a code block in a markdown document."""
    language: str
    content: str
    start_line: int
    end_line: int


@dataclass
class MarkdownSection:
    """Represents a section in a markdown document."""
    title: str
    level: int
    content: str
    start_line: int
    end_line: int
    code_blocks: List[CodeBlock]


class MarkdownParser:
    """
    Parser for markdown runbooks with research-specific features.
    Handles frontmatter extraction, code block preservation, and section-aware parsing.
    """
    
    def __init__(self):
        self.frontmatter_pattern = re.compile(r'^---\s*\n(.*?)\n---\s*\n', re.DOTALL | re.MULTILINE)
        self.header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        self.code_block_pattern = re.compile(r'```(\w+)?\n(.*?)```', re.DOTALL)
        self.inline_code_pattern = re.compile(r'`([^`]+)`')
    
    def parse(self, content: str, source_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Parse markdown content and extract structured information.
        
        Args:
            content: Raw markdown content
            source_path: Optional source file path
            
        Returns:
            Dictionary with parsed content and metadata
        """
        try:
            # Extract frontmatter
            frontmatter = self._extract_frontmatter(content)
            
            # Remove frontmatter from content
            content_without_frontmatter = self._remove_frontmatter(content)
            
            # Parse sections
            sections = self._parse_sections(content_without_frontmatter)
            
            # Extract code blocks
            code_blocks = self._extract_code_blocks(content_without_frontmatter)
            
            # Extract inline code
            inline_code = self._extract_inline_code(content_without_frontmatter)
            
            # Generate metadata
            metadata = self._generate_metadata(frontmatter, sections, code_blocks, source_path)
            
            return {
                "content": content_without_frontmatter,
                "frontmatter": frontmatter,
                "sections": [self._section_to_dict(section) for section in sections],
                "code_blocks": [self._code_block_to_dict(block) for block in code_blocks],
                "inline_code": inline_code,
                "metadata": metadata,
                "source_path": source_path
            }
            
        except Exception as e:
            logger.error(f"Failed to parse markdown: {e}")
            raise
    
    def _extract_frontmatter(self, content: str) -> Dict[str, Any]:
        """Extract YAML frontmatter from markdown content."""
        match = self.frontmatter_pattern.match(content)
        if not match:
            return {}
        
        try:
            frontmatter_text = match.group(1)
            return yaml.safe_load(frontmatter_text) or {}
        except yaml.YAMLError as e:
            logger.warning(f"Failed to parse frontmatter: {e}")
            return {}
    
    def _remove_frontmatter(self, content: str) -> str:
        """Remove frontmatter from markdown content."""
        return self.frontmatter_pattern.sub('', content, count=1)
    
    def _parse_sections(self, content: str) -> List[MarkdownSection]:
        """Parse markdown content into sections based on headers."""
        sections = []
        lines = content.split('\n')
        
        current_section = None
        current_content = []
        
        for i, line in enumerate(lines):
            header_match = self.header_pattern.match(line)
            
            if header_match:
                # Save previous section
                if current_section:
                    current_section.content = '\n'.join(current_content).strip()
                    current_section.end_line = i - 1
                    sections.append(current_section)
                
                # Start new section
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                current_section = MarkdownSection(
                    title=title,
                    level=level,
                    content="",
                    start_line=i,
                    end_line=i,
                    code_blocks=[]
                )
                current_content = []
            else:
                current_content.append(line)
        
        # Add final section
        if current_section:
            current_section.content = '\n'.join(current_content).strip()
            current_section.end_line = len(lines) - 1
            sections.append(current_section)
        
        # Extract code blocks for each section
        for section in sections:
            section.code_blocks = self._extract_code_blocks_from_section(section.content)
        
        return sections
    
    def _extract_code_blocks(self, content: str) -> List[CodeBlock]:
        """Extract all code blocks from content."""
        code_blocks = []
        lines = content.split('\n')
        
        for match in self.code_block_pattern.finditer(content):
            language = match.group(1) or "text"
            code_content = match.group(2).strip()
            
            # Find line numbers
            start_line = content[:match.start()].count('\n')
            end_line = start_line + code_content.count('\n')
            
            code_blocks.append(CodeBlock(
                language=language,
                content=code_content,
                start_line=start_line,
                end_line=end_line
            ))
        
        return code_blocks
    
    def _extract_code_blocks_from_section(self, section_content: str) -> List[CodeBlock]:
        """Extract code blocks from a specific section."""
        return self._extract_code_blocks(section_content)
    
    def _extract_inline_code(self, content: str) -> List[str]:
        """Extract inline code snippets."""
        return self.inline_code_pattern.findall(content)
    
    def _generate_metadata(self, frontmatter: Dict[str, Any], sections: List[MarkdownSection], 
                         code_blocks: List[CodeBlock], source_path: Optional[str]) -> Dict[str, Any]:
        """Generate comprehensive metadata for the document."""
        metadata = {
            "document_type": "runbook",
            "source_path": source_path,
            "section_count": len(sections),
            "code_block_count": len(code_blocks),
            "total_lines": len(sections[0].content.split('\n')) if sections else 0,
            "languages": list(set(block.language for block in code_blocks)),
            "section_titles": [section.title for section in sections]
        }
        
        # Add frontmatter metadata
        metadata.update(frontmatter)
        
        # Add research-specific metadata
        metadata.update({
            "has_troubleshooting": any("troubleshoot" in section.title.lower() for section in sections),
            "has_commands": len(code_blocks) > 0,
            "has_diagrams": any("```mermaid" in section.content for section in sections),
            "complexity_score": self._calculate_complexity_score(sections, code_blocks)
        })
        
        return metadata
    
    def _calculate_complexity_score(self, sections: List[MarkdownSection], code_blocks: List[CodeBlock]) -> float:
        """Calculate a complexity score for the runbook."""
        score = 0.0
        
        # Base score from section count
        score += len(sections) * 0.1
        
        # Add score for code blocks
        score += len(code_blocks) * 0.2
        
        # Add score for nested sections
        for section in sections:
            if section.level > 2:
                score += 0.1
        
        # Add score for long content
        total_content_length = sum(len(section.content) for section in sections)
        score += min(total_content_length / 1000, 2.0)  # Cap at 2.0
        
        return min(score, 10.0)  # Cap at 10.0
    
    def _section_to_dict(self, section: MarkdownSection) -> Dict[str, Any]:
        """Convert section to dictionary."""
        return {
            "title": section.title,
            "level": section.level,
            "content": section.content,
            "start_line": section.start_line,
            "end_line": section.end_line,
            "code_blocks": [self._code_block_to_dict(block) for block in section.code_blocks]
        }
    
    def _code_block_to_dict(self, block: CodeBlock) -> Dict[str, Any]:
        """Convert code block to dictionary."""
        return {
            "language": block.language,
            "content": block.content,
            "start_line": block.start_line,
            "end_line": block.end_line,
            "line_count": block.content.count('\n') + 1
        }
    
    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """Parse a markdown file from disk."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return self.parse(content, file_path)
        except Exception as e:
            logger.error(f"Failed to parse markdown file {file_path}: {e}")
            raise
    
    def extract_action_items(self, content: str) -> List[Dict[str, Any]]:
        """Extract action items from runbook content."""
        action_items = []
        
        # Look for common action item patterns
        patterns = [
            r'^\s*[-*]\s+(.+)$',  # Bullet points
            r'^\s*\d+\.\s+(.+)$',  # Numbered lists
            r'^\s*>\s*(.+)$',  # Blockquotes
            r'^\s*TODO:\s*(.+)$',  # TODO items
            r'^\s*ACTION:\s*(.+)$',  # ACTION items
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            for match in matches:
                action_items.append({
                    "text": match.strip(),
                    "type": "action_item",
                    "pattern": pattern
                })
        
        return action_items
    
    def extract_commands(self, content: str) -> List[Dict[str, Any]]:
        """Extract shell commands from content."""
        commands = []
        
        # Extract from code blocks
        for match in self.code_block_pattern.finditer(content):
            language = match.group(1) or "text"
            code_content = match.group(2).strip()
            
            if language in ['bash', 'sh', 'shell', 'zsh', 'fish']:
                command_lines = [line.strip() for line in code_content.split('\n') if line.strip()]
                for line in command_lines:
                    if not line.startswith('#') and line:  # Skip comments
                        commands.append({
                            "command": line,
                            "language": language,
                            "type": "code_block"
                        })
        
        # Extract from inline code
        inline_commands = self.inline_code_pattern.findall(content)
        for cmd in inline_commands:
            if any(cmd.startswith(prefix) for prefix in ['kubectl', 'docker', 'git', 'curl', 'wget']):
                commands.append({
                    "command": cmd,
                    "language": "inline",
                    "type": "inline_code"
                })
        
        return commands


# Utility functions for research analysis
def analyze_runbook_structure(parsed_content: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze runbook structure for research purposes."""
    sections = parsed_content.get("sections", [])
    code_blocks = parsed_content.get("code_blocks", [])
    
    return {
        "section_depth": max(section.get("level", 0) for section in sections) if sections else 0,
        "code_coverage": len(code_blocks) / max(len(sections), 1),
        "average_section_length": sum(len(section.get("content", "")) for section in sections) / max(len(sections), 1),
        "has_troubleshooting": any("troubleshoot" in section.get("title", "").lower() for section in sections),
        "has_commands": len(code_blocks) > 0,
        "complexity_score": parsed_content.get("metadata", {}).get("complexity_score", 0)
    }


def extract_research_features(parsed_content: Dict[str, Any]) -> Dict[str, Any]:
    """Extract features useful for research analysis."""
    metadata = parsed_content.get("metadata", {})
    
    return {
        "document_type": metadata.get("document_type", "runbook"),
        "service": metadata.get("service"),
        "severity": metadata.get("severity"),
        "component": metadata.get("component"),
        "has_troubleshooting": metadata.get("has_troubleshooting", False),
        "has_commands": metadata.get("has_commands", False),
        "complexity_score": metadata.get("complexity_score", 0),
        "languages": metadata.get("languages", []),
        "section_count": metadata.get("section_count", 0),
        "code_block_count": metadata.get("code_block_count", 0)
    }
