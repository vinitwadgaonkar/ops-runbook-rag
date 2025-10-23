"""
Semantic chunking pipeline with code block preservation and section awareness.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import re
from dataclasses import dataclass
import tiktoken
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    content: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any]
    parent_section: Optional[str] = None
    code_blocks: List[Dict[str, Any]] = None
    chunk_type: str = "text"


class SemanticChunker:
    """
    Advanced semantic chunking with research-specific features.
    Preserves code blocks, maintains section awareness, and optimizes for retrieval.
    """
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Patterns for different content types
        self.code_block_pattern = re.compile(r'```(\w+)?\n(.*?)```', re.DOTALL)
        self.section_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        self.list_pattern = re.compile(r'^\s*[-*+]\s+', re.MULTILINE)
        self.numbered_list_pattern = re.compile(r'^\s*\d+\.\s+', re.MULTILINE)
        
        # Command patterns for operational content
        self.command_patterns = [
            r'^\s*\$?\s*([a-zA-Z][a-zA-Z0-9_-]*(?:\s+[^\n]+)?)$',  # Shell commands
            r'kubectl\s+[a-z]+\s+[^\n]+',  # kubectl commands
            r'docker\s+[a-z]+\s+[^\n]+',  # Docker commands
            r'curl\s+[^\n]+',  # curl commands
        ]
    
    def chunk_document(self, content: str, metadata: Dict[str, Any], 
                      source_path: Optional[str] = None) -> List[Chunk]:
        """
        Chunk a document with semantic awareness.
        
        Args:
            content: Document content to chunk
            metadata: Document metadata
            source_path: Optional source file path
            
        Returns:
            List of Chunk objects
        """
        try:
            # Extract code blocks first
            code_blocks = self._extract_code_blocks(content)
            
            # Extract sections
            sections = self._extract_sections(content)
            
            # Create chunks with semantic awareness
            chunks = self._create_semantic_chunks(
                content, code_blocks, sections, metadata, source_path
            )
            
            # Post-process chunks for quality
            chunks = self._post_process_chunks(chunks)
            
            logger.info(f"Created {len(chunks)} chunks from document")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to chunk document: {e}")
            raise
    
    def _extract_code_blocks(self, content: str) -> List[Dict[str, Any]]:
        """Extract code blocks with their positions."""
        code_blocks = []
        
        for match in self.code_block_pattern.finditer(content):
            language = match.group(1) or "text"
            code_content = match.group(2).strip()
            
            # Calculate position in document
            start_pos = match.start()
            end_pos = match.end()
            
            code_blocks.append({
                "language": language,
                "content": code_content,
                "start_pos": start_pos,
                "end_pos": end_pos,
                "line_count": code_content.count('\n') + 1,
                "char_count": len(code_content)
            })
        
        return code_blocks
    
    def _extract_sections(self, content: str) -> List[Dict[str, Any]]:
        """Extract document sections."""
        sections = []
        lines = content.split('\n')
        
        current_section = None
        current_content = []
        
        for i, line in enumerate(lines):
            header_match = self.section_pattern.match(line)
            
            if header_match:
                # Save previous section
                if current_section:
                    current_section["content"] = '\n'.join(current_content).strip()
                    current_section["end_line"] = i - 1
                    sections.append(current_section)
                
                # Start new section
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                current_section = {
                    "title": title,
                    "level": level,
                    "start_line": i,
                    "end_line": i,
                    "content": ""
                }
                current_content = []
            else:
                current_content.append(line)
        
        # Add final section
        if current_section:
            current_section["content"] = '\n'.join(current_content).strip()
            current_section["end_line"] = len(lines) - 1
            sections.append(current_section)
        
        return sections
    
    def _create_semantic_chunks(self, content: str, code_blocks: List[Dict[str, Any]], 
                               sections: List[Dict[str, Any]], metadata: Dict[str, Any],
                               source_path: Optional[str]) -> List[Chunk]:
        """Create chunks with semantic awareness."""
        chunks = []
        chunk_index = 0
        
        # If document has sections, chunk by section
        if sections:
            chunks = self._chunk_by_sections(content, sections, code_blocks, metadata, source_path)
        else:
            # Fallback to sliding window chunking
            chunks = self._sliding_window_chunk(content, code_blocks, metadata, source_path)
        
        return chunks
    
    def _chunk_by_sections(self, content: str, sections: List[Dict[str, Any]], 
                          code_blocks: List[Dict[str, Any]], metadata: Dict[str, Any],
                          source_path: Optional[str]) -> List[Chunk]:
        """Chunk document by sections with code block preservation."""
        chunks = []
        chunk_index = 0
        
        for section in sections:
            section_content = section["content"]
            section_title = section["title"]
            
            # Check if section contains code blocks
            section_code_blocks = self._get_code_blocks_in_range(
                code_blocks, section["start_line"], section["end_line"]
            )
            
            # If section is small enough, create single chunk
            if self._estimate_tokens(section_content) <= self.chunk_size:
                chunk = Chunk(
                    content=section_content,
                    chunk_index=chunk_index,
                    start_char=0,  # Will be calculated properly
                    end_char=len(section_content),
                    metadata={
                        **metadata,
                        "section_title": section_title,
                        "section_level": section["level"],
                        "source_path": source_path,
                        "chunk_type": "section"
                    },
                    parent_section=section_title,
                    code_blocks=section_code_blocks,
                    chunk_type="section"
                )
                chunks.append(chunk)
                chunk_index += 1
            else:
                # Split large section into smaller chunks
                section_chunks = self._split_large_section(
                    section_content, section_title, section_code_blocks, 
                    metadata, source_path, chunk_index
                )
                chunks.extend(section_chunks)
                chunk_index += len(section_chunks)
        
        return chunks
    
    def _split_large_section(self, content: str, section_title: str, 
                           code_blocks: List[Dict[str, Any]], metadata: Dict[str, Any],
                           source_path: Optional[str], start_chunk_index: int) -> List[Chunk]:
        """Split large section into smaller chunks while preserving code blocks."""
        chunks = []
        chunk_index = start_chunk_index
        
        # Split by paragraphs first
        paragraphs = content.split('\n\n')
        current_chunk_content = []
        current_chunk_code_blocks = []
        
        for paragraph in paragraphs:
            paragraph_tokens = self._estimate_tokens(paragraph)
            
            # If adding this paragraph would exceed chunk size, finalize current chunk
            if (self._estimate_tokens('\n\n'.join(current_chunk_content + [paragraph])) > 
                self.chunk_size and current_chunk_content):
                
                chunk = Chunk(
                    content='\n\n'.join(current_chunk_content),
                    chunk_index=chunk_index,
                    start_char=0,
                    end_char=len('\n\n'.join(current_chunk_content)),
                    metadata={
                        **metadata,
                        "section_title": section_title,
                        "source_path": source_path,
                        "chunk_type": "section_part"
                    },
                    parent_section=section_title,
                    code_blocks=current_chunk_code_blocks.copy(),
                    chunk_type="section_part"
                )
                chunks.append(chunk)
                chunk_index += 1
                
                # Start new chunk
                current_chunk_content = [paragraph]
                current_chunk_code_blocks = []
            else:
                current_chunk_content.append(paragraph)
                
                # Add code blocks in this paragraph
                paragraph_code_blocks = self._get_code_blocks_in_text(paragraph)
                current_chunk_code_blocks.extend(paragraph_code_blocks)
        
        # Add final chunk if there's content
        if current_chunk_content:
            chunk = Chunk(
                content='\n\n'.join(current_chunk_content),
                chunk_index=chunk_index,
                start_char=0,
                end_char=len('\n\n'.join(current_chunk_content)),
                metadata={
                    **metadata,
                    "section_title": section_title,
                    "source_path": source_path,
                    "chunk_type": "section_part"
                },
                parent_section=section_title,
                code_blocks=current_chunk_code_blocks,
                chunk_type="section_part"
            )
            chunks.append(chunk)
        
        return chunks
    
    def _sliding_window_chunk(self, content: str, code_blocks: List[Dict[str, Any]], 
                            metadata: Dict[str, Any], source_path: Optional[str]) -> List[Chunk]:
        """Fallback sliding window chunking."""
        chunks = []
        chunk_index = 0
        
        # Split content into sentences
        sentences = self._split_into_sentences(content)
        
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self._estimate_tokens(sentence)
            
            # If adding this sentence would exceed chunk size, finalize current chunk
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                chunk_content = ' '.join(current_chunk)
                chunk = Chunk(
                    content=chunk_content,
                    chunk_index=chunk_index,
                    start_char=0,
                    end_char=len(chunk_content),
                    metadata={
                        **metadata,
                        "source_path": source_path,
                        "chunk_type": "sliding_window"
                    },
                    chunk_type="sliding_window"
                )
                chunks.append(chunk)
                chunk_index += 1
                
                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences + [sentence]
                current_tokens = sum(self._estimate_tokens(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_content = ' '.join(current_chunk)
            chunk = Chunk(
                content=chunk_content,
                chunk_index=chunk_index,
                start_char=0,
                end_char=len(chunk_content),
                metadata={
                    **metadata,
                    "source_path": source_path,
                    "chunk_type": "sliding_window"
                },
                chunk_type="sliding_window"
            )
            chunks.append(chunk)
        
        return chunks
    
    def _get_code_blocks_in_range(self, code_blocks: List[Dict[str, Any]], 
                                 start_line: int, end_line: int) -> List[Dict[str, Any]]:
        """Get code blocks within a line range."""
        # This is a simplified implementation
        # In practice, you'd need to map character positions to line numbers
        return [cb for cb in code_blocks if cb.get("start_line", 0) >= start_line and cb.get("end_line", 0) <= end_line]
    
    def _get_code_blocks_in_text(self, text: str) -> List[Dict[str, Any]]:
        """Get code blocks within a text snippet."""
        code_blocks = []
        
        for match in self.code_block_pattern.finditer(text):
            language = match.group(1) or "text"
            code_content = match.group(2).strip()
            
            code_blocks.append({
                "language": language,
                "content": code_content,
                "line_count": code_content.count('\n') + 1,
                "char_count": len(code_content)
            })
        
        return code_blocks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting - in practice, use a proper NLP library
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Get overlap sentences for chunk continuity."""
        overlap_tokens = 0
        overlap_sentences = []
        
        # Add sentences from the end until we reach overlap size
        for sentence in reversed(sentences):
            sentence_tokens = self._estimate_tokens(sentence)
            if overlap_tokens + sentence_tokens > self.chunk_overlap:
                break
            overlap_sentences.insert(0, sentence)
            overlap_tokens += sentence_tokens
        
        return overlap_sentences
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        try:
            return len(self.tokenizer.encode(text))
        except Exception:
            # Fallback estimation
            return len(text.split()) * 1.3
    
    def _post_process_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Post-process chunks for quality and consistency."""
        processed_chunks = []
        
        for chunk in chunks:
            # Clean up content
            chunk.content = chunk.content.strip()
            
            # Skip empty chunks
            if not chunk.content:
                continue
            
            # Add chunk-specific metadata
            chunk.metadata.update({
                "token_count": self._estimate_tokens(chunk.content),
                "char_count": len(chunk.content),
                "has_code_blocks": len(chunk.code_blocks or []) > 0,
                "code_languages": list(set(cb.get("language", "text") for cb in chunk.code_blocks or [])),
                "chunk_quality": self._assess_chunk_quality(chunk)
            })
            
            processed_chunks.append(chunk)
        
        return processed_chunks
    
    def _assess_chunk_quality(self, chunk: Chunk) -> str:
        """Assess the quality of a chunk for retrieval."""
        content = chunk.content.lower()
        
        # Check for operational content indicators
        operational_indicators = [
            "troubleshoot", "debug", "error", "issue", "problem",
            "solution", "fix", "resolve", "command", "kubectl",
            "docker", "curl", "log", "monitor", "alert"
        ]
        
        indicator_count = sum(1 for indicator in operational_indicators if indicator in content)
        
        if indicator_count >= 3:
            return "high"
        elif indicator_count >= 1:
            return "medium"
        else:
            return "low"
    
    def chunk_file(self, file_path: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Chunk a file from disk."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return self.chunk_document(content, metadata, file_path)
        except Exception as e:
            logger.error(f"Failed to chunk file {file_path}: {e}")
            raise


# Utility functions for research analysis
def analyze_chunk_quality(chunks: List[Chunk]) -> Dict[str, Any]:
    """Analyze chunk quality for research purposes."""
    if not chunks:
        return {"quality_score": 0.0, "analysis": "No chunks provided"}
    
    total_chunks = len(chunks)
    high_quality_chunks = sum(1 for chunk in chunks if chunk.metadata.get("chunk_quality") == "high")
    medium_quality_chunks = sum(1 for chunk in chunks if chunk.metadata.get("chunk_quality") == "medium")
    
    avg_token_count = sum(chunk.metadata.get("token_count", 0) for chunk in chunks) / total_chunks
    chunks_with_code = sum(1 for chunk in chunks if chunk.metadata.get("has_code_blocks", False))
    
    quality_score = (high_quality_chunks * 1.0 + medium_quality_chunks * 0.5) / total_chunks
    
    return {
        "quality_score": quality_score,
        "total_chunks": total_chunks,
        "high_quality_chunks": high_quality_chunks,
        "medium_quality_chunks": medium_quality_chunks,
        "low_quality_chunks": total_chunks - high_quality_chunks - medium_quality_chunks,
        "avg_token_count": avg_token_count,
        "chunks_with_code": chunks_with_code,
        "code_coverage": chunks_with_code / total_chunks,
        "analysis": "Good chunking" if quality_score > 0.7 else "Poor chunking" if quality_score < 0.3 else "Medium chunking"
    }


def extract_chunk_features(chunks: List[Chunk]) -> Dict[str, Any]:
    """Extract features from chunks for research analysis."""
    if not chunks:
        return {}
    
    features = {
        "total_chunks": len(chunks),
        "chunk_types": list(set(chunk.chunk_type for chunk in chunks)),
        "section_titles": list(set(chunk.parent_section for chunk in chunks if chunk.parent_section)),
        "code_languages": list(set(
            lang for chunk in chunks 
            for lang in chunk.metadata.get("code_languages", [])
        )),
        "avg_token_count": sum(chunk.metadata.get("token_count", 0) for chunk in chunks) / len(chunks),
        "chunks_with_code": sum(1 for chunk in chunks if chunk.metadata.get("has_code_blocks", False)),
        "quality_distribution": {
            "high": sum(1 for chunk in chunks if chunk.metadata.get("chunk_quality") == "high"),
            "medium": sum(1 for chunk in chunks if chunk.metadata.get("chunk_quality") == "medium"),
            "low": sum(1 for chunk in chunks if chunk.metadata.get("chunk_quality") == "low")
        }
    }
    
    return features
