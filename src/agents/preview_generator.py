"""
PreviewGenerator for creating interactive HTML previews of customized CV and cover letters.

This module provides comprehensive preview functionality including side-by-side comparisons,
match score visualization, and interactive approval interface for customized application materials.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import html

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.job_data import JobData
from models.match_result import MatchResult
from utils.logging_config import get_agents_logger
from utils.file_manager import create_application_folder, generate_filename

logger = get_agents_logger()


class PreviewGeneratorError(Exception):
    """Base exception for preview generator errors."""
    pass


class HTMLGenerationError(PreviewGeneratorError):
    """Exception raised when HTML generation fails."""
    pass


class ValidationError(PreviewGeneratorError):
    """Exception raised when input validation fails."""
    pass


@dataclass
class PreviewOptions:
    """Configuration options for preview generation."""
    show_side_by_side: bool = True
    include_match_analysis: bool = True
    enable_interactive_approval: bool = True
    highlight_changes: bool = True
    theme: str = "professional"  # professional, modern, minimal
    max_content_length: int = 10000


class PreviewGenerator:
    """
    Generates interactive HTML previews for customized CV and cover letters.

    This class creates comprehensive preview pages that display customized content
    alongside original content with visual comparisons, match analysis, and
    interactive approval interfaces.
    """

    def __init__(self, options: Optional[PreviewOptions] = None):
        """
        Initialize the PreviewGenerator.

        Args:
            options: Configuration options for preview generation
        """
        self.options = options or PreviewOptions()
        self.logger = logger
        self.logger.info("PreviewGenerator initialized")

    def generate_preview(
        self,
        customized_html: Dict[str, str],
        match_result: MatchResult,
        changes: List[str],
        original_html: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Main entry point that creates interactive preview HTML.

        Args:
            customized_html: Dictionary with 'cv' and 'cover_letter' HTML content
            match_result: MatchResult instance with job matching analysis
            changes: List of changes made during customization
            original_html: Optional original HTML for comparison

        Returns:
            Complete preview HTML string for browser display

        Raises:
            ValidationError: If input validation fails
            HTMLGenerationError: If HTML generation fails
        """
        self.logger.info("Generating interactive preview")

        try:
            # Validate inputs
            self._validate_inputs(customized_html, match_result, changes)

            # Extract individual components
            cv_html = customized_html.get('cv', '')
            cover_letter_html = customized_html.get('cover_letter', '')

            # Create comparison sections if original content is provided
            comparison_html = ""
            if original_html and self.options.show_side_by_side:
                comparison_html = self._create_side_by_side_comparison(
                    original_html, customized_html
                )

            # Generate the complete preview
            preview_html = self.create_preview_html(
                cv_html,
                cover_letter_html,
                match_result,
                comparison_html,
                changes
            )

            self.logger.info("Preview generated successfully")
            return preview_html

        except Exception as e:
            error_msg = f"Failed to generate preview: {e}"
            self.logger.error(error_msg)
            raise HTMLGenerationError(error_msg) from e

    def create_side_by_side_comparison(
        self,
        original: Dict[str, str],
        customized: Dict[str, str]
    ) -> str:
        """
        Creates visual comparison between original and customized content.

        Args:
            original: Dictionary with original HTML content
            customized: Dictionary with customized HTML content

        Returns:
            HTML section for side-by-side display

        Raises:
            ValidationError: If content validation fails
        """
        self.logger.debug("Creating side-by-side comparison")

        try:
            comparison_sections = []

            for doc_type in ['cv', 'cover_letter']:
                original_content = original.get(doc_type, '')
                customized_content = customized.get(doc_type, '')

                if original_content or customized_content:
                    section_html = self._create_comparison_section(
                        doc_type.replace('_', ' ').title(),
                        original_content,
                        customized_content
                    )
                    comparison_sections.append(section_html)

            return self._wrap_comparison_sections(comparison_sections)

        except Exception as e:
            error_msg = f"Failed to create side-by-side comparison: {e}"
            self.logger.error(error_msg)
            raise HTMLGenerationError(error_msg) from e

    def _create_side_by_side_comparison(
        self,
        original: Dict[str, str],
        customized: Dict[str, str]
    ) -> str:
        """Internal method for creating side-by-side comparison."""
        return self.create_side_by_side_comparison(original, customized)

    def create_preview_html(
        self,
        cv_html: str,
        cover_letter_html: str,
        match_data: MatchResult,
        comparison_html: str = "",
        changes: Optional[List[str]] = None
    ) -> str:
        """
        Combines CV and cover letter into unified preview.

        Args:
            cv_html: CV HTML content
            cover_letter_html: Cover letter HTML content
            match_data: MatchResult with job analysis
            comparison_html: Optional comparison section
            changes: List of changes made

        Returns:
            Complete HTML preview with styling and interactive elements

        Raises:
            HTMLGenerationError: If HTML generation fails
        """
        self.logger.debug("Creating unified preview HTML")

        try:
            changes = changes or []

            # Sanitize HTML content
            cv_html = self._sanitize_html_content(cv_html)
            cover_letter_html = self._sanitize_html_content(cover_letter_html)

            # Generate CSS styles
            css_styles = self._generate_css_styles()

            # Generate match score visualization
            match_score_html = self._create_match_score_visualization(match_data)

            # Generate changes summary
            changes_html = self._create_changes_summary(changes)

            # Generate approval interface
            approval_html = self._create_approval_interface() if self.options.enable_interactive_approval else ""

            # Create the complete HTML structure
            html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Application Preview - {match_data.job_title} at {match_data.company_name}</title>
    <style>{css_styles}</style>
</head>
<body>
    <div class="preview-container">
        <header class="preview-header">
            <h1>Application Preview</h1>
            <div class="job-info">
                <h2>{html.escape(match_data.job_title)}</h2>
                <h3>{html.escape(match_data.company_name)}</h3>
                <p class="preview-date">Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
            </div>
        </header>

        {match_score_html}

        {changes_html}

        {comparison_html}

        <main class="documents-section">
            <div class="document-grid">
                <section class="document cv-section">
                    <h2>Curriculum Vitae</h2>
                    <div class="document-content">
                        {cv_html}
                    </div>
                </section>

                <section class="document cover-letter-section">
                    <h2>Cover Letter</h2>
                    <div class="document-content">
                        {cover_letter_html}
                    </div>
                </section>
            </div>
        </main>

        {approval_html}

        <footer class="preview-footer">
            <p>Generated by IntelligentApply CV Generator</p>
            <p>Match analysis confidence: {match_data.confidence_level:.1%}</p>
        </footer>
    </div>

    <script>{self._generate_javascript()}</script>
</body>
</html>
"""

            self.logger.debug("Preview HTML created successfully")
            return html_content

        except Exception as e:
            error_msg = f"Failed to create preview HTML: {e}"
            self.logger.error(error_msg)
            raise HTMLGenerationError(error_msg) from e

    def get_user_approval(self) -> bool:
        """
        Handles user interaction for approval/rejection.

        Note: This is a placeholder for future implementation.
        In a web interface, this would be handled by JavaScript callbacks.
        In a CLI interface, this could prompt the user for input.

        Returns:
            Boolean indicating user decision (currently returns True as placeholder)
        """
        self.logger.info("User approval requested")
        # This would be implemented based on the interface (web/CLI)
        # For now, return True as placeholder
        return True

    def _validate_inputs(
        self,
        customized_html: Dict[str, str],
        match_result: MatchResult,
        changes: List[str]
    ) -> None:
        """Validate input parameters."""
        if not isinstance(customized_html, dict):
            raise ValidationError("customized_html must be a dictionary")

        if not isinstance(match_result, MatchResult):
            raise ValidationError("match_result must be a MatchResult instance")

        if not isinstance(changes, list):
            raise ValidationError("changes must be a list")

        # Check required keys
        required_keys = ['cv', 'cover_letter']
        for key in required_keys:
            if key not in customized_html:
                raise ValidationError(f"Missing required key: {key}")

        # Validate content length
        for key, content in customized_html.items():
            if len(content) > self.options.max_content_length:
                raise ValidationError(f"Content too long for {key}: {len(content)} > {self.options.max_content_length}")

    def _sanitize_html_content(self, content: str) -> str:
        """Sanitize HTML content for safe display."""
        if not content:
            return "<p>No content available</p>"

        # Basic HTML sanitization - in production, use a proper HTML sanitizer
        # This is a simple implementation for demonstration
        sanitized = html.escape(content)

        # Allow basic HTML tags back
        allowed_tags = {
            '&lt;p&gt;': '<p>',
            '&lt;/p&gt;': '</p>',
            '&lt;br&gt;': '<br>',
            '&lt;strong&gt;': '<strong>',
            '&lt;/strong&gt;': '</strong>',
            '&lt;em&gt;': '<em>',
            '&lt;/em&gt;': '</em>',
            '&lt;ul&gt;': '<ul>',
            '&lt;/ul&gt;': '</ul>',
            '&lt;li&gt;': '<li>',
            '&lt;/li&gt;': '</li>',
            '&lt;h1&gt;': '<h1>',
            '&lt;/h1&gt;': '</h1>',
            '&lt;h2&gt;': '<h2>',
            '&lt;/h2&gt;': '</h2>',
            '&lt;h3&gt;': '<h3>',
            '&lt;/h3&gt;': '</h3>',
        }

        for escaped, allowed in allowed_tags.items():
            sanitized = sanitized.replace(escaped, allowed)

        return sanitized

    def _create_comparison_section(
        self,
        title: str,
        original: str,
        customized: str
    ) -> str:
        """Create a comparison section for a document type."""
        return f"""
        <div class="comparison-section">
            <h3>{html.escape(title)} Comparison</h3>
            <div class="comparison-grid">
                <div class="original-content">
                    <h4>Original</h4>
                    <div class="content-box">
                        {self._sanitize_html_content(original)}
                    </div>
                </div>
                <div class="customized-content">
                    <h4>Customized</h4>
                    <div class="content-box highlighted">
                        {self._sanitize_html_content(customized)}
                    </div>
                </div>
            </div>
        </div>
        """

    def _wrap_comparison_sections(self, sections: List[str]) -> str:
        """Wrap comparison sections in a container."""
        if not sections:
            return ""

        return f"""
        <section class="comparisons-section">
            <h2>Content Comparison</h2>
            <div class="comparisons-container">
                {''.join(sections)}
            </div>
        </section>
        """

    def _create_match_score_visualization(self, match_data: MatchResult) -> str:
        """Create HTML for match score visualization."""
        score_percentage = (match_data.score / 10) * 100
        category = match_data.get_match_category()

        # Determine color based on score
        if match_data.score >= 8:
            score_color = "#28a745"  # Green
        elif match_data.score >= 6:
            score_color = "#ffc107"  # Yellow
        elif match_data.score >= 4:
            score_color = "#fd7e14"  # Orange
        else:
            score_color = "#dc3545"  # Red

        return f"""
        <section class="match-analysis">
            <h2>Match Analysis</h2>
            <div class="match-score-container">
                <div class="score-circle">
                    <div class="score-progress" style="--score: {score_percentage}%; --color: {score_color};">
                        <span class="score-number">{match_data.score}/10</span>
                    </div>
                </div>
                <div class="score-details">
                    <h3>{category}</h3>
                    <p class="match-summary">{match_data.get_summary()}</p>
                    <div class="skills-breakdown">
                        <div class="stat">
                            <label>Skills Match:</label>
                            <span>{match_data.analysis_details.skills_match_percentage:.1f}%</span>
                        </div>
                        <div class="stat">
                            <label>Experience Match:</label>
                            <span>{match_data.analysis_details.experience_match_percentage:.1f}%</span>
                        </div>
                        <div class="stat">
                            <label>Matched Skills:</label>
                            <span>{len(match_data.matched_skills)}</span>
                        </div>
                    </div>
                </div>
            </div>
        </section>
        """

    def _create_changes_summary(self, changes: List[str]) -> str:
        """Create HTML for changes summary."""
        if not changes:
            return ""

        changes_list = ''.join([f"<li>{html.escape(change)}</li>" for change in changes])

        return f"""
        <section class="changes-summary">
            <h2>Customizations Applied</h2>
            <div class="changes-container">
                <p class="changes-intro">The following customizations were made to optimize your application:</p>
                <ul class="changes-list">
                    {changes_list}
                </ul>
                <p class="changes-count">Total customizations: {len(changes)}</p>
            </div>
        </section>
        """

    def _create_approval_interface(self) -> str:
        """Create HTML for interactive approval interface."""
        return """
        <section class="approval-interface">
            <h2>Review & Approve</h2>
            <div class="approval-container">
                <p class="approval-text">Please review your customized application materials above.</p>
                <div class="approval-buttons">
                    <button class="btn btn-approve" onclick="approveApplication()">
                        <span class="btn-icon">✓</span>
                        Approve & Generate PDFs
                    </button>
                    <button class="btn btn-reject" onclick="rejectApplication()">
                        <span class="btn-icon">✗</span>
                        Reject & Make Changes
                    </button>
                    <button class="btn btn-download" onclick="downloadPreview()">
                        <span class="btn-icon">↓</span>
                        Download Preview
                    </button>
                </div>
                <div class="approval-status" id="approval-status" style="display: none;">
                    <p id="status-message"></p>
                </div>
            </div>
        </section>
        """

    def _generate_css_styles(self) -> str:
        """Generate CSS styles for the preview."""
        base_styles = """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f8f9fa;
        }

        .preview-container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: white;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
            min-height: 100vh;
        }

        .preview-header {
            border-bottom: 3px solid #007bff;
            padding-bottom: 20px;
            margin-bottom: 30px;
            text-align: center;
        }

        .preview-header h1 {
            color: #007bff;
            font-size: 2.5rem;
            margin-bottom: 10px;
        }

        .job-info h2 {
            color: #333;
            font-size: 1.8rem;
            margin-bottom: 5px;
        }

        .job-info h3 {
            color: #666;
            font-size: 1.4rem;
            margin-bottom: 10px;
        }

        .preview-date {
            color: #888;
            font-style: italic;
        }

        .match-analysis {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 30px;
            border-left: 5px solid #007bff;
        }

        .match-score-container {
            display: flex;
            align-items: center;
            gap: 30px;
            flex-wrap: wrap;
        }

        .score-circle {
            width: 120px;
            height: 120px;
            position: relative;
        }

        .score-progress {
            width: 100%;
            height: 100%;
            border-radius: 50%;
            background: conic-gradient(var(--color) 0deg, var(--color) calc(var(--score) * 3.6deg), #e9ecef calc(var(--score) * 3.6deg), #e9ecef 360deg);
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
        }

        .score-progress::before {
            content: '';
            width: 80%;
            height: 80%;
            background: white;
            border-radius: 50%;
            position: absolute;
        }

        .score-number {
            font-size: 1.5rem;
            font-weight: bold;
            color: var(--color);
            z-index: 1;
            position: relative;
        }

        .score-details {
            flex: 1;
            min-width: 300px;
        }

        .score-details h3 {
            font-size: 1.5rem;
            margin-bottom: 10px;
            color: #007bff;
        }

        .match-summary {
            margin-bottom: 15px;
            color: #666;
        }

        .skills-breakdown {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }

        .stat {
            display: flex;
            justify-content: space-between;
            padding: 10px;
            background: white;
            border-radius: 5px;
            border: 1px solid #dee2e6;
        }

        .stat label {
            font-weight: 600;
            color: #495057;
        }

        .stat span {
            font-weight: bold;
            color: #007bff;
        }

        .changes-summary {
            background: #e8f4fd;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 30px;
            border-left: 5px solid #17a2b8;
        }

        .changes-list {
            margin: 15px 0;
            padding-left: 20px;
        }

        .changes-list li {
            margin-bottom: 8px;
            color: #495057;
        }

        .changes-count {
            font-weight: bold;
            color: #17a2b8;
            margin-top: 15px;
        }

        .comparisons-section {
            margin-bottom: 30px;
        }

        .comparison-section {
            margin-bottom: 25px;
            border: 1px solid #dee2e6;
            border-radius: 10px;
            overflow: hidden;
        }

        .comparison-section h3 {
            background: #007bff;
            color: white;
            padding: 15px;
            margin: 0;
        }

        .comparison-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0;
        }

        .original-content, .customized-content {
            padding: 20px;
        }

        .original-content {
            background: #f8f9fa;
            border-right: 1px solid #dee2e6;
        }

        .customized-content {
            background: #e8f5e8;
        }

        .content-box {
            background: white;
            padding: 15px;
            border-radius: 5px;
            border: 1px solid #dee2e6;
            max-height: 300px;
            overflow-y: auto;
        }

        .highlighted {
            border-color: #28a745;
            box-shadow: 0 0 5px rgba(40, 167, 69, 0.3);
        }

        .documents-section {
            margin-bottom: 30px;
        }

        .document-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }

        .document {
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 10px;
            overflow: hidden;
        }

        .document h2 {
            background: #495057;
            color: white;
            padding: 15px;
            margin: 0;
        }

        .document-content {
            padding: 20px;
            max-height: 600px;
            overflow-y: auto;
        }

        .approval-interface {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 30px;
            text-align: center;
        }

        .approval-buttons {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin: 20px 0;
            flex-wrap: wrap;
        }

        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 5px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 8px;
            min-width: 180px;
            justify-content: center;
        }

        .btn-approve {
            background: #28a745;
            color: white;
        }

        .btn-approve:hover {
            background: #218838;
            transform: translateY(-2px);
        }

        .btn-reject {
            background: #dc3545;
            color: white;
        }

        .btn-reject:hover {
            background: #c82333;
            transform: translateY(-2px);
        }

        .btn-download {
            background: #007bff;
            color: white;
        }

        .btn-download:hover {
            background: #0056b3;
            transform: translateY(-2px);
        }

        .approval-status {
            margin-top: 15px;
            padding: 10px;
            border-radius: 5px;
            font-weight: bold;
        }

        .preview-footer {
            border-top: 1px solid #dee2e6;
            padding-top: 20px;
            text-align: center;
            color: #6c757d;
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            .document-grid, .comparison-grid {
                grid-template-columns: 1fr;
            }

            .match-score-container {
                flex-direction: column;
                text-align: center;
            }

            .approval-buttons {
                flex-direction: column;
                align-items: center;
            }

            .btn {
                width: 100%;
                max-width: 300px;
            }
        }

        /* Print Styles */
        @media print {
            .approval-interface, .preview-footer {
                display: none;
            }

            .preview-container {
                box-shadow: none;
                max-width: none;
            }
        }
        """

        return base_styles

    def _generate_javascript(self) -> str:
        """Generate JavaScript for interactive functionality."""
        return """
        function approveApplication() {
            showStatus('Application approved! Generating PDFs...', 'success');
            // In a real implementation, this would trigger PDF generation
            setTimeout(() => {
                showStatus('PDFs generated successfully!', 'success');
            }, 2000);
        }

        function rejectApplication() {
            showStatus('Application rejected. You can make changes and regenerate.', 'info');
        }

        function downloadPreview() {
            showStatus('Downloading preview...', 'info');
            // In a real implementation, this would trigger download
            window.print();
        }

        function showStatus(message, type) {
            const statusDiv = document.getElementById('approval-status');
            const messageP = document.getElementById('status-message');

            messageP.textContent = message;
            statusDiv.style.display = 'block';

            // Style based on type
            statusDiv.className = 'approval-status';
            if (type === 'success') {
                statusDiv.style.backgroundColor = '#d4edda';
                statusDiv.style.color = '#155724';
                statusDiv.style.border = '1px solid #c3e6cb';
            } else if (type === 'error') {
                statusDiv.style.backgroundColor = '#f8d7da';
                statusDiv.style.color = '#721c24';
                statusDiv.style.border = '1px solid #f5c6cb';
            } else {
                statusDiv.style.backgroundColor = '#cce7ff';
                statusDiv.style.color = '#004085';
                statusDiv.style.border = '1px solid #b0d4ff';
            }
        }

        // Auto-hide status messages after 5 seconds
        document.addEventListener('DOMContentLoaded', function() {
            const statusDiv = document.getElementById('approval-status');
            if (statusDiv) {
                const observer = new MutationObserver(function(mutations) {
                    mutations.forEach(function(mutation) {
                        if (mutation.type === 'attributes' && mutation.attributeName === 'style') {
                            if (statusDiv.style.display !== 'none') {
                                setTimeout(() => {
                                    statusDiv.style.display = 'none';
                                }, 5000);
                            }
                        }
                    });
                });
                observer.observe(statusDiv, { attributes: true });
            }
        });
        """