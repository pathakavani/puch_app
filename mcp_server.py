from typing import Annotated
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
import markdownify
from mcp import ErrorData, McpError
from mcp.server.auth.provider import AccessToken
from mcp.types import INTERNAL_ERROR, INVALID_PARAMS, TextContent
from pydantic import BaseModel, AnyUrl, Field
import readabilipy
from pathlib import Path
import asyncio
import os
import docx2txt
import PyPDF2
import io

TOKEN = "7c533121f40d"
MY_NUMBER = "17623168039"


class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None


class SimpleBearerAuthProvider(BearerAuthProvider):
    """
    A simple BearerAuthProvider that does not require any specific configuration.
    It allows any valid bearer token to access the MCP server.
    For a more complete implementation that can authenticate dynamically generated tokens,
    please use `BearerAuthProvider` with your public key or JWKS URI.
    """

    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(
            public_key=k.public_key, jwks_uri=None, issuer=None, audience=None
        )
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(
                token=token,
                client_id="unknown",
                scopes=[],
                expires_at=None,  # No expiration for simplicity
            )
        return None


class ResumeProcessor:
    """Handle different resume formats and convert them to markdown"""
    
    @staticmethod
    def read_pdf(file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            raise Exception(f"Failed to read PDF: {str(e)}")
    
    @staticmethod
    def read_docx(file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            text = docx2txt.process(file_path)
            return text.strip() if text else ""
        except Exception as e:
            raise Exception(f"Failed to read DOCX: {str(e)}")
    
    @staticmethod
    def read_txt(file_path: str) -> str:
        """Read plain text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    return file.read().strip()
            except Exception as e:
                raise Exception(f"Failed to read TXT file: {str(e)}")
        except Exception as e:
            raise Exception(f"Failed to read TXT file: {str(e)}")
    
    @staticmethod
    def text_to_markdown(text: str) -> str:
        """Convert plain text to basic markdown format"""
        if not text:
            return "# Resume\n\nNo content found in resume file."
            
        lines = text.strip().split('\n')
        markdown_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                markdown_lines.append("")
                continue
            
            # Simple heuristics to format as markdown
            # Check for section headers (all caps, short lines)
            if line.isupper() and len(line) < 50 and len(line) > 2:
                markdown_lines.append(f"## {line.title()}")
            # Check for lines ending with colon (likely subsections)
            elif line.endswith(':') and len(line) < 50:
                markdown_lines.append(f"### {line}")
            # Check for bullet points
            elif line.startswith(('•', '-', '*', '◦', '▪')):
                markdown_lines.append(f"- {line[1:].strip()}")
            # Check for numbered lists
            elif len(line) > 3 and line[0].isdigit() and line[1:3] in ['. ', ') ']:
                markdown_lines.append(f"{line}")
            # Check for email addresses (make them bold)
            elif '@' in line and '.' in line and len(line.split()) == 1:
                markdown_lines.append(f"**{line}**")
            # Check for phone numbers (simple heuristic)
            elif any(char.isdigit() for char in line) and any(char in line for char in ['-', '(', ')', ' ', '.']):
                if len([c for c in line if c.isdigit()]) >= 7:  # At least 7 digits
                    markdown_lines.append(f"**{line}**")
                else:
                    markdown_lines.append(line)
            else:
                markdown_lines.append(line)
        
        return '\n'.join(markdown_lines)


class Fetch:
    IGNORE_ROBOTS_TXT = True
    USER_AGENT = "Puch/1.0 (Autonomous)"

    @classmethod
    async def fetch_url(
        cls,
        url: str,
        user_agent: str,
        force_raw: bool = False,
    ) -> tuple[str, str]:
        """
        Fetch the URL and return the content in a form ready for the LLM, as well as a prefix string with status information.
        """
        from httpx import AsyncClient, HTTPError

        async with AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    follow_redirects=True,
                    headers={"User-Agent": user_agent},
                    timeout=30,
                )
            except HTTPError as e:
                raise McpError(
                    ErrorData(
                        code=INTERNAL_ERROR, message=f"Failed to fetch {url}: {e!r}"
                    )
                )
            if response.status_code >= 400:
                raise McpError(
                    ErrorData(
                        code=INTERNAL_ERROR,
                        message=f"Failed to fetch {url} - status code {response.status_code}",
                    )
                )

            page_raw = response.text

        content_type = response.headers.get("content-type", "")
        is_page_html = (
            "<html" in page_raw[:100] or "text/html" in content_type or not content_type
        )

        if is_page_html and not force_raw:
            return cls.extract_content_from_html(page_raw), ""

        return (
            page_raw,
            f"Content type {content_type} cannot be simplified to markdown, but here is the raw content:\n",
        )

    @staticmethod
    def extract_content_from_html(html: str) -> str:
        """Extract and convert HTML content to Markdown format.

        Args:
            html: Raw HTML content to process

        Returns:
            Simplified markdown version of the content
        """
        ret = readabilipy.simple_json.simple_json_from_html_string(
            html, use_readability=True
        )
        if not ret["content"]:
            return "<error>Page failed to be simplified from HTML</error>"
        content = markdownify.markdownify(
            ret["content"],
            heading_style=markdownify.ATX,
        )
        return content


mcp = FastMCP(
    "My MCP Server",
    auth=SimpleBearerAuthProvider(TOKEN),
)

ResumeToolDescription = RichToolDescription(
    description="Serve your resume in plain markdown.",
    use_when="Puch (or anyone) asks for your resume; this must return raw markdown, \
no extra formatting.",
    side_effects=None,
)

@mcp.tool(description=ResumeToolDescription.model_dump_json())
async def resume() -> str:
    """
    Return your resume exactly as markdown text.
    This function searches for resume files and converts them to markdown format.
    """
    # Define possible resume file locations and names
    resume_paths = [
        "resume.pdf",
        "resume.docx", 
        "resume.txt",
        "resume.md",
        "CV.pdf",
        "CV.docx",
        "CV.txt",
        "CV.md",
        "./resume/resume.pdf",
        "./resume/resume.docx",
        "./resume/resume.txt",
        "./resume/resume.md",
        "./documents/resume.pdf",
        "./documents/resume.docx",
        "./documents/resume.txt",
        "./documents/resume.md",
        # Also check current directory for any files containing "resume" or "cv"
    ]
    
    # Also search for files in current directory that might be resumes
    try:
        current_dir = Path(".")
        for file_path in current_dir.glob("*"):
            if file_path.is_file():
                filename_lower = file_path.name.lower()
                if any(keyword in filename_lower for keyword in ["resume", "cv"]) and \
                   file_path.suffix.lower() in [".pdf", ".docx", ".txt", ".md"]:
                    resume_paths.append(str(file_path))
    except Exception:
        pass  # Continue with predefined paths if directory search fails
    
    processor = ResumeProcessor()
    
    # Try to find and read resume file
    for path in resume_paths:
        if os.path.exists(path):
            file_path = Path(path)
            extension = file_path.suffix.lower()
            
            try:
                if extension == '.pdf':
                    text = processor.read_pdf(path)
                elif extension == '.docx':
                    text = processor.read_docx(path)
                elif extension == '.txt':
                    text = processor.read_txt(path)
                elif extension == '.md':
                    # Already markdown, just read and return
                    text = processor.read_txt(path)
                    return text if text else "# Resume\n\nEmpty resume file found."
                else:
                    continue
                
                if text:
                    # Convert to markdown
                    markdown_resume = processor.text_to_markdown(text)
                    return markdown_resume
                    
            except Exception as e:
                # Log error but continue trying other files
                print(f"Error reading {path}: {e}")
                continue
    
    # If no resume file found, return helpful message
    return """# Resume Not Found

Please place your resume file in one of these locations:
- resume.pdf
- resume.docx  
- resume.txt
- resume.md
- CV.pdf
- CV.docx
- CV.txt
- CV.md

Or create a folder called 'resume' or 'documents' and place your resume there.

**Supported formats:** PDF, DOCX, TXT, MD

**Current directory searched:** Files containing 'resume' or 'cv' in the name will be automatically detected."""


@mcp.tool
async def validate() -> str:
    """
    NOTE: This tool must be present in an MCP server used by puch.
    """
    return MY_NUMBER


FetchToolDescription = RichToolDescription(
    description="Fetch a URL and return its content.",
    use_when="Use this tool when the user provides a URL and asks for its content, or when the user wants to fetch a webpage.",
    side_effects="The user will receive the content of the requested URL in a simplified format, or raw HTML if requested.",
)


@mcp.tool(description=FetchToolDescription.model_dump_json())
async def fetch(
    url: Annotated[AnyUrl, Field(description="URL to fetch")],
    max_length: Annotated[
        int,
        Field(
            default=5000,
            description="Maximum number of characters to return.",
            gt=0,
            lt=1000000,
        ),
    ] = 5000,
    start_index: Annotated[
        int,
        Field(
            default=0,
            description="On return output starting at this character index, useful if a previous fetch was truncated and more context is required.",
            ge=0,
        ),
    ] = 0,
    raw: Annotated[
        bool,
        Field(
            default=False,
            description="Get the actual HTML content if the requested page, without simplification.",
        ),
    ] = False,
) -> list[TextContent]:
    """Fetch a URL and return its content."""
    url_str = str(url).strip()
    if not url:
        raise McpError(ErrorData(code=INVALID_PARAMS, message="URL is required"))

    content, prefix = await Fetch.fetch_url(url_str, Fetch.USER_AGENT, force_raw=raw)
    original_length = len(content)
    if start_index >= original_length:
        content = "<error>No more content available.</error>"
    else:
        truncated_content = content[start_index : start_index + max_length]
        if not truncated_content:
            content = "<error>No more content available.</error>"
        else:
            content = truncated_content
            actual_content_length = len(truncated_content)
            remaining_content = original_length - (start_index + actual_content_length)
            # Only add the prompt to continue fetching if there is still remaining content
            if actual_content_length == max_length and remaining_content > 0:
                next_start = start_index + actual_content_length
                content += f"\n\n<error>Content truncated. Call the fetch tool with a start_index of {next_start} to get more content.</error>"
    return [TextContent(type="text", text=f"{prefix}Contents of {url}:\n{content}")]


async def main():
    print("Starting Puch AI MCP Server...")
    print(f"Server will be available at: http://0.0.0.0:8085")
    print(f"MCP endpoint: http://0.0.0.0:8085/mcp")
    print(f"Auth token configured: {TOKEN}")
    print(f"Phone number configured: {MY_NUMBER}")
    
    await mcp.run_async(
        "streamable-http",
        host="0.0.0.0",
        port=8085,
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())