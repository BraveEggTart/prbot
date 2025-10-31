import os
import re
import hmac
import hashlib
import json
import logging
import time
import base64
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from functools import wraps

import requests
import openai
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


# --- 1. Configuration Management ---
@dataclass
class Config:
    GITHUB_TOKEN: str
    GITHUB_WEBHOOK_SECRET: str
    OPENAI_API_KEY: str
    OPENAI_API_BASE: Optional[str] = None

    AI_MODEL_NAME: str = 'gpt-4o'
    REVIEW_LABEL: str = 'ReviewedByUllrAI'
    MAX_PROMPT_LENGTH: int = 100000
    INCLUDE_FILE_CONTEXT: bool = True
    CONTEXT_MAX_LINES: int = 400
    CONTEXT_SURROUNDING_LINES: int = 50

    MAX_RETRY_ATTEMPTS: int = 3
    RETRY_DELAY: float = 2.0
    REQUEST_TIMEOUT: int = 60
    MAX_FILES_PER_REVIEW: int = 50
    OUTPUT_LANGUAGE: str = 'english'

    @classmethod
    def from_env(cls) -> 'Config':
        required_vars = ['GITHUB_TOKEN', 'OPENAI_API_KEY']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

        return cls(
            GITHUB_TOKEN=os.getenv('GITHUB_TOKEN'),
            GITHUB_WEBHOOK_SECRET=os.getenv('GITHUB_WEBHOOK_SECRET', ''),
            OPENAI_API_KEY=os.getenv('OPENAI_API_KEY'),
            OPENAI_API_BASE=os.getenv('OPENAI_API_BASE'),
            AI_MODEL_NAME=os.getenv('AI_MODEL_NAME', 'gpt-4o'),
            REVIEW_LABEL=os.getenv('REVIEW_LABEL', 'ReviewedByUllrAI'),
            MAX_PROMPT_LENGTH=int(os.getenv('MAX_PROMPT_LENGTH', '100000')),
            INCLUDE_FILE_CONTEXT=os.getenv('INCLUDE_FILE_CONTEXT', 'true').lower() in ('true', '1', 't'),
            CONTEXT_MAX_LINES=int(os.getenv('CONTEXT_MAX_LINES', '400')),
            CONTEXT_SURROUNDING_LINES=int(os.getenv('CONTEXT_SURROUNDING_LINES', '50')),
            MAX_RETRY_ATTEMPTS=int(os.getenv('MAX_RETRY_ATTEMPTS', '3')),
            RETRY_DELAY=float(os.getenv('RETRY_DELAY', '2.0')),
            REQUEST_TIMEOUT=int(os.getenv('REQUEST_TIMEOUT', '60')),
            MAX_FILES_PER_REVIEW=int(os.getenv('MAX_FILES_PER_REVIEW', '50')),
            OUTPUT_LANGUAGE=os.getenv('OUTPUT_LANGUAGE', 'english'),
        )


# --- 2. Logging Configuration ---
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    return logging.getLogger(__name__)


# --- 3. Error Handling and Retry Decorator ---
def retry_on_failure(max_attempts: int = 3, delay: float = 1.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"Function {func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {e}. Retrying in {delay * (attempt + 1):.1f} seconds..."
                        )
                        time.sleep(delay * (attempt + 1))
                    else:
                        logger.error(f"Function {func.__name__} failed after {max_attempts} attempts.")
            raise last_exception
        return wrapper
    return decorator


# --- 4. Initialization ---
logger = setup_logging()
config = Config.from_env()
app = FastAPI(title="PR Code Review Bot", version="2.1.0")

# OpenAI client setup
openai.api_key = config.OPENAI_API_KEY
if config.OPENAI_API_BASE:
    openai.api_base = config.OPENAI_API_BASE
    logger.info(f"Using custom OpenAI API base: {config.OPENAI_API_BASE}")


class GitHubClient:
    def __init__(self, token: str, timeout: int):
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28"
        })
        self.timeout = timeout

    @retry_on_failure(max_attempts=config.MAX_RETRY_ATTEMPTS)
    def get_pr_details(self, owner: str, repo: str, pr_number: int) -> Dict[str, Any]:
        logger.info(f"[GitHub API] ==> 'get_pr_details' for PR #{pr_number}")
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    @retry_on_failure(max_attempts=config.MAX_RETRY_ATTEMPTS)
    def get_pr_files(self, owner: str, repo: str, pr_number: int) -> List[Dict[str, Any]]:
        logger.info(f"[GitHub API] ==> 'get_pr_files' for PR #{pr_number}")
        url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/files"
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        files = response.json()
        logger.info(f"  Output: Successfully retrieved {len(files)} file changes.")
        return files

    @retry_on_failure(max_attempts=config.MAX_RETRY_ATTEMPTS)
    def get_file_content_from_repo(self, owner: str, repo: str, file_path: str, ref: str) -> str:
        logger.info(f"[GitHub API] ==> 'get_file_content_from_repo'")
        logger.info(f"  Input: owner={owner}, repo={repo}, path={file_path}, ref={ref}")
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
        params = {"ref": ref}
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        if 'content' not in data:
            raise ValueError(f"'content' field not found for file '{file_path}' in API response.")
        content_b64 = data['content']
        content_bytes = base64.b64decode(content_b64)
        try:
            content_str = content_bytes.decode('utf-8')
        except UnicodeDecodeError:
            logger.warning(
                f"Failed to decode file '{file_path}' as UTF-8, using latin-1 with replacement characters."
            )
            content_str = content_bytes.decode('latin-1', errors='replace')
        logger.info(f"  Output: Successfully retrieved and decoded file content, size {len(content_str)} bytes.")
        return content_str

    @retry_on_failure(max_attempts=config.MAX_RETRY_ATTEMPTS)
    def post_comment(self, owner: str, repo: str, pr_number: int, comment: str) -> Dict[str, Any]:
        logger.info(f"[GitHub API] ==> 'post_comment' on PR #{pr_number}")
        url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/comments"
        response = self.session.post(url, json={"body": comment}, timeout=self.timeout)
        response.raise_for_status()
        response_json = response.json()
        logger.info(f"  Output: Comment successfully posted. URL: {response_json.get('html_url')}")
        return response_json

    @retry_on_failure(max_attempts=config.MAX_RETRY_ATTEMPTS)
    def add_label(self, owner: str, repo: str, pr_number: int, label: str) -> None:
        logger.info(f"[GitHub API] ==> 'add_label' on PR #{pr_number}")
        url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/labels"
        response = self.session.post(url, json={"labels": [label]}, timeout=self.timeout)
        response.raise_for_status()
        logger.info(f"  Output: Label '{label}' successfully added.")


github_client = GitHubClient(config.GITHUB_TOKEN, config.REQUEST_TIMEOUT)


class PRReviewer:
    @staticmethod
    def _get_context_line_from_patch(patch: str) -> int:
        match = re.search(r"@@ -(\d+),?\d* \+", patch)
        return int(match.group(1)) if match else 1

    @staticmethod
    def create_review_prompt(files: List[Dict[str, Any]], pr_data: Dict[str, Any]) -> str:
        logger.info("[Step] Creating AI review prompt...")
        files = files[:config.MAX_FILES_PER_REVIEW]

        repo_info = pr_data.get("base", {}).get("repo", {})
        owner = repo_info.get("owner", {}).get("login")
        repo = repo_info.get("name")
        base_sha = pr_data.get("base", {}).get("sha")
        head_sha = pr_data.get("head", {}).get("sha")

        prompt_parts = []
        total_length = 0

        for file in files:
            filename = file.get('filename', 'unknown')
            patch = file.get('patch', '')
            status = file.get('status', 'modified')

            file_prompt = f"## File: `{filename}` (Status: {status})\n\n"

            if config.INCLUDE_FILE_CONTEXT and status == 'modified' and head_sha and owner and repo:
                try:
                    modified_content = github_client.get_file_content_from_repo(
                        owner, repo, filename, head_sha
                    )
                    lines = modified_content.splitlines()

                    context_header = "### Modified File Context"
                    if len(lines) > config.CONTEXT_MAX_LINES:
                        start_line = PRReviewer._get_context_line_from_patch(patch)
                        slice_start = max(0, start_line - config.CONTEXT_SURROUNDING_LINES)
                        slice_end = min(len(lines), start_line + config.CONTEXT_SURROUNDING_LINES)
                        context_content = "\n".join(lines[slice_start:slice_end])
                        context_header += f" (Code snippet around line {start_line})"
                        logger.info(
                            f"  - Extracted code snippet for '{filename}' ({slice_end - slice_start} lines)."
                        )
                    else:
                        context_content = modified_content
                        context_header += " (Complete file)"
                        logger.info(
                            f"  - Included complete file content for '{filename}' ({len(lines)} lines)."
                        )

                    file_prompt += f"{context_header}\n```\n{context_content}\n```\n\n"
                except Exception as e:
                    logger.warning(f"  - Unable to get context for '{filename}': {e}")
                    file_prompt += "_[Unable to get modified file context]_\n\n"

            safe_patch = patch.replace("```", "`` `") if patch else "_No changes_"
            file_prompt += f"### Diff for This Commit\n```diff\n{safe_patch}\n```\n\n---\n\n"

            if total_length + len(file_prompt) > config.MAX_PROMPT_LENGTH:
                logger.warning(
                    f"  Prompt length reached limit. Stopped after processing {len(prompt_parts)} files."
                )
                prompt_parts.append("\n_[More files omitted due to total length limit...]_")
                break

            prompt_parts.append(file_prompt)
            total_length += len(file_prompt)

        diffs_text = "".join(prompt_parts)
        pr_title = pr_data.get('title', '')
        pr_body = pr_data.get('body', '')

        language_instruction = PRReviewer._get_language_instruction(config.OUTPUT_LANGUAGE)

        prompt = f"""# Review Instructions
Please conduct a professional and thorough review of the following code changes. You have access to both the modified file content (showing the final state after changes) and the diff (showing what was changed). Your goal is to identify potential issues and provide specific, constructive modification suggestions. Follow GitHub Code Review best practices, keep comments objective and concise, and prioritize by importance and urgency.

# Review Focus Areas
1.  **Logic and Functionality**: Does the modified code correctly implement its intended goals? Are there any bugs, logical flaws, or unhandled edge cases in the final implementation?
2.  **Performance**: Are there obvious performance bottlenecks in the modified code, such as unnecessary loops, inefficient queries, or memory issues?
3.  **Security**: Are there common security risks in the modified code (such as SQL injection, XSS, hardcoded sensitive information, etc.)?
4.  **Code Style and Readability**: If any, categorize and describe in the last issue. Does the modified code follow project or language best practices and common standards? But ignore some code style issues that don't affect logic, such as indentation, spaces, line breaks, etc.
5.  **Error Handling**: Are exceptions and error conditions properly handled in the modified code?

# PR Context
* **Title**: {pr_title}
* **Description**:
{pr_body}

# Output
Please use Markdown to format your review comments. For each finding, follow the template below. For issues that require urgent attention, use appropriate emphasis like ⚠️ in the title. If the code quality is good and no issues are found, please clearly state so.

---
**[1] Title**
* **Category**: [Logic Error / Performance / Security / Code Style / Suggestion etc.]
* **Code Location**: `[filename]:[line number]`
* **Description**: [Briefly describe the issue and its impact.]
* **Suggestion**:
    ```[language]
    // Paste suggested modified code snippet
    ```
---

# Code to Review
For each file, you will see:
1. **Modified File Context**: The complete file content or relevant code snippet after the changes have been applied
2. **Diff**: The specific changes made in this commit

Use both pieces of information together to understand the full context and impact of the changes.

{diffs_text}

Please start your review comments directly without any opening remarks.{language_instruction}"""

        logger.info(
            f"  Output: Prompt creation completed. Length: {len(prompt)} characters. Files processed: {len(prompt_parts)}."
        )
        return prompt

    @staticmethod
    def _get_language_instruction(output_language: str) -> str:
        if output_language and output_language.lower() != 'english':
            return f' Output language: {output_language}.'
        return ''

    @staticmethod
    @retry_on_failure(max_attempts=config.MAX_RETRY_ATTEMPTS, delay=config.RETRY_DELAY)
    def get_ai_review(prompt: str) -> str:
        logger.info("[Step] Calling OpenAI API for code review...")
        try:
            response = openai.ChatCompletion.create(
                model=config.AI_MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a senior software engineer performing a code review."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
            )
            review_text = response.choices[0].message['content']
            logger.info(
                f"  Output: Successfully received AI review comments, length {len(review_text)} characters."
            )
            return review_text
        except Exception as e:
            logger.error(f"  Error occurred during OpenAI API call: {e}")
            raise

    @staticmethod
    def process_pr_review(pr_data: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        pr_number = pr_data['number']
        repo_info = pr_data.get("base", {}).get("repo", {})
        owner = repo_info.get("owner", {}).get("login")
        repo = repo_info.get("name")

        logger.info(f"--- Starting review process for PR #{pr_number} in {owner}/{repo} ---")
        result = {"pr_number": pr_number, "status": "success", "message": "", "duration": 0}

        try:
            files = github_client.get_pr_files(owner, repo, pr_number)
            if not files:
                result.update({"status": "skipped", "message": "PR has no file changes."})
                return result

            prompt = PRReviewer.create_review_prompt(files, pr_data)
            review_comment = PRReviewer.get_ai_review(prompt)

            comment_with_footer = (
                f"{review_comment}\n\n---\n*🤖 This comment was generated by UllrAI Code Review Assistant using {config.AI_MODEL_NAME} model*"
            )
            github_client.post_comment(owner, repo, pr_number, comment_with_footer)
            github_client.add_label(owner, repo, pr_number, config.REVIEW_LABEL)

            result["message"] = f"Successfully reviewed {len(files)} files."

        except Exception as e:
            result.update({"status": "error", "message": str(e)})
            logger.error(f"--- PR #{pr_number} review process failed: {e} ---", exc_info=True)

        finally:
            duration = round(time.time() - start_time, 2)
            result["duration"] = duration
            logger.info(
                f"PR #{pr_number} total processing duration: {duration} seconds. Result: {result['status']}"
            )

        return result


# --- 6. Web Endpoints ---
@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "pr-reviewer", "version": "2.1.0", "model": config.AI_MODEL_NAME}


@app.post("/webhook")
async def github_webhook(request: Request):
    event_type = request.headers.get('X-GitHub-Event', 'unknown')
    delivery_id = request.headers.get('X-GitHub-Delivery', 'unknown')
    logger.info(f"--- Received Webhook request. Event: '{event_type}', Delivery ID: '{delivery_id}' ---")

    try:
        data = await request.json()
        should_process, pr_data = should_process_event(data, event_type)

        if not should_process:
            logger.info(f"Event (Delivery ID: {delivery_id}) does not need processing, skipped.")
            return JSONResponse({"status": "skipped", "reason": "Event does not meet processing conditions."}, status_code=200)

        pr_number = pr_data.get('number')
        logger.info(f"Event '{event_type}' (PR #{pr_number}) will be processed.")

        result = PRReviewer.process_pr_review(pr_data)
        return JSONResponse(result, status_code=200)

    except Exception as e:
        logger.error(
            f"Uncaught error occurred while processing Webhook, Delivery ID: {delivery_id}: {e}",
            exc_info=True,
        )
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


def should_process_event(data: Dict[str, Any], event_type: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
    repo_info = data.get('repository', {})
    owner = repo_info.get('owner', {}).get('login')
    repo = repo_info.get('name')
    logger.info(f"[Step] Determining if event '{event_type}' for '{owner}/{repo}' needs processing...")
    action = data.get('action')

    if event_type == 'pull_request' and action in ['opened', 'synchronize', 'reopened']:
        pr_data = data.get('pull_request', {})
        if pr_data and not pr_data.get('draft', False):
            logger.info(f"  Output: Processing 'pull_request.{action}' event for PR #{pr_data.get('number')}.")
            return True, pr_data
        else:
            logger.info("  Output: Skipped. PR is a draft.")

    elif event_type == 'issue_comment' and action == 'created':
        if 'pull_request' in data.get('issue', {}):
            comment_body = data.get('comment', {}).get('body', '')
            if '/review' in comment_body.lower():
                pr_number = data.get('issue', {}).get('number')
                try:
                    pr_data = github_client.get_pr_details(owner, repo, pr_number)
                    logger.info(
                        f"  Output: Processing 'issue_comment' event for PR #{pr_number} (triggered by '/review')."
                    )
                    return True, pr_data
                except Exception as e:
                    logger.error(
                        f"  Unable to get PR #{pr_number} details for comment-triggered review: {e}"
                    )
            else:
                logger.info("  Output: Skipped. Comment does not contain '/review' trigger command.")
        else:
            logger.info("  Output: Skipped. Comment is on Issue, not Pull Request.")

    logger.info(f"  Output: Skipped. Event '{event_type}.{action}' is not a target event.")
    return False, None


# --- 7. Error handlers are implicit in FastAPI via exception handlers ---


# --- 8. Application Startup ---
if __name__ == '__main__':
    import uvicorn
    port = int(os.getenv('PORT', 5001))
    logger.info("="*50)
    logger.info(f"PR Code Review Bot starting (v2.1.0)")
    logger.info(f"Listening on port: {port}")
    logger.info(f"AI Model: {config.AI_MODEL_NAME}")
    logger.info(f"Include file context: {config.INCLUDE_FILE_CONTEXT}")
    logger.info("="*50)
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)


