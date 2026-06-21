
import os
import json
import time
import logging

from groq import Groq

logger = logging.getLogger(__name__)

# --- Configuration ---------------------------------------------------------

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    logger.warning(
        "GROQ_API_KEY not set. Falling back to keyword feedback."
    )
    _client = None
else:
    _client = Groq(api_key=GROQ_API_KEY)


# --- Prompt ------------------------------------------------------------

_PROMPT_TEMPLATE = """You are an experienced technical recruiter reviewing how well a candidate's resume matches a job description.

You are given:
1. A similarity score (0-1) already computed using TF-IDF/cosine similarity.
2. A list of keywords/skills that matched between the resume and JD.
3. A list of keywords/skills present in the JD but missing from the resume.
4. The raw resume text and job description text for additional context.

Your job is to act as a senior technical recruiter and ATS reviewer.

Provide professional, concise, and actionable feedback.

Guidelines:
- Focus on skills, projects, technologies, and experience relevant to the job description.
- Highlight the candidate's strongest matching qualifications.
- Identify important missing skills or qualifications.
- Suggest specific resume improvements that can increase interview chances.
- Avoid generic advice.
- Do not repeat the same point across multiple sections.
- Keep each bullet short and impactful.
- Write feedback in a professional recruiter tone.
- Base feedback on both the resume content and the job description.
- Do not mention or quote the similarity score in the summary.

Respond ONLY with valid JSON, no markdown formatting, no backticks, in this exact shape:
{{
  "summary": "2-3 sentence plain-English verdict on fit",
  "strengths": ["short bullet", "short bullet"],
  "gaps": ["short bullet", "short bullet"],
  "suggestions": ["specific, actionable rewrite/addition suggestion", "another one"]
}}

Similarity score: {match_score}
Matched keywords: {matched_keywords}
Missing keywords: {missing_keywords}

Resume text (truncated):
{resume_excerpt}

Job description text (truncated):
{jd_excerpt}
"""


def _truncate(text: str, max_chars: int = 3000) -> str:
    """Keep prompts small to control token usage/cost and latency."""
    text = text or ""
    return text[:max_chars]


def _build_prompt(resume_text, job_description, match_score, matched_keywords, missing_keywords):
    return _PROMPT_TEMPLATE.format(
        match_score=round(match_score, 3),
        matched_keywords=", ".join(matched_keywords) if matched_keywords else "none detected",
        missing_keywords=", ".join(missing_keywords) if missing_keywords else "none detected",
        resume_excerpt=_truncate(resume_text),
        jd_excerpt=_truncate(job_description),
    )


def _fallback_feedback(match_score, matched_keywords, missing_keywords):
    """Used if the LLM call fails or no API key is configured, so the
    app degrades gracefully instead of crashing or hanging."""
    return {
        "summary": (
            f"This resume matches approximately {round(match_score * 100)}% "
            "of the job requirements based on keyword analysis."
        ),
        "strengths": matched_keywords[:5],
        "gaps": missing_keywords[:5],
        "suggestions": [
            "Add the missing keywords above to your resume if you genuinely have that experience."
        ],
    }


def generate_feedback(
    resume_text: str,
    job_description: str,
    match_score: float,
    matched_keywords: list,
    missing_keywords: list,
    retries: int = 2,
) -> dict:
    """
    Calls Gemini to generate qualitative feedback layered on top of the
    existing TF-IDF score. Returns a dict with summary/strengths/gaps/suggestions.

    Never raises -- on any failure it returns the deterministic fallback
    so the rest of the app keeps working even if the LLM call fails.
    """
    if _client is None:
        return _fallback_feedback(match_score, matched_keywords, missing_keywords)

    prompt = _build_prompt(
        resume_text, job_description, match_score, matched_keywords, missing_keywords
    )

    last_error = None
    for attempt in range(retries + 1):
        try:
            response = _client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3
            )

            raw = response.choices[0].message.content.strip()
            # Strip accidental markdown fences in case the model adds them anyway
            raw = raw.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(raw)
            for key in ("summary", "strengths", "gaps", "suggestions"):
                parsed.setdefault(key, [])
            return parsed
        except Exception as e:
            last_error = e
            logger.warning("Gemini feedback attempt %d failed: %s", attempt + 1, e)
            time.sleep(1)

    logger.error("All Gemini feedback attempts failed: %s", last_error)
    return _fallback_feedback(match_score, matched_keywords, missing_keywords)
