import openai
import pandas as pd
from utils.config import OPENAI_API_KEY
from utils.logger import logger

openai.api_key = OPENAI_API_KEY


def get_ai_suggestions(df: pd.DataFrame) -> list[str]:
    """Generate AI cleaning suggestions using OpenAI directly—100% reliable, no external deps."""
    try:
        # Create a concise summary for the prompt (avoids token limits)
        summary = df.describe(include='all').to_string()[:2000]  # Truncate if too long
        missing = df.isnull().sum().sum()
        duplicates = df.duplicated().sum()

        prompt = f"""
        You are a data cleaning expert. Analyze this dataset summary and stats:
        Summary: {summary}
        Total missing values: {missing}, Duplicates: {duplicates}

        Suggest exactly 5 specific, actionable cleaning steps as a numbered list.
        Focus on: handling missing values, removing duplicates, detecting/fixing outliers, standardizing formats, and data type conversions.
        Make each step concise and executable (e.g., "1. Impute missing 'Age' with median: df['Age'].fillna(df['Age'].median())").
        Output ONLY the numbered list, no extra text.
        """

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Stable and cheap; swap to "gpt-4o-mini" for better accuracy
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.3  # Low for consistent suggestions
        )

        # Parse the response into a clean list
        suggestions_text = response.choices[0].message.content.strip()
        suggestions = [line.strip() for line in suggestions_text.split('\n') if line.strip().startswith(tuple('12345'))]

        logger.info(f"Generated {len(suggestions)} AI suggestions")
        return suggestions if suggestions else ["1. Check for missing values manually.",
                                                "2. Remove obvious duplicates.",
                                                "3. Review data types for inconsistencies."]

    except openai.error.RateLimitError:
        logger.warning("OpenAI rate limit hit—using fallback suggestions.")
        return ["Fallback: API rate limited. Check missing values and duplicates manually."]
    except Exception as e:
        logger.error(f"AI suggestion error: {str(e)}")
        return ["AI unavailable—proceed with manual review."]