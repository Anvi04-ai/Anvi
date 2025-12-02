# src/data_cleaning/ai_suggestor.py
import openai
import pandas as pd
from utils.config import OPENAI_API_KEY
from utils.logger import logger
from data_cleaning.profiler import profile_data

openai.api_key = OPENAI_API_KEY

def get_ai_suggestions(df: pd.DataFrame, model: str = "gpt-3.5-turbo") -> list[str]:
    try:
        profile = profile_data(df)
        summary = df.describe(include='all').to_string()[:1500]

        prompt = f"""
        Dataset profile: Shape {profile.get('shape')}, Missing: {sum(profile.get('missing', {}).values())}, Duplicates: {profile.get('duplicates', 0)}
        Summary excerpt: {summary}

        Provide exactly 5 numbered cleaning steps with Pandas code.
        Output only the numbered list.
        """

        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.2
        )

        raw = response.choices[0].message.content.strip()
        suggestions = [line.strip() for line in raw.split('\n') if line.strip() and line[0].isdigit()]
        return suggestions or ["Review data manually."]

    except Exception as e:
        logger.error(f"AI failed: {e}")
        return ["1. Check for missing values", "2. Remove duplicates", "3. Fix data types"]