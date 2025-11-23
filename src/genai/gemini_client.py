import os
from google import generativeai as genai

class GeminiClient:
    def __init__(self, model="gemini-2.0-flash-lite"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("❌ GEMINI_API_KEY not set in environment variables.")

        genai.configure(api_key=api_key)
        self.model = model

    def explain_prediction(self, features: dict, prediction: str):
        prompt = f"""
        You are an advertising insights analyst.

        Input Features: {features}
        Model Prediction: {prediction}

        Task:
        1. Explain in 3 bullet points why the ad prediction is {prediction}.
        2. Provide 3 actionable improvements (max 12 words each).
        3. Give 2 alternative short ad headlines (max 8 words each).
        """

        response = genai.generate_text(
            model=self.model,
            prompt=prompt,
            max_output_tokens=250,
            temperature=0.3
        )
        return response.text

    def improve_ad(self, features: dict):
        prompt = f"""
        Provide 5 improvements to strengthen this advertisement:
        {features}
        Keep each suggestion under 12 words.
        """

        response = genai.generate_text(
            model=self.model,
            prompt=prompt,
            max_output_tokens=200
        )
        return response.text
