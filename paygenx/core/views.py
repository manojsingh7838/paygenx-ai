from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import QA
from .serializers import QASerializer
from .faiss_index import search_similar, add_to_index, embed_text
import requests
import os
import pickle

PERPLEXITY_API_KEY = "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"

def call_perplexity_api(prompt):
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "sonar-pro",
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=body
        )

        if response.status_code != 200:
            return f"Perplexity API error {response.status_code}: {response.text}"

        data = response.json()
        if "choices" in data and data["choices"]:
            return data["choices"][0]["message"]["content"]
        else:
            return "Perplexity API did not return a valid response."

    except Exception as e:
        return f"Exception occurred while calling Perplexity API: {str(e)}"

def refine_answer(answer):
    return f"Here’s what I found: {answer.strip().replace('Perplexity', 'our system')}"

class AskView(APIView):
    def post(self, request):
        question = request.data.get("question")
        if not question:
            return Response({"error": "Question is required"}, status=400)

        # Check FAISS for similar question
        similar = search_similar(question)
        if similar:
            q, a = similar
            return Response({
                "source": "local",
                "question": q,
                "answer": a
            })

        # No match, call Perplexity
        raw_answer = call_perplexity_api(question)

        # If error returned from Perplexity API
        if raw_answer.startswith("Perplexity API error") or raw_answer.startswith("Exception"):
            return Response({
                "source": "perplexity",
                "error": raw_answer
            }, status=500)

        refined = refine_answer(raw_answer)
        vec = embed_text(question)

        # Save in database
        obj = QA.objects.create(
            question=question,
            answer=refined,
            embedding=pickle.dumps(vec)
        )

        # Save in FAISS
        add_to_index(question, refined)

        return Response({
            "source": "perplexity",
            "question": question,
            "answer": refined
        })
