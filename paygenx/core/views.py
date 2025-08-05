from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import QA
from .serializers import QASerializer
from .faiss_index import search_similar, add_to_index, embed_text
import requests
import os
from .train_local_model import train_on_new_qa

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





# class AskView(APIView):
#     def post(self, request):
#         question = request.data.get("question")
#         if not question:
#             return Response({"error": "Question is required"}, status=400)

#         # Check FAISS for similar question
#         similar = search_similar(question)
#         if similar:
#             q, a = similar
#             return Response({
#                 "source": "local",
#                 "question": q,
#                 "answer": refine_answer(a)  # Even local answer refined
#             })

#         # No match → call Perplexity
#         raw_answer = call_perplexity_api(question)

#         if raw_answer.startswith("Perplexity API error") or raw_answer.startswith("Exception"):
#             return Response({
#                 "source": "perplexity",
#                 "error": raw_answer
#             }, status=500)

#         refined = refine_answer(raw_answer)
#         vec = embed_text(question)

#         # Save to DB
#         obj = QA.objects.create(
#             question=question,
#             answer=refined,
#             embedding=pickle.dumps(vec)
#         )

#         # Save to FAISS
#         add_to_index(question, refined)

#         return Response({
#             "source": "perplexity",
#             "question": question,
#             "answer": refined
#         })


import json
import torch
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .faiss_index import search_similar, add_to_index
from .models import QA

from .train_local_model import train_on_new_qa  

class AskAPIView(APIView):
    def post(self, request):
        user_question = request.data.get("question", "").strip()

        if not user_question:
            return Response({"error": "Question is required."}, status=status.HTTP_400_BAD_REQUEST)

        # Check similar question in FAISS
        result = search_similar(user_question, top_k=3, threshold=0.85)

        if result:
            all_answers = []
            matched_questions = []

            # For each similar result, collect answers and train on them
            for q, a, db_id in result:
                matched_questions.append(q)
                all_answers.append(a)
                train_on_new_qa(q, a)  # ✅ Train model on matched Q&A

            # Merge + refine all similar answers using Perplexity
            merged_answer = "\n\n".join(all_answers)
            prompt = f"User asked: {user_question}\n\nThese are similar answers:\n{merged_answer}\n\nGive one final clean, relevant answer for this user query."
            refined = refine_answer(call_perplexity_api(prompt))

            return Response({
                "source": "matched",
                "matched_questions": matched_questions,
                "refined_answer": refined
            })

        # Call Perplexity if no match found
        base_answer = call_perplexity_api(user_question)
        refined_answer = refine_answer(base_answer)

        # Save in DB
        new_qa = QA.objects.create(
            question=user_question,
            answer=refined_answer
        )

        # Add to FAISS
        add_to_index(user_question, refined_answer)

        # Train model on new data
        train_on_new_qa(user_question, refined_answer)

        return Response({
            "source": "google",
            "refined_answer": refined_answer
        }, status=status.HTTP_200_OK)
