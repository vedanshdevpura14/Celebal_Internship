import os
from main import call_llm, process_chat, get_db_connection

def evaluate_response(query: str, response: str, context: str) -> dict:
    """Uses LLM-as-a-judge to evaluate response quality."""
    prompt = f"""
    Evaluate the following chatbot response out of 10 based on two metrics:
    1. Correctness: Does it directly and accurately answer the user's query?
    2. Context Relevance: Does it properly use the retrieved context?

    Query: {query}
    Context Available: {context}
    Bot Response: {response}

    Return pure JSON format: {{"correctness": 8, "context_relevance": 9, "feedback": "good answer"}}
    """
    
    try:
        eval_result = call_llm(prompt, "Return ONLY valid JSON.")
        start = eval_result.find("{")
        end = eval_result.rfind("}") + 1
        if start != -1 and end != -1:
            import json
            return json.loads(eval_result[start:end])
    except Exception as e:
        print(f"Eval failed: {e}")
        
    return {"correctness": 0, "context_relevance": 0, "feedback": "Failed to parse eval"}

def run_evaluation():
    print("--- Starting Beginner Evaluation Framework ---")
    test_queries = [
        "What is machine learning?",
        "What are the main goals of Artificial Intelligence?",
        "Tell me a joke."
    ]
    
    for i, query in enumerate(test_queries):
        print(f"\nTest {i+1}: '{query}'")
        # Run through the pipeline
        result = process_chat(query, "eval_user", "eval_session")
        response = result["response"]
        source = result["source_used"]
        
        # Get context to evaluate against
        context = ""
        if source != "Direct":
            from main import retrieve_rag, web_search, get_graph_context
            if "Web Search" in source:
                context = web_search(query)
            else:
                context = retrieve_rag(query) + "\n" + get_graph_context(query)
                
        eval_scores = evaluate_response(query, response, context)
        print(f"Source Used: {source}")
        print(f"Correctness: {eval_scores.get('correctness')}/10")
        print(f"Context Relevance: {eval_scores.get('context_relevance')}/10")
        print(f"Feedback: {eval_scores.get('feedback')}")
        print("-" * 40)
        
if __name__ == "__main__":
    run_evaluation()
