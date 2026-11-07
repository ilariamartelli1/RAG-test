def message_with_context(question: str, context: str) -> str:
    """
    Build the user message with context for the LLM prompt.
    Args:
        question (str): User question.
        context (str): Retrieved context.
    Returns:
        str: Formatted user message.
    """
    user_message = f"""Here is the relevant context from the document:

{context}

---

User question: {question}"""
    return user_message
