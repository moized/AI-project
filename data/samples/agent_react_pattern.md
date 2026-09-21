# ReAct (Reasoning and Acting) Agent Pattern

## Concept
The ReAct pattern integrates reasoning and action in language models. Instead of just generating a final response, the agent interleaves reasoning traces (Thought) and task-specific actions (Action).

## Execution Loop
1. **Thought**: The agent analyzes the user request and decides what information or tool is needed.
2. **Action**: The agent invokes a tool (e.g., RAG retrieval, calculator, or mock web search).
3. **Observation**: The agent receives the result from the tool.
4. **Repeat**: Steps 1-3 are repeated until the agent has enough information to formulate the final answer.
