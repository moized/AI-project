# Provider Contracts

## LLM provider
Input:
- prompt/context assembled by the agent
- function declarations
- application-controlled tool executor

Output:
- final text
- zero or more executed tool call records

Guarantees:
- The model never directly executes application Python.
- Tool execution is performed only by the application's allow-listed executor.
- Tool loops are bounded by MAX_TOOL_ROUNDS.

## Embedding provider
Input:
- document title/text pairs or query strings

Output:
- one vector per input
- fixed dimension

Guarantees:
- Provider is replaceable without changing RAG business logic.
- Provider returns exactly one vector per input.
- Vector dimension matches the Qdrant collection.
