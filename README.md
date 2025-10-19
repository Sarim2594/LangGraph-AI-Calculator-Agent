# 🤖 LangGraph AI Calculator Agent

An **intelligent calculator agent** built using **LangGraph**, **LangChain**, and **PostgreSQL** — capable of performing arithmetic operations while maintaining a history of interactions and results in a database.  
This agent supports both **OpenAI (via OpenRouter)** and **Google Gemini** models.

---

## 🚀 Features

- 🔗 **LangGraph-powered workflow:** Manages stateful conversations and tool calling.
- 🧮 **AI Calculator Tools:** Perform addition, subtraction, multiplication, and division.
- 💾 **PostgreSQL Integration:** Logs all operations with timestamps for traceability.
- 🧠 **LLM Flexible:** Works seamlessly with both **OpenAI GPT models** and **Google Gemini**.
- 🗂️ **Operation History:** Retrieve or clear previous operations.
- ⚙️ **LangSmith Tracing:** Monitor the reasoning and execution flow of your agent.

---

## 🧩 Components

### 🧠 LangGraph Agent
- Handles conversation flow using `StateGraph`.
- Determines whether to continue the loop or end based on tool calls.
- Integrates with `ToolNode` for executing defined calculator tools.

### 🧮 Calculator Tools
- **Adder** → Adds two numbers  
- **Subtractor** → Subtracts two numbers  
- **Multiplier** → Multiplies two numbers  
- **Divider** → Divides two numbers (handles division by zero gracefully)  
- **get_last_operations** → Fetches recent operations from PostgreSQL  
- **truncate_operations** → Clears all logged operations  

Each tool automatically logs the performed operation to the database.

### 🧾 Database Layer
- Uses `psycopg2` for PostgreSQL interaction.  
- Stores each operation in an `operations` table with details like operation type, operands, result, and timestamp.

### 🧰 LLM Configuration
- `get_llm()` dynamically initializes:
  - **OpenAI models (via OpenRouter)**
  - **Google Gemini models**
- Controlled by environment variables for API keys and model selection.

---

## 🧠 Example Interaction
```
Calculator AI Agent (type 'exit', 'quit', or 'bye' to stop)

You: What is 8 + 12?
AI: The sum of 8 and 12 is 20.

You: Multiply that by 3
AI: 20 × 3 = 60.

You: Show my last operations
AI: Here are your 5 most recent operations:
1. add(8, 12) → 20
2. multiply(20, 3) → 60
```
---

## 🔍 Environment Variables

Create a .env file in the project directory with the following variables:
```
API_KEY=your_openrouter_or_gemini_api_key
DB_USER=your_postgres_user
DB_PASSWORD=your_postgres_password
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=LangGraph Calculator
```
---

## 🧾 Database Schema
```
CREATE TABLE operations (
    id SERIAL PRIMARY KEY,
    operation_type TEXT NOT NULL,
    operand1 DOUBLE PRECISION NOT NULL,
    operand2 DOUBLE PRECISION NOT NULL,
    result DOUBLE PRECISION NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```
---

## 💬 System Prompt

> "If the user requests an impossible operation (like division by zero), respond politely indicating that the operation cannot be performed and explain the reason."

---

## 🧰 Supported Operations

| Function              | Description             | Example                 |
| --------------------- | ----------------------- | ----------------------- |
| `adder`               | Adds two numbers        | 5 + 3 = 8               |
| `subtractor`          | Subtracts two numbers   | 10 - 6 = 4              |
| `mutiplier`           | Multiplies two numbers  | 7 × 4 = 28              |
| `divider`             | Divides two numbers     | 20 ÷ 5 = 4              |
| `get_last_operations` | Shows recent operations | last 5 operations       |
| `truncate_operations` | Clears the log table    | reset operation history |

---

## 🧩 Integration Highlights

 - LangGraph: For dynamic agent workflows.
 - LangChain: For structured AI message handling.
 - PostgreSQL: Persistent operation logging.
 - LangSmith: Tracing and debugging support.
 - OpenAI / Gemini APIs: For model inference flexibility.

> [!NOTE]
> Only supports open router OpenAI and offical Gemini API
