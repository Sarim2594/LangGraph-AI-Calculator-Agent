"""
LangGraph AI Agent Template
PostgreSQL Integrated 
Calculator Agent
Supports both OpenAI and Gemini
"""
import os
import sys
import warnings

# Suppress gRPC/ALTS warnings - MUST be set before importing Google libraries
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GRPC_TRACE'] = ''
os.environ['GRPC_VERBOSITY'] = 'NONE'

# Temporarily redirect stderr during imports
import io
stderr_backup = sys.stderr
sys.stderr = io.StringIO()

from typing import TypedDict, Annotated, List, Union
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import operator
from dotenv import load_dotenv
import psycopg2

# Restore stderr
sys.stderr = stderr_backup

warnings.filterwarnings('ignore')

# ============================================================================
# DATABASE CONNECTION
# ============================================================================

def get_connection():
    return psycopg2.connect(
        host="localhost",
        database="calculator_ai_agent",
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        port="5432"
    )

def log_operation(operation_type, a, b, result):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO operations (operation_type, operand1, operand2, result)
        VALUES (%s, %s, %s, %s)
        """,
        (operation_type, a, b, result)
    )
    conn.commit()
    cur.close()
    conn.close()

# ============================================================================
# CONFIGURATION
# ============================================================================

load_dotenv()

# LangSmith Configuration
os.environ['LANGSMITH_TRACING'] = 'true'
os.environ['LANGSMITH_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGSMITH_API_KEY'] = os.getenv('LANGSMITH_API_KEY')
os.environ['LANGSMITH_PROJECT'] = os.getenv('LANGSMITH_PROJECT')

def get_llm(model: str, temperature: float = 0):
    """
    Returns an initialized LLM (ChatOpenAi) based on the provider
    
    Args:
        model: Model name (optional, uses default if not provided)
        temperature: Temperature for generation
    """
    # OpenRouter configuration
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise ValueError("API_KEY not found in environment variables")
    
    if model.startswith("gpt") or model.startswith("openai/"):
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "http://localhost:3000",  # Optional: your site URL
                "X-Title": "LangGraph Agent",  # Optional: your app name
            },
        )
    elif model.startswith("gemini") or model.startswith("google/"):
        llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=api_key,
            
        )
    
    return llm

# ============================================================================
# 1. DEFINE STATE
# ============================================================================

class AgentState(TypedDict):
    """
    Define the state structure that flows through the graph.
    This tracks the conversation and any intermediate results.
    """
    messages: Annotated[List[BaseMessage], operator.add]
    # Add custom fields as needed:
    # task_completed: bool
    # user_data: dict
    # iteration_count: int

# ============================================================================
# 2. DEFINE TOOLS
# ============================================================================

def adder(num1: Union[int, float], num2: Union[int, float]) -> Union[int, float]:
    """
    Adds two numbers.
    """
    result = num1 + num2
    log_operation("add", num1, num2, result)
    return result

def mutiplier(num1: Union[int, float], num2: Union[int, float]) -> Union[int, float]:
    """
    Multiplies two numbers.
    """
    result = num1 * num2
    log_operation("multiply", num1, num2, result)
    return result

def subtractor(num1: Union[int, float], num2: Union[int, float]) -> Union[int, float]:
    """
    Subtracts two numbers.
    """
    result = num1 - num2
    log_operation("subtract", num1, num2, result)
    return result

def divider(num1: Union[int, float], num2: Union[int, float]) -> Union[int, float]:
    """
    Divides two numbers.
    """
    if num2 == 0:
        raise ValueError("Cannot divide by zero")
    result = num1 / num2
    log_operation("divide", num1, num2, result)
    return result

def get_last_operations(limit: int = 5):
    """
    Retrieves the last 'limit' operations from the database.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM operations ORDER BY timestamp DESC LIMIT %s", (limit,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

def truncate_operations():
    """
    Truncates the operations table.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("TRUNCATE TABLE operations RESTART IDENTITY;")
    conn.commit()
    cur.close()
    conn.close()

tools = [adder, mutiplier, subtractor, divider, get_last_operations, truncate_operations]

# ============================================================================
# 3. DEFINE NODES
# ============================================================================

def agent_node(state: AgentState) -> AgentState:
    """
    Main agent node that uses LLM to decide actions.
    """
    # Initialize LLM with tools
    llm = get_llm(model='gemini-2.5-flash', temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    
    # Get LLM response
    response = llm_with_tools.invoke(state["messages"])
    
    # Return updated state
    return {"messages": [response]}

# ============================================================================
# 4. DEFINE ROUTING LOGIC
# ============================================================================

def should_continue(state: AgentState) -> str:
    """
    Determines whether to continue with tools or end the conversation.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # If there are tool calls, continue to tools
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "continue"
    
    # Otherwise, end the conversation
    return "end"

# ============================================================================
# 5. BUILD THE GRAPH
# ============================================================================

def create_agent_graph():
    """
    Construct the LangGraph workflow.
    """
    # Initialize graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))
    
    # Add edge from start to agent
    workflow.add_edge(START, "agent")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END
        }
    )
    
    # Add edge from tools back to agent
    workflow.add_edge("tools", "agent")
    
    # Compile the graph
    app = workflow.compile()
    
    return app

# ============================================================================
# 6. RUN THE AGENT
# ============================================================================

system_prompt = """
If the user requests an impossible operation (like division by zero), respond politely indicating that the operation cannot be performed and explain the reason.
"""

def run_agent(user_input: str, conversation_history: List[BaseMessage] = None):
    """
    Execute the agent with user input.
    
    Args:
        user_input: User's input string
        conversation_history: Optional list of previous messages for context
    
    Returns:
        Tuple of (final_state, updated_conversation_history)
    """
    # Create the graph
    app = create_agent_graph()
    
    # Initialize state with conversation history
    if conversation_history is None:
        conversation_history = []
    
    initial_state = {
        "messages": [AIMessage(content=system_prompt)] + conversation_history + [HumanMessage(content=user_input)]
    }
    
    # Get final state
    final_state = app.invoke(initial_state)
    
    return final_state, final_state["messages"]

# ============================================================================
# 7. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":    
    # Example usage
    print("Calculator AI Agent (type 'exit', 'quit', or 'bye' to stop)\n")
    
    input_prompt = input("You: ")
    user_query = input_prompt.strip()
    
    result, conversation_history = run_agent(user_query)
    
    print("AI:", result["messages"][-1].content)
    
    # For interactive mode with conversation history
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("\nGoodbye!")
            break
        
        result, conversation_history = run_agent(user_input, conversation_history)
        print("AI:", result["messages"][-1].content)