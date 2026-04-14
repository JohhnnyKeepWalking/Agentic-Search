import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional, Union
from enum import Enum, auto

from src.config.logging import logger
from src.config.setup import MODEL, initialize_genai_client
from src.llm.gemini_text import generate_content
from src.tools.registry import (
    get_google_search_results,
    get_multimodal_reasoning,
    get_walmart_basic_search,
)
from src.utils.io import read_file

PROMPT_TEMPLATE_PATH = "./templates/react_simple.txt"


class ToolName(Enum):
    """Available tools for the agent."""
    GOOGLE_SEARCH = auto()
    WALMART_SEARCH = auto()
    GEMINI_MULTIMODAL = auto()
    NONE = "none"


# Tool registry mapping names to functions
TOOL_REGISTRY: Dict[ToolName, Callable] = {
    ToolName.GOOGLE_SEARCH: get_google_search_results,
    ToolName.WALMART_SEARCH: get_walmart_basic_search,
    ToolName.GEMINI_MULTIMODAL: get_multimodal_reasoning,
}


@dataclass
class Message:
    """A message in the conversation history."""
    role: str
    content: str

    def __post_init__(self):
        if isinstance(self.content, dict):
            self.content = json.dumps(self.content)


@dataclass
class AgentState:
    """Tracks the agent's current state."""
    query: str = ""
    image_path: Optional[str] = None
    messages: List[Message] = field(default_factory=list)
    iteration: int = 0
    last_result: Optional[Any] = None


class Agent:
    """
    ReAct agent that iteratively thinks and acts to answer queries.
    """

    def __init__(self, model: str = MODEL, max_iterations: int = 5):
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        
        self.model = model
        self.max_iterations = max_iterations
        self.client = initialize_genai_client()
        self.template = read_file(PROMPT_TEMPLATE_PATH)
        self.state = AgentState()

    def reset(self):
        """Reset agent state for a new query."""
        self.state = AgentState()

    def trace(self, role: str, content: Union[str, dict]) -> None:
        """Log a message to history (skip system messages)."""
        if role != "system":
            self.state.messages.append(Message(role=role, content=content))

    def get_history(self) -> str:
        """Get formatted conversation history."""
        lines = [f"{m.role}: {m.content}" for m in self.state.messages]
        if self.state.last_result:
            lines.append(f"Last action result: {json.dumps(self.state.last_result, indent=2)}")
        return "\n".join(lines)

    def parse_response(self, raw: str) -> dict:
        """Parse model response, extracting JSON or wrapping plain text."""
        text = raw.strip()
        
        # Remove markdown code blocks
        if text.startswith("```"):
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
            text = match.group(1).strip() if match else text.strip("`").strip()
        
        # Remove 'json' prefix
        if text.lower().startswith("json"):
            text = text[4:].strip()
        
        # Try parsing as JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try extracting embedded JSON
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        
        # Fallback: wrap as plain answer
        logger.warning("Model returned plain text, wrapping as answer")
        return {"thought": text, "action": None}

    def call_model(self, prompt: str) -> dict:
        """Call the language model and return parsed response."""
        try:
            if self.state.image_path:
                response = TOOL_REGISTRY[ToolName.GEMINI_MULTIMODAL]({
                    "text": prompt,
                    "image_path": self.state.image_path
                })
            else:
                result = generate_content(self.client, self.model, prompt)
                response = str(result.text) if result else ""
            
            if not response or not response.strip():
                return {"error": "Empty response from model"}
            
            logger.info(f"Raw model response: {response[:500]}...")
            return self.parse_response(response)
            
        except Exception as e:
            logger.error(f"Model call failed: {e}")
            return {"error": str(e)}

    def use_tool(self, name: ToolName, query: Union[str, dict]) -> str:
        """Execute a tool and return the result."""
        func = TOOL_REGISTRY.get(name)
        if not func:
            return f"Unknown tool: {name}"
        
        try:
            logger.info(f"Using tool {name} with query: {query}")
            
            # Handle different input formats
            if name == ToolName.GEMINI_MULTIMODAL:
                result = func(q=query if isinstance(query, dict) else {"text": query})
            elif isinstance(query, dict):
                # Extract text from dict if needed
                text = query.get("text", query.get("q", ""))
                result = func(text) if text else func(**query)
            else:
                result = func(query)
            
            logger.info(f"Tool {name} result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Tool {name} failed: {e}")
            return str(e)

    def think(self) -> Optional[dict]:
        """Generate the next thought/action based on current state."""
        self.state.iteration += 1
        
        if self.state.iteration > self.max_iterations:
            self.trace("assistant", "Max iterations reached without satisfactory answer.")
            return None
        
        prompt = self.template.format(
            query=self.state.query,
            image_context=self.state.image_path,
            history=self.get_history(),
            tools=", ".join(t.name for t in TOOL_REGISTRY.keys()),
            last_result=json.dumps(self.state.last_result) if self.state.last_result else "None"
        )
        
        response = self.call_model(prompt)
        
        if "error" in response:
            self.trace("assistant", f"Error: {response['error']}")
            return None
        
        self.trace("assistant", f"Thought: {response}")
        return response

    def act(self, response: dict) -> Optional[str]:
        """Execute action from model response, return final answer if found."""
        if "error" in response:
            self.trace("assistant", f"Error: {response['error']}")
            return None
        
        # Check for final answer first
        if "answer" in response:
            answer = response["answer"]
            if answer and str(answer).strip():
                self.trace("assistant", f"Final Answer: {answer}")
                return answer
        
        # Process action
        action = response.get("action")
        if not action or not isinstance(action, dict):
            return None
        
        name_str = (action.get("name") or "").upper()
        if name_str == "NONE" or not name_str:
            return None
        
        # Resolve tool name
        try:
            tool_name = ToolName[name_str]
        except KeyError:
            logger.error(f"Unknown tool: {name_str}")
            self.trace("assistant", f"Unknown tool '{name_str}', trying different approach.")
            return None
        
        # Prepare input
        action_input = action.get("input", self.state.query)
        if tool_name == ToolName.GEMINI_MULTIMODAL or self.state.image_path:
            if isinstance(action_input, str):
                action_input = {"text": action_input, "image_path": self.state.image_path}
            elif isinstance(action_input, dict) and "image_path" not in action_input:
                action_input["image_path"] = self.state.image_path
        
        self.trace("assistant", f"Action: Using {tool_name.name}")
        
        # Execute tool
        result = self.use_tool(tool_name, action_input)
        self.state.last_result = result
        self.trace("system", f"Observation from {tool_name.name}: {result}")
        
        return None

    def run(self, query: Union[str, Dict[str, Any]]) -> Generator[Dict[str, Any], None, None]:
        """
        Run the agent on a query, yielding state after each iteration.
        
        Args:
            query: String query or dict with 'text' and optional 'image_path'
            
        Yields:
            Dict with iteration number, messages, and completion status
        """
        self.reset()
        
        # Parse query
        if isinstance(query, dict):
            self.state.query = query.get("text", "")
            if query.get("image_path"):
                self.state.image_path = os.path.abspath(query["image_path"])
        else:
            self.state.query = str(query)
        
        # Log initial query
        content = {"text": self.state.query, "image_path": self.state.image_path} if self.state.image_path else self.state.query
        self.trace("user", content)
        
        logger.info(f"Starting agent with query: {self.state.query}")
        
        # Main loop
        final_answer = None
        while final_answer is None and self.state.iteration < self.max_iterations:
            msg_start = len(self.state.messages) - 1
            
            response = self.think()
            if response is None:
                yield {"iteration": self.state.iteration, "messages": [], "done": True}
                return
            
            final_answer = self.act(response)
            
            yield {
                "iteration": self.state.iteration,
                "messages": self.state.messages[msg_start:],
                "done": final_answer is not None
            }
        
        yield {"iteration": self.state.iteration, "messages": [], "done": True}


def run_react_agent(query: Union[str, Dict[str, Any]], max_iterations: int = 5):
    """
    Convenience function to run the ReAct agent.
    
    Args:
        query: String or dict with 'text' and optional 'image_path'
        max_iterations: Maximum thinking iterations
        
    Returns:
        Generator yielding iteration data
    """
    agent = Agent(max_iterations=max_iterations)
    return agent.run(query)


if __name__ == "__main__":
    test_query = {
        "text": "Tell me about the history of the Eiffel Tower and suggest some nearby attractions.",
    }
    
    for data in run_react_agent(test_query, max_iterations=5):
        logger.info(f"Iteration {data['iteration']}:")
        for msg in data["messages"]:
            logger.info(f"  {msg.role}: {msg.content}")
        if data["done"]:
            logger.info("Task completed.")
            break