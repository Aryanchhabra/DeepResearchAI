"""
Workflow orchestrator for the Deep Research AI Agentic System.
Uses LangGraph to coordinate the research and answer agents.
"""

import os
import logging
from typing import Dict, List, Any, Literal, TypedDict, Annotated, Union, cast
from dotenv import load_dotenv

# Import LangGraph components
from langgraph.graph import StateGraph, END

# Import agents
from deep_research.agents.research_agent import ResearchAgent
from deep_research.agents.answer_agent import AnswerAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Define the state type
class ResearchState(TypedDict):
    """Type definition for the research state."""
    question: Annotated[str, "question"]
    research_results: Annotated[Dict[str, Any], "research_results"]
    draft_answer: Annotated[Dict[str, Any], "draft_answer"]
    fact_check: Annotated[Dict[str, Any], "fact_check"]
    final_answer: Annotated[Dict[str, Any], "final_answer"]
    current_step: Annotated[str, "current_step"]
    errors: Annotated[List[str], "errors"]

# Define the workflow steps
class WorkflowStep:
    """Workflow step constants."""
    RESEARCH = "research"
    DRAFT_ANSWER = "draft_answer_node"
    FACT_CHECK = "fact_check_node"
    FINALIZE = "finalize"
    END = "end"
    ERROR = "error"

class DeepResearchWorkflow:
    """
    Workflow orchestrator for the Deep Research AI Agentic System.
    Uses LangGraph to coordinate the research and answer agents.
    """
    
    def __init__(self):
        """Initialize the workflow orchestrator."""
        logger.info("Initializing Deep Research Workflow")
        
        # Initialize agents
        self.research_agent = ResearchAgent()
        self.answer_agent = AnswerAgent()
        
        # Create the workflow graph
        self.graph = self._create_workflow_graph()
    
    def _create_workflow_graph(self) -> StateGraph:
        """
        Create the workflow graph.
        
        Returns:
            StateGraph: The configured workflow graph
        """
        # Create a new graph
        graph = StateGraph(ResearchState)
        
        # Add nodes for each step in the workflow
        graph.add_node(WorkflowStep.RESEARCH, self._conduct_research)
        graph.add_node(WorkflowStep.DRAFT_ANSWER, self._draft_answer)
        graph.add_node(WorkflowStep.FACT_CHECK, self._fact_check)
        graph.add_node(WorkflowStep.FINALIZE, self._finalize_answer)
        graph.add_node(WorkflowStep.ERROR, self._handle_error)
        
        # Set the research step as the entry point
        graph.set_entry_point(WorkflowStep.RESEARCH)
        
        # Define the edges
        graph.add_edge(WorkflowStep.RESEARCH, WorkflowStep.DRAFT_ANSWER)
        graph.add_edge(WorkflowStep.DRAFT_ANSWER, WorkflowStep.FACT_CHECK)
        graph.add_edge(WorkflowStep.FACT_CHECK, WorkflowStep.FINALIZE)
        graph.add_edge(WorkflowStep.FINALIZE, END)
        
        # Add conditional edges for error handling
        graph.add_conditional_edges(
            WorkflowStep.ERROR,
            self._determine_next_step_after_error,
            {
                WorkflowStep.RESEARCH: WorkflowStep.RESEARCH,
                WorkflowStep.DRAFT_ANSWER: WorkflowStep.DRAFT_ANSWER,
                WorkflowStep.FACT_CHECK: WorkflowStep.FACT_CHECK,
                WorkflowStep.FINALIZE: WorkflowStep.FINALIZE,
                WorkflowStep.END: END
            }
        )
        
        # Add error edges from each node to the error handler
        graph.add_edge(WorkflowStep.RESEARCH, WorkflowStep.ERROR)
        graph.add_edge(WorkflowStep.DRAFT_ANSWER, WorkflowStep.ERROR)
        graph.add_edge(WorkflowStep.FACT_CHECK, WorkflowStep.ERROR)
        graph.add_edge(WorkflowStep.FINALIZE, WorkflowStep.ERROR)
        
        # Compile the graph
        return graph.compile()
    
    def _conduct_research(self, state: ResearchState) -> ResearchState:
        """
        Conduct research on the question.
        
        Args:
            state: The current research state
            
        Returns:
            Updated research state
        """
        logger.info(f"Conducting research on: {state['question']}")
        
        try:
            # Conduct research
            research_results = self.research_agent.conduct_research(state["question"])
            
            # Update state
            state["research_results"] = research_results
            state["current_step"] = WorkflowStep.DRAFT_ANSWER
            
            return state
        except Exception as e:
            error_msg = f"Error in research step: {str(e)}"
            logger.error(error_msg)
            
            # Create a new state to avoid modifying the original
            new_state = state.copy()
            
            # Update state with error
            if "errors" not in new_state:
                new_state["errors"] = []
            new_state["errors"] = new_state["errors"] + [error_msg]  # Append to the list
            new_state["current_step"] = WorkflowStep.ERROR
            
            return new_state
    
    def _draft_answer(self, state: ResearchState) -> ResearchState:
        """
        Draft an answer based on research results.
        
        Args:
            state: The current research state
            
        Returns:
            Updated research state
        """
        logger.info("Drafting answer")
        
        try:
            # Draft answer
            draft_answer = self.answer_agent.draft_answer(state["research_results"])
            
            # Update state
            state["draft_answer"] = draft_answer
            state["current_step"] = WorkflowStep.FACT_CHECK
            
            return state
        except Exception as e:
            error_msg = f"Error in draft answer step: {str(e)}"
            logger.error(error_msg)
            
            # Create a new state to avoid modifying the original
            new_state = state.copy()
            
            # Update state with error
            if "errors" not in new_state:
                new_state["errors"] = []
            new_state["errors"] = new_state["errors"] + [error_msg]  # Append to the list
            new_state["current_step"] = WorkflowStep.ERROR
            
            return new_state
    
    def _fact_check(self, state: ResearchState) -> ResearchState:
        """
        Fact check the drafted answer.
        
        Args:
            state: The current research state
            
        Returns:
            Updated research state
        """
        logger.info("Fact checking answer")
        
        try:
            # Fact check
            fact_check = self.answer_agent.fact_check(state["draft_answer"], state["research_results"])
            
            # Update state
            state["fact_check"] = fact_check
            state["current_step"] = WorkflowStep.FINALIZE
            
            return state
        except Exception as e:
            error_msg = f"Error in fact check step: {str(e)}"
            logger.error(error_msg)
            
            # Create a new state to avoid modifying the original
            new_state = state.copy()
            
            # Update state with error
            if "errors" not in new_state:
                new_state["errors"] = []
            new_state["errors"] = new_state["errors"] + [error_msg]  # Append to the list
            new_state["current_step"] = WorkflowStep.ERROR
            
            return new_state
    
    def _finalize_answer(self, state: ResearchState) -> ResearchState:
        """
        Finalize the answer based on fact checking.
        
        Args:
            state: The current research state
            
        Returns:
            Updated research state
        """
        logger.info("Finalizing answer")
        
        try:
            # Finalize answer
            final_answer = self.answer_agent.finalize_answer(state["fact_check"])
            
            # Update state
            state["final_answer"] = final_answer
            state["current_step"] = WorkflowStep.END
            
            return state
        except Exception as e:
            error_msg = f"Error in finalize answer step: {str(e)}"
            logger.error(error_msg)
            
            # Create a new state to avoid modifying the original
            new_state = state.copy()
            
            # Update state with error
            if "errors" not in new_state:
                new_state["errors"] = []
            new_state["errors"] = new_state["errors"] + [error_msg]  # Append to the list
            new_state["current_step"] = WorkflowStep.ERROR
            
            return new_state
    
    def _handle_error(self, state: ResearchState) -> ResearchState:
        """
        Handle errors in the workflow.
        
        Args:
            state: The current research state
            
        Returns:
            Updated research state
        """
        logger.info("Handling error")
        
        # Create a new state to avoid modifying the original
        new_state = state.copy()
        
        # Log all errors
        if "errors" in new_state and new_state["errors"]:
            for error in new_state["errors"]:
                logger.error(f"Workflow error: {error}")
        
        # Determine the next step based on the current step
        current_step = new_state.get("current_step", WorkflowStep.RESEARCH)
        
        # For now, we'll just continue to the next step
        # In a more sophisticated system, we might retry or take alternative actions
        
        return new_state
    
    def _check_for_errors(self, state: ResearchState) -> bool:
        """
        Check if there are errors in the state.
        
        Args:
            state: The current research state
            
        Returns:
            True if there are errors, False otherwise
        """
        return "errors" in state and len(state["errors"]) > 0
    
    def _determine_next_step_after_error(self, state: ResearchState) -> str:
        """
        Determine the next step after an error.
        
        Args:
            state: The current research state
            
        Returns:
            The next step to take
        """
        # Get the current step
        current_step = state.get("current_step", WorkflowStep.RESEARCH)
        
        # For simplicity, we'll just continue to the next step
        # In a more sophisticated system, we might retry or take alternative actions
        step_sequence = [
            WorkflowStep.RESEARCH,
            WorkflowStep.DRAFT_ANSWER,
            WorkflowStep.FACT_CHECK,
            WorkflowStep.FINALIZE,
            WorkflowStep.END
        ]
        
        try:
            current_index = step_sequence.index(current_step)
            next_index = current_index + 1
            
            if next_index < len(step_sequence):
                return step_sequence[next_index]
            else:
                return WorkflowStep.END
        except ValueError:
            # If the current step is not in the sequence, start from the beginning
            return WorkflowStep.RESEARCH
    
    def run(self, question: str) -> Dict[str, Any]:
        """
        Run the research workflow for a given question.
        
        Args:
            question: The research question
            
        Returns:
            Dictionary with the research results
        """
        logger.info(f"Starting research workflow for question: {question}")
        
        # Initialize the state
        state: ResearchState = {
            "question": question,
            "research_results": {},
            "draft_answer": {},
            "fact_check": {},
            "final_answer": {},
            "current_step": WorkflowStep.RESEARCH,
            "errors": []
        }
        
        try:
            # Run the graph
            final_state = None
            
            # Stream the execution and capture the final state
            for output in self.graph.stream(state):
                # Just log the events but don't try to access event-specific fields
                logger.info(f"Processing workflow step")
                
                # Update the final state with each iteration
                if "state" in output:
                    final_state = output["state"]
            
            # If we didn't get a final state, use the initial state
            if not final_state:
                final_state = state
            
            # Extract the final answer
            if "final_answer" in final_state and final_state["final_answer"]:
                return {
                    "question": question,
                    "answer": final_state["final_answer"].get("answer", "No answer generated."),
                    "sources": final_state["final_answer"].get("sources", []),
                    "status": "success"
                }
            elif "draft_answer" in final_state and final_state["draft_answer"]:
                return {
                    "question": question,
                    "answer": final_state["draft_answer"].get("answer", "No answer generated."),
                    "sources": final_state["draft_answer"].get("sources", []),
                    "status": "partial"
                }
            else:
                return {
                    "question": question,
                    "answer": "Failed to generate an answer.",
                    "sources": [],
                    "status": "failed",
                    "errors": final_state.get("errors", [])
                }
                
        except Exception as e:
            logger.error(f"Error in research workflow: {str(e)}")
            return {
                "question": question,
                "answer": f"An error occurred during research: {str(e)}",
                "sources": [],
                "status": "error",
                "errors": [str(e)]
            } 