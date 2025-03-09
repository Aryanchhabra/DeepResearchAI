#!/usr/bin/env python
"""
Flask web application for Deep Research AI.
"""

import os
import sys
import json
import logging
import markdown
import time
import threading
import queue
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, Response, stream_with_context
from flask_cors import CORS
from flask_wtf.csrf import CSRFProtect

# Add the parent directory to the path so we can import from the deep_research package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the research functionality
from deep_research.agents.research_agent import ResearchAgent
from deep_research.agents.answer_agent import AnswerAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('web/web_app.log')
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)
csrf = CSRFProtect(app)
CORS(app)

# Exempt the research endpoint from CSRF protection
csrf.exempt('research')
csrf.exempt('stream_research')

# Store active research tasks
research_tasks = {}

# Add context processor for current date/time
@app.context_processor
def inject_now():
    return {'now': datetime.now()}

# Initialize session storage for research history
if not os.path.exists('web/research_history'):
    os.makedirs('web/research_history')

def get_research_history():
    """Get the research history from the session."""
    if 'history' not in session:
        session['history'] = []
    return session['history']

def save_research_to_history(question, answer, sources):
    """Save the research results to the history."""
    history = get_research_history()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create a unique ID for this research
    research_id = f"{timestamp.replace(' ', '_').replace(':', '-')}_{hash(question) % 10000}"
    
    research_data = {
        "id": research_id,
        "timestamp": timestamp,
        "question": question,
        "answer": answer,
        "sources": sources
    }
    
    # Save to session
    history.append({
        "id": research_id,
        "timestamp": timestamp,
        "question": question
    })
    session['history'] = history
    
    # Save detailed results to file
    with open(f"web/research_history/{research_id}.json", 'w') as f:
        json.dump(research_data, f)
    
    return research_id

def get_research_by_id(research_id):
    """Get the research results by ID."""
    try:
        with open(f"web/research_history/{research_id}.json", 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def run_research_with_updates(question, use_langgraph, task_id, progress_queue):
    """
    Run the research workflow with progress updates.
    
    Args:
        question: The research question
        use_langgraph: Whether to use the LangGraph implementation
        task_id: The ID of the research task
        progress_queue: Queue to send progress updates
    """
    try:
        # Send initial update
        progress_queue.put({
            "status": "in_progress",
            "step": "starting",
            "message": "Starting research process...",
            "progress": 5
        })
        
        if use_langgraph:
            # Import the workflow (only when needed to avoid unnecessary imports)
            from deep_research.workflow import DeepResearchWorkflow
            
            # Initialize the workflow
            progress_queue.put({
                "status": "in_progress",
                "step": "initializing",
                "message": "Initializing LangGraph workflow...",
                "progress": 10
            })
            workflow = DeepResearchWorkflow()
            
            # Run the workflow
            progress_queue.put({
                "status": "in_progress",
                "step": "researching",
                "message": "Conducting research with LangGraph...",
                "progress": 20
            })
            results = workflow.run(question)
        else:
            # Initialize agents
            progress_queue.put({
                "status": "in_progress",
                "step": "initializing",
                "message": "Initializing research agents...",
                "progress": 10
            })
            research_agent = ResearchAgent()
            answer_agent = AnswerAgent()
            
            # Step 1: Conduct research
            progress_queue.put({
                "status": "in_progress",
                "step": "searching",
                "message": "Generating search queries and performing web searches...",
                "progress": 20
            })
            time.sleep(1)  # Give time for the frontend to update
            
            progress_queue.put({
                "status": "in_progress",
                "step": "extracting",
                "message": "Extracting content from web pages...",
                "progress": 40
            })
            research_results = research_agent.conduct_research(question)
            
            # Step 2: Draft answer
            progress_queue.put({
                "status": "in_progress",
                "step": "drafting",
                "message": "Drafting initial answer based on research findings...",
                "progress": 60
            })
            draft_answer = answer_agent.draft_answer(research_results)
            
            # Step 3: Fact check
            progress_queue.put({
                "status": "in_progress",
                "step": "fact_checking",
                "message": "Fact-checking the drafted answer against sources...",
                "progress": 80
            })
            fact_check = answer_agent.fact_check(draft_answer, research_results)
            
            # Step 4: Finalize answer
            progress_queue.put({
                "status": "in_progress",
                "step": "finalizing",
                "message": "Finalizing the answer with citations...",
                "progress": 90
            })
            final_answer = answer_agent.finalize_answer(fact_check)
            
            results = {
                "question": question,
                "answer": final_answer.get("answer", "No answer generated."),
                "sources": final_answer.get("sources", []),
                "status": "success"
            }
        
        # Save to history
        research_id = save_research_to_history(
            question, 
            results.get("answer", "No answer generated."),
            results.get("sources", [])
        )
        results["id"] = research_id
        
        # Send final update
        progress_queue.put({
            "status": "completed",
            "step": "completed",
            "message": "Research completed successfully!",
            "progress": 100,
            "results": results
        })
        
    except Exception as e:
        logger.error(f"Error in research workflow: {str(e)}")
        error_results = {
            "question": question,
            "answer": f"An error occurred during research: {str(e)}",
            "sources": [],
            "status": "error",
            "errors": [str(e)]
        }
        
        # Send error update
        progress_queue.put({
            "status": "error",
            "step": "error",
            "message": f"Error: {str(e)}",
            "progress": 100,
            "results": error_results
        })

def run_research(question, use_langgraph=False):
    """
    Run the research workflow for a given question.
    
    Args:
        question: The research question
        use_langgraph: Whether to use the LangGraph implementation
        
    Returns:
        Dictionary with the research results
    """
    logger.info(f"Starting research for question: {question}")
    
    try:
        if use_langgraph:
            # Import the workflow (only when needed to avoid unnecessary imports)
            from deep_research.workflow import DeepResearchWorkflow
            
            # Initialize the workflow
            workflow = DeepResearchWorkflow()
            
            # Run the workflow
            results = workflow.run(question)
        else:
            # Initialize agents
            research_agent = ResearchAgent()
            answer_agent = AnswerAgent()
            
            # Step 1: Conduct research
            research_results = research_agent.conduct_research(question)
            
            # Step 2: Draft answer
            draft_answer = answer_agent.draft_answer(research_results)
            
            # Step 3: Fact check
            fact_check = answer_agent.fact_check(draft_answer, research_results)
            
            # Step 4: Finalize answer
            final_answer = answer_agent.finalize_answer(fact_check)
            
            results = {
                "question": question,
                "answer": final_answer.get("answer", "No answer generated."),
                "sources": final_answer.get("sources", []),
                "status": "success"
            }
        
        # Save to history
        research_id = save_research_to_history(
            question, 
            results.get("answer", "No answer generated."),
            results.get("sources", [])
        )
        results["id"] = research_id
        
        return results
    except Exception as e:
        logger.error(f"Error in research workflow: {str(e)}")
        return {
            "question": question,
            "answer": f"An error occurred during research: {str(e)}",
            "sources": [],
            "status": "error",
            "errors": [str(e)]
        }

@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html', history=get_research_history())

@app.route('/research', methods=['POST'])
def research():
    """Handle research requests."""
    data = request.form
    question = data.get('question', '')
    use_langgraph = data.get('use_langgraph', 'false').lower() == 'true'
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    results = run_research(question, use_langgraph)
    
    return jsonify(results)

@app.route('/stream_research', methods=['POST'])
def stream_research():
    """Handle streaming research requests."""
    data = request.form
    question = data.get('question', '')
    use_langgraph = data.get('use_langgraph', 'false').lower() == 'true'
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    # Create a unique task ID
    task_id = f"task_{int(time.time())}_{hash(question) % 10000}"
    
    # Create a queue for progress updates
    progress_queue = queue.Queue()
    
    # Store the task
    research_tasks[task_id] = {
        "queue": progress_queue,
        "status": "starting"
    }
    
    # Start the research in a background thread
    thread = threading.Thread(
        target=run_research_with_updates,
        args=(question, use_langgraph, task_id, progress_queue)
    )
    thread.daemon = True
    thread.start()
    
    # Return the task ID
    return jsonify({"task_id": task_id})

@app.route('/research_progress/<task_id>')
def research_progress(task_id):
    """Stream research progress updates."""
    if task_id not in research_tasks:
        return jsonify({"error": "Task not found"}), 404
    
    def generate():
        task = research_tasks[task_id]
        queue = task["queue"]
        
        try:
            # Send an initial heartbeat
            yield f"data: {json.dumps({'status': 'connected'})}\n\n"
            
            while True:
                try:
                    # Get the next update from the queue (with a timeout)
                    try:
                        update = queue.get(block=True, timeout=0.5)
                        
                        # Send the update as a server-sent event
                        yield f"data: {json.dumps(update)}\n\n"
                        
                        # If this is the final update, break the loop
                        if update["status"] in ["completed", "error"]:
                            # Clean up the task
                            research_tasks.pop(task_id, None)
                            break
                            
                    except queue.Empty:
                        # No updates available, send a heartbeat
                        yield f"data: {json.dumps({'status': 'heartbeat'})}\n\n"
                        
                except Exception as e:
                    # Log the error
                    logger.error(f"Error in SSE stream: {str(e)}")
                    # Send an error update
                    yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"
                    break
        except GeneratorExit:
            # Client disconnected
            logger.info(f"Client disconnected from SSE stream for task {task_id}")
            # Clean up the task if it's still there
            if task_id in research_tasks:
                research_tasks.pop(task_id, None)
    
    return Response(
        stream_with_context(generate()),
        content_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'  # Disable buffering in Nginx
        }
    )

@app.route('/history')
def history():
    """Render the history page."""
    return render_template('history.html', history=get_research_history())

@app.route('/research/<research_id>')
def view_research(research_id):
    """View a specific research result."""
    research = get_research_by_id(research_id)
    if research:
        # Convert markdown to HTML
        if research.get("answer"):
            research["answer_html"] = markdown.markdown(research["answer"])
        return render_template('research.html', research=research)
    return redirect(url_for('history'))

@app.route('/api/research/<research_id>')
def api_research(research_id):
    """API endpoint to get research data."""
    research = get_research_by_id(research_id)
    if research:
        return jsonify(research)
    return jsonify({"error": "Research not found"}), 404

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear the research history."""
    session.pop('history', None)
    return redirect(url_for('history'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 