#!/usr/bin/env python
"""
Run the Deep Research AI web application.
"""

import os
import sys
from web.app import app

if __name__ == '__main__':
    # Create research history directory if it doesn't exist
    if not os.path.exists('web/research_history'):
        os.makedirs('web/research_history')
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000) 