"""
AWS Lambda Handler for MLOps FastAPI Application
"""

from mangum import Mangum
from api.app import app, load_models

# Load models at cold start (module level - runs once per Lambda container)
load_models()

handler = Mangum(app, lifespan="off")
