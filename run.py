"""
run.py
------
Development entry point for the ORION Flask application.

For production, serve with:
    gunicorn -w 4 -b 0.0.0.0:8000 "run:create_app()"
"""

from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
