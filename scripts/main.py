# scripts/main.py

from rallyrobopilot import prepare_game_app, RemoteController
from flask import Flask
from threading import Thread

# Setup Flask
flask_app = Flask(__name__)

# --- MODIFIED: Add daemon=True ---
# This tells Python to automatically shut down this thread when the main app closes.
flask_thread = Thread(target=flask_app.run, kwargs={'host': "0.0.0.0", 'port': 5000}, daemon=True)

print("Flask server running on port 5000")
flask_thread.start()
        
app, car = prepare_game_app()
remote_controller = RemoteController(car = car, connection_port=7654, flask_app=flask_app)

def input(key):
    if key == 'exit':
        quit()

app.run()