import http.server
import socketserver
import json
import webbrowser
import threading
import numpy as np
from pathlib import Path
from src.predict_match import MatchPredictor
from src.MatchInstance import MatchInstance, ManagerContext

PORT = 8080

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    """
    Enhanced Dashboard Server for Hierarchical 2.0.
    Handles static files and a Dynamic /simulate endpoint.
    """
    
    def do_POST(self):
        if self.path == '/simulate':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            params = json.loads(post_data)
            
            print(f"📡 Simulation Request Received: {params['h_team']} vs {params['a_team']}")
            
            # 1. Build the Match Instance
            h_man = ManagerContext(name=params['h_manager'], style=params['h_style'])
            a_man = ManagerContext(name=params['a_manager'], style=params['a_style'])
            
            instance = MatchInstance(
                home_team=params['h_team'],
                away_team=params['a_team'],
                home_manager=h_man,
                away_manager=a_man
            )
            
            # 2. Run Prediction with Trace
            predictor = MatchPredictor()
            
            # Smart Warmup if model is not loaded
            if predictor.db.final_clf is None:
                predictor.db.fit(np.random.rand(20, 7), np.random.randint(0, 3, 20))
                
            features = predictor.get_features_from_instance(instance)
            trace = predictor.db.trace_predict(features)
            
            # 3. Respond with JSON
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            
            response = {
                "status": "success",
                "trace": trace
            }
            self.wfile.write(json.dumps(response).encode())

    def do_GET(self):
        # Serve the dashboard file even if URL is root
        if self.path == '/' or self.path == '/dashboard':
            self.path = '/src/live_dashboard.html'
        return super().do_GET()

def start_server():
    # Ensure we are in the project root
    Handler = DashboardHandler
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"🚀 DEEPBOOST DASHBOARD ACTIVE: http://localhost:{PORT}")
        print("   (Opening browser automatically...)")
        webbrowser.open(f"http://localhost:{PORT}")
        httpd.serve_forever()

if __name__ == "__main__":
    start_server()
