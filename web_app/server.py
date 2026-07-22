import os
import sys
import json
import io
import base64
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler, ThreadingHTTPServer

# Ensure UTF-8 output encoding on Windows terminals to prevent UnicodeEncodeError
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
        sys.stderr.reconfigure(encoding='utf-8', errors='ignore')
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from web_app.inference import engine
from rank_experiments import rank_experiments

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

def parse_multipart(body_bytes, boundary_bytes):
    parts = body_bytes.split(b'--' + boundary_bytes)
    form_data = {}
    file_bytes = None

    for part in parts:
        if not part or part == b'--\r\n' or part == b'--':
            continue
        if b'\r\n\r\n' in part:
            header_part, content = part.split(b'\r\n\r\n', 1)
            if content.endswith(b'\r\n'):
                content = content[:-2]
            
            header_str = header_part.decode('utf-8', errors='ignore')
            if 'name="model"' in header_str:
                form_data['model'] = content.decode('utf-8', errors='ignore').strip()
            elif 'name="image"' in header_str or 'filename="' in header_str:
                file_bytes = content

    return form_data, file_bytes

class WebUIRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path

        if path == "/api/models":
            self.send_json_response({
                "models": [
                    {"id": "lora", "name": "LoRA (Baseline)", "desc": "Broad adaptation on all non-depthwise Conv2d layers"},
                    {"id": "qlora", "name": "CNN-QLoRA (Quantized Q-Path)", "desc": "INT8 weight-only on MBConv 1x1 project/expand convs"},
                    {"id": "qklora", "name": "CNN-QKLoRA (Quantized Q + FP32 K)", "desc": "Selective INT8 Q-Path (r=16) + FP32 SE/K-Path (r=4)"}
                ]
            })
            return

        elif path == "/api/experiments":
            try:
                rank_experiments()
            except Exception as e:
                print("Error running rank_experiments:", e)

            csv_path = os.path.join(PROJECT_ROOT, "experiments", "results", "eval", "cross_method_ranking.csv")
            data = []
            if os.path.exists(csv_path):
                import pandas as pd
                try:
                    df = pd.read_csv(csv_path)
                    data = df.to_dict(orient="records")
                except Exception as e:
                    print("Error reading cross_method_ranking.csv:", e)
            self.send_json_response({"experiments": data})
            return

        elif path == "/api/plantdoc":
            csv_path = os.path.join(PROJECT_ROOT, "experiments", "results", "eval", "plantdoc_dual_split_results.csv")
            data = []
            if os.path.exists(csv_path):
                import pandas as pd
                try:
                    df = pd.read_csv(csv_path)
                    data = df.to_dict(orient="records")
                except Exception as e:
                    print("Error reading plantdoc_dual_split_results.csv:", e)
            self.send_json_response({"field_results": data})
            return

        elif path == "/api/checkpoints":
            eval_dir = os.path.join(PROJECT_ROOT, "experiments", "results", "eval")
            checkpoints_dir = os.path.join(PROJECT_ROOT, "experiments", "results", "checkpoints")
            checkpoints_data = []
            seen_checkpoints = set()

            if os.path.exists(eval_dir):
                import pandas as pd
                for f in os.listdir(eval_dir):
                    if f.endswith("_checkpoint_ranking.csv"):
                        try:
                            df = pd.read_csv(os.path.join(eval_dir, f))
                            method = f.replace("_checkpoint_ranking.csv", "").upper()
                            records = df.to_dict(orient="records")
                            for r in records:
                                r["method"] = method
                                chk_name = os.path.basename(str(r.get("checkpoint", "")))
                                seen_checkpoints.add(chk_name)
                            checkpoints_data.extend(records)
                        except Exception:
                            pass

            # Fallback: scan checkpoints_dir for any missing .pth files (e.g. QLoRA or QKLoRA)
            if os.path.exists(checkpoints_dir):
                for f in sorted(os.listdir(checkpoints_dir)):
                    if f.endswith(".pth") and f not in seen_checkpoints:
                        method_prefix = f.split("_")[0].upper()
                        size_mb = round(os.path.getsize(os.path.join(checkpoints_dir, f)) / (1024 * 1024), 2)
                        
                        # Check if overall summary has test accuracy for best checkpoint
                        acc = None
                        f1 = None
                        if "best" in f:
                            csv_summary = os.path.join(PROJECT_ROOT, "experiments", "results", "experiment_results.csv")
                            if os.path.exists(csv_summary):
                                import pandas as pd
                                try:
                                    sdf = pd.read_csv(csv_summary)
                                    row = sdf[sdf["experiment"].str.lower() == method_prefix.lower()]
                                    if not row.empty:
                                        acc = float(row.iloc[0].get("test_accuracy", 0))
                                        f1 = float(row.iloc[0].get("test_f1_macro", 0))
                                except Exception:
                                    pass

                        checkpoints_data.append({
                            "method": method_prefix,
                            "checkpoint": f,
                            "size_mb": size_mb,
                            "accuracy": acc,
                            "f1_macro": f1,
                            "binary_accuracy": None,
                            "binary_f1": None,
                            "binary_roc_auc": None,
                            "both_correct_pct": None,
                            "name_only_correct_pct": None
                        })

            self.send_json_response({"checkpoints": checkpoints_data})
            return

        elif path.startswith("/plots/"):
            plot_name = os.path.basename(path)
            plot_path = os.path.join(PROJECT_ROOT, "experiments", "results", "plots", plot_name)
            if os.path.exists(plot_path):
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.end_headers()
                with open(plot_path, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                self.send_error(404, "Plot image not found")
                return

        return super().do_GET()

    def do_POST(self):
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path

        if path == "/api/classify":
            content_type = self.headers.get('Content-Type', '')
            content_len = int(self.headers.get('Content-Length', 0))
            post_body = self.rfile.read(content_len)

            model_key = 'lora'
            image_bytes = None

            if 'multipart/form-data' in content_type:
                boundary = content_type.split("boundary=")[1].encode('utf-8')
                form_data, image_bytes = parse_multipart(post_body, boundary)
                if 'model' in form_data:
                    model_key = form_data['model']
            elif 'application/json' in content_type:
                body_json = json.loads(post_body.decode('utf-8'))
                model_key = body_json.get('model', 'lora')
                if 'image_base64' in body_json:
                    b64_data = body_json['image_base64']
                    if ',' in b64_data:
                        b64_data = b64_data.split(',')[1]
                    image_bytes = base64.b64decode(b64_data)

            if not image_bytes:
                self.send_json_response({"error": "No image payload provided"}, status=400)
                return

            try:
                res = engine.predict(image_bytes, model_key=model_key)
                self.send_json_response(res)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_json_response({"error": str(e)}, status=500)
            return

        self.send_error(404, "Endpoint not found")

    def send_json_response(self, data, status=200):
        body = json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

def start_server(port=8000):
    server_address = ('', port)
    httpd = ThreadingHTTPServer(server_address, WebUIRequestHandler)
    print("============================================================")
    print("Leaf Disease Classification Web UI Server Running!")
    print(f"Local Web Server: http://localhost:{port}")
    print("============================================================")
    httpd.serve_forever()

if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    start_server(port)
