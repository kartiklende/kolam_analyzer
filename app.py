import os
from flask import Flask, request, jsonify, send_from_directory, send_file
from werkzeug.utils import secure_filename
import numpy as np
from flask_cors import CORS

# Import the existing processor
from kolam_gen import EnhancedKolamProcessor

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app, resources={r"/*": {"origins": "*"}})

# Ensure input/output dirs exist (the processor also does this)
os.makedirs('data/input_images', exist_ok=True)
os.makedirs('data/output_results', exist_ok=True)

processor = EnhancedKolamProcessor()


@app.route('/')
def index():
    # Serve the UI file
    return send_from_directory(app.static_folder, 'ui.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': True, 'error_message': 'No image file part provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': True, 'error_message': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join('data/input_images', filename)
    file.save(save_path)

    # Run analysis
    results = processor.analyze_kolam_from_image(save_path)

    # Optionally save visualization and include path
    try:
        fig = processor.create_comprehensive_visualization()
    except Exception:
        # Visualization is optional; ignore errors
        pass

    # Create JSON export and collect latest artifacts
    output_dir = 'data/output_results'
    latest_png = ''
    latest_json = ''
    try:
        # Export results JSON
        json_path = processor.export_results()
        # Find latest PNG
        pngs = [
            os.path.join(output_dir, f) for f in os.listdir(output_dir)
            if f.lower().endswith('.png')
        ]
        latest_png = max(pngs, key=os.path.getmtime) if pngs else ''
        latest_json = json_path if json_path else ''
    except Exception:
        latest_png = ''
        latest_json = ''

    # Normalize to URL-friendly relative paths
    def to_rel_url_path(path):
        if not path:
            return ''
        rel = os.path.relpath(path, start='.')
        return rel.replace('\\', '/')

    if latest_png or latest_json:
        results['metadata'] = results.get('metadata', {})
        if latest_png:
            results['metadata']['visualization_image'] = to_rel_url_path(latest_png)
        if latest_json:
            results['metadata']['results_json'] = to_rel_url_path(latest_json)

    # Ensure JSON serializable payload
    return jsonify(_to_serializable(results))


@app.route('/health')
def health():
    return 'ok', 200


def _safe_output_path(path_param: str) -> str:
    """Ensure the requested path stays within data/output_results."""
    base = os.path.abspath('data/output_results')
    target = os.path.abspath(os.path.join('.', path_param))
    if not target.startswith(base):
        raise ValueError('Invalid path')
    return target


@app.route('/download')
def download_file():
    path_param = request.args.get('path', '')
    if not path_param:
        return jsonify({'error': True, 'error_message': 'Missing path parameter'}), 400
    try:
        target = _safe_output_path(path_param)
        return send_file(target, as_attachment=True)
    except Exception as e:
        return jsonify({'error': True, 'error_message': str(e)}), 400


def _to_serializable(obj):
    """Recursively convert numpy types/arrays to JSON-serializable Python types."""
    # numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # numpy scalars
    if isinstance(obj, (np.generic,)):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    # dicts
    if isinstance(obj, dict):
        return {str(_to_serializable(k)): _to_serializable(v) for k, v in obj.items()}
    # lists/tuples/sets
    if isinstance(obj, (list, tuple, set)):
        return [_to_serializable(v) for v in obj]
    # other types pass through
    return obj


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)


