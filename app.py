import os
import gc
import uuid
from flask import Flask, request, jsonify, send_from_directory, send_file
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
from flask_cors import CORS

# Import the existing processor
from kolam_gen import EnhancedKolamProcessor

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app, resources={r"/*": {"origins": "*"}})

# Ensure input/output dirs exist (the processor also does this)
os.makedirs('data/input_images', exist_ok=True)
os.makedirs('data/output_results', exist_ok=True)


@app.route('/')
def index():
    # Serve the UI file
    return send_from_directory(app.static_folder, 'index.html')


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

    # Downscale large images to save memory/CPU
    try:
        with Image.open(save_path) as img:
            img = img.convert('RGB')
            max_side = 1280
            if max(img.size) > max_side:
                img.thumbnail((max_side, max_side))
                img.save(save_path, format='JPEG', quality=90)
    except Exception:
        pass

    # Run analysis with per-request processor to allow GC
    processor = EnhancedKolamProcessor()
    results = processor.analyze_kolam_from_image(save_path)

    # Optionally save visualization and include path
    # Prepare unique IDs/paths for outputs
    uid = uuid.uuid4().hex[:12]
    output_dir = 'data/output_results'
    os.makedirs(output_dir, exist_ok=True)
    viz_out_path = os.path.join(output_dir, f'kolam_analysis_{uid}.png')
    json_out_path = os.path.join(output_dir, f'enhanced_kolam_analysis_{uid}.json')

    # Optional visualization (costly). Enable by /analyze?viz=1
    generate_viz = request.args.get('viz') == '1'
    if generate_viz:
        try:
            fig = processor.create_comprehensive_visualization(save_path=viz_out_path)
            try:
                import matplotlib.pyplot as plt
                plt.close(fig)
            except Exception:
                pass
        except Exception:
            # Visualization is optional; ignore errors
            pass

    # Create JSON export and collect latest artifacts
    latest_png = viz_out_path if generate_viz and os.path.exists(viz_out_path) else ''
    latest_json = ''
    try:
        # Export results JSON to deterministic path for this request
        json_path = processor.export_results(json_out_path)
        latest_json = json_path if json_path else ''
    except Exception:
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
    response = jsonify(_to_serializable(results))

    # Free memory aggressively
    try:
        processor.original_image = None
        processor.processed_image = None
    except Exception:
        pass
    del processor
    gc.collect()

    return response


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
        resp = send_file(target, as_attachment=True)
        # Expose filename header to browsers (for JS-triggered downloads across origins)
        resp.headers['Access-Control-Expose-Headers'] = 'Content-Disposition'
        resp.headers['Cache-Control'] = 'no-store'
        return resp
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


