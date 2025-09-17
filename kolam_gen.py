"""
Enhanced Kolam Processor - Fixed Version for Colored Kolams
===========================================================
Handles colored patterns and provides better dot detection
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import json
import math
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Optional
import warnings
import os
from datetime import datetime

# Image processing imports
from PIL import Image, ImageEnhance
from skimage import io, color, filters, morphology, measure, segmentation
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.morphology import skeletonize, disk
from sklearn.cluster import DBSCAN
import scipy.ndimage as ndi

warnings.filterwarnings('ignore')

class EnhancedKolamProcessor:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.analysis_results = {}
        self.pattern_graph = nx.Graph()
        self.original_image = None
        self.processed_image = None

        os.makedirs('data/input_images', exist_ok=True)
        os.makedirs('data/output_results', exist_ok=True)

        print("🎯 Enhanced Kolam Processor (Fixed Version) initialized!")
        print("📷 Improved colored Kolam processing capabilities")
        print("📁 Input folder: data/input_images/")
        print("📁 Output folder: data/output_results/")

    def _default_config(self) -> Dict:
        return {
            'image_processing': {
                'blur_sigma': 1.5,
                'threshold_method': 'adaptive',  # Better for colored images
                'manual_threshold': 128,
                'morph_kernel_size': 2,
                'min_dot_area': 10,
                'max_dot_area': 200,
                'dot_circularity_threshold': 0.5
            },
            'blob_detection': {
                'method': 'log',
                'min_sigma': 1,
                'max_sigma': 8,
                'num_sigma': 15,
                'threshold': 0.05,  # Lower threshold for better detection
                'overlap': 0.5
            },
            'line_processing': {
                'skeletonize_method': 'zhang',
                'min_line_length': 5,
                'line_gap_threshold': 10
            },
            'graph_analysis': {
                'connection_threshold': 100,  # Increased for colored patterns
                'clustering_eps': 40,
                'min_cluster_samples': 2
            },
            'pattern_reconstruction': {
                'enable_synthetic_dots': True,  # Add synthetic dots if none detected
                'grid_inference': True,  # Try to infer grid from pattern structure
                'symmetry_completion': True  # Complete pattern using symmetry
            }
        }

    def analyze_kolam_from_image(self, image_path: str) -> Dict:
        print(f"\n🎯 Analyzing Kolam from image: {os.path.basename(image_path)}")
        print("=" * 60)

        try:
            # Step 1: Load and preprocess image
            print("🔬 Stage 1: Image Analysis Engine")
            self._load_and_preprocess_image(image_path)

            # Step 2: Extract dots and lines
            print("🔍 Stage 2: Feature Extraction")
            features = self._extract_real_features()

            # Step 3: Handle empty detection case
            if len(features.get('dots', [])) == 0:
                print("⚠️  No dots detected - trying alternative methods...")
                features = self._alternative_feature_extraction()

            # Step 4: Build graph representation
            print("📊 Stage 3: Graph Construction")
            graph = self._build_graph_from_features(features)

            # Step 5: Handle empty graph case
            if graph.number_of_nodes() == 0:
                print("⚠️  Empty graph - creating synthetic structure...")
                graph, features = self._create_synthetic_structure(features)

            # Step 6: Analyze patterns
            print("🧠 Stage 4: Pattern Analysis")
            principles = self._analyze_extracted_principles(graph, features)

            # Step 7: Generate recreation
            print("🎨 Stage 5: Recreation Generation")
            recreation = self._generate_recreation_from_analysis(principles, features)

            # Compile results
            self.analysis_results = {
                'input_info': {
                    'image_path': image_path,
                    'image_size': self.original_image.shape if self.original_image is not None else None,
                    'processing_method': 'enhanced_colored_analysis'
                },
                'extracted_features': features,
                'graph_data': self._safe_graph_to_dict(graph),
                'design_principles': principles,
                'recreation_data': recreation,
                'metadata': {
                    'analysis_timestamp': datetime.now().isoformat(),
                    'processor_version': '2.1.0_colored_fixed',
                    'configuration_used': self.config
                }
            }

            print("✅ Enhanced image analysis completed successfully!")
            return self.analysis_results

        except Exception as e:
            print(f"❌ Error during image analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return partial results if possible
            return self._create_error_result(str(e))

    def _load_and_preprocess_image(self, image_path: str):
        print("   📷 Loading image...")

        # Load image
        self.original_image = io.imread(image_path)

        # Handle different image types
        if len(self.original_image.shape) == 3:
            # For colored images like your Kolam, convert more carefully
            print("   🎨 Processing colored image...")

            # Try different color space conversions
            gray_image = self._smart_color_conversion(self.original_image)
        else:
            gray_image = self.original_image

        print(f"   📐 Image size: {gray_image.shape}")

        # Enhanced preprocessing for colored Kolams
        processed = self._enhanced_preprocessing(gray_image)
        self.processed_image = processed

        print("   ✅ Enhanced preprocessing completed")

    def _smart_color_conversion(self, color_image: np.ndarray) -> np.ndarray:
        """Smart color conversion that preserves Kolam structure"""
        # Method 1: Standard grayscale
        gray1 = color.rgb2gray(color_image)

        # Method 2: Focus on the white lines (common in Kolams)
        # Extract the brightness channel
        hsv = color.rgb2hsv(color_image)
        value_channel = hsv[:, :, 2]  # V channel (brightness)

        # Method 3: Create a mask for white/light areas
        white_mask = np.all(color_image > 200, axis=2)  # Very bright pixels

        # Combine methods - use value channel but enhance white areas
        result = value_channel.copy()
        result[white_mask] = 1.0  # Make white areas very bright

        return result

    def _enhanced_preprocessing(self, gray_image: np.ndarray) -> np.ndarray:
        """Enhanced preprocessing for colored Kolam patterns"""
        # Step 1: Enhance contrast
        from skimage import exposure
        enhanced = exposure.equalize_adapthist(gray_image, clip_limit=0.03)

        # Step 2: Apply gentle blur
        sigma = self.config['image_processing']['blur_sigma']
        blurred = filters.gaussian(enhanced, sigma=sigma)

        # Step 3: Multiple thresholding approaches
        # Try Otsu first
        try:
            otsu_thresh = filters.threshold_otsu(blurred)
            binary1 = blurred > otsu_thresh
        except:
            binary1 = blurred > 0.5

        # Try adaptive threshold
        try:
            adaptive_thresh = filters.threshold_local(blurred, block_size=51, offset=0.02)
            binary2 = blurred > adaptive_thresh
        except:
            binary2 = binary1

        # Combine both approaches
        combined = binary1 | binary2

        # Clean up
        kernel = disk(2)
        cleaned = morphology.opening(combined, kernel)
        cleaned = morphology.closing(cleaned, kernel)

        return cleaned.astype(np.uint8)

    def _extract_real_features(self) -> Dict:
        print("   🎯 Detecting dots with improved algorithm...")
        dots = self._improved_dot_detection()

        print("   📏 Tracing lines with better method...")
        lines, skeleton = self._improved_line_tracing()

        print("   📊 Analyzing structure...")
        spatial_info = self._enhanced_spatial_analysis(dots, lines)

        features = {
            'dots': dots,
            'lines': lines,
            'skeleton': skeleton,
            'spatial_info': spatial_info,
            'image_shape': self.processed_image.shape,
            'extraction_method': 'improved_cv'
        }

        print(f"   ✅ Improved extraction: {len(dots)} dots, {len(lines)} line segments")
        return features

    def _improved_dot_detection(self) -> List[Tuple[float, float]]:
        """Improved dot detection for colored Kolams"""
        dots = []

        # Method 1: Traditional blob detection with adjusted parameters
        try:
            # Invert for blob detection
            inverted = 1 - self.processed_image

            # Multiple blob detection methods
            blobs_log = blob_log(inverted,
                               min_sigma=self.config['blob_detection']['min_sigma'],
                               max_sigma=self.config['blob_detection']['max_sigma'],
                               num_sigma=self.config['blob_detection']['num_sigma'],
                               threshold=self.config['blob_detection']['threshold'])

            for blob in blobs_log:
                dots.append((float(blob[1]), float(blob[0])))

        except Exception as e:
            print(f"   ⚠️ Blob detection failed: {e}")

        # Method 2: Find small circular regions
        try:
            # Label connected components
            labeled = measure.label(self.processed_image)
            regions = measure.regionprops(labeled)

            for region in regions:
                # Check if region looks like a dot
                area = region.area
                if (self.config['image_processing']['min_dot_area'] <= area <=
                    self.config['image_processing']['max_dot_area']):

                    # Check circularity
                    perimeter = region.perimeter
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity >= self.config['image_processing']['dot_circularity_threshold']:
                            centroid = region.centroid
                            dots.append((float(centroid[1]), float(centroid[0])))

        except Exception as e:
            print(f"   ⚠️ Region-based detection failed: {e}")

        # Method 3: Corner detection as potential dots
        try:
            from skimage.feature import corner_harris, corner_peaks

            corners = corner_harris(self.processed_image.astype(float))
            corner_peaks_coords = corner_peaks(corners, min_distance=20, threshold_rel=0.1)

            for coord in corner_peaks_coords:
                dots.append((float(coord[1]), float(coord[0])))

        except Exception as e:
            print(f"   ⚠️ Corner detection failed: {e}")

        # Remove duplicates
        if dots:
            dots = self._remove_duplicate_dots(dots, min_distance=15)

        return dots

    def _remove_duplicate_dots(self, dots: List[Tuple[float, float]], min_distance: float) -> List[Tuple[float, float]]:
        """Remove duplicate dots that are too close to each other"""
        if not dots:
            return dots

        unique_dots = []
        for dot in dots:
            is_duplicate = False
            for existing_dot in unique_dots:
                distance = math.sqrt((dot[0] - existing_dot[0])**2 + (dot[1] - existing_dot[1])**2)
                if distance < min_distance:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_dots.append(dot)

        return unique_dots

    def _improved_line_tracing(self) -> Tuple[List[List[Tuple[float, float]]], np.ndarray]:
        """Improved line tracing"""
        try:
            # Create skeleton
            skeleton = skeletonize(self.processed_image.astype(bool))

            # Extract line segments
            line_segments = self._extract_line_segments_from_skeleton(skeleton)

            return line_segments, skeleton

        except Exception as e:
            print(f"   ⚠️ Line tracing failed: {e}")
            return [], np.zeros_like(self.processed_image)

    def _extract_line_segments_from_skeleton(self, skeleton: np.ndarray) -> List[List[Tuple[float, float]]]:
        """Extract line segments from skeleton"""
        try:
            labeled_skeleton = measure.label(skeleton)
            line_segments = []

            for region in measure.regionprops(labeled_skeleton):
                coords = region.coords
                if len(coords) >= self.config['line_processing']['min_line_length']:
                    line_points = [(float(coord[1]), float(coord[0])) for coord in coords]
                    line_segments.append(line_points)

            return line_segments

        except Exception as e:
            print(f"   ⚠️ Skeleton processing failed: {e}")
            return []

    def _enhanced_spatial_analysis(self, dots: List[Tuple[float, float]],
                                 lines: List[List[Tuple[float, float]]]) -> Dict:
        """Enhanced spatial analysis"""
        analysis = {
            'grid_detected': False,
            'grid_dimensions': (0, 0),
            'pattern_type': 'unknown'
        }

        if not dots and not lines:
            return analysis

        # If we have dots, analyze them
        if dots:
            grid_info = self._detect_grid_structure(dots)
            analysis.update(grid_info)

        # Analyze the overall pattern structure
        if lines:
            pattern_info = self._analyze_pattern_structure(lines)
            analysis.update(pattern_info)

        return analysis

    def _detect_grid_structure(self, dots: List[Tuple[float, float]]) -> Dict:
        """Detect grid structure with improved algorithm"""
        if len(dots) < 4:
            return {'grid_detected': False, 'grid_dimensions': (0, 0)}

        try:
            dot_array = np.array(dots)

            # Use DBSCAN clustering
            x_coords = dot_array[:, 0].reshape(-1, 1)
            y_coords = dot_array[:, 1].reshape(-1, 1)

            eps = self.config['graph_analysis']['clustering_eps']
            min_samples = self.config['graph_analysis']['min_cluster_samples']

            x_clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(x_coords)
            y_clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(y_coords)

            n_cols = len(set(x_clustering.labels_)) - (1 if -1 in x_clustering.labels_ else 0)
            n_rows = len(set(y_clustering.labels_)) - (1 if -1 in y_clustering.labels_ else 0)

            is_grid = n_rows >= 2 and n_cols >= 2

            return {
                'grid_detected': is_grid,
                'grid_dimensions': (n_rows, n_cols) if is_grid else (0, 0),
                'clustering_successful': True
            }

        except Exception as e:
            print(f"   ⚠️ Grid detection failed: {e}")
            return {'grid_detected': False, 'grid_dimensions': (0, 0)}

    def _analyze_pattern_structure(self, lines: List[List[Tuple[float, float]]]) -> Dict:
        """Analyze overall pattern structure"""
        if not lines:
            return {'pattern_type': 'unknown'}

        # Count total line segments
        total_segments = len(lines)
        total_points = sum(len(line) for line in lines)

        # Simple pattern classification
        if total_segments > 20:
            pattern_type = 'complex'
        elif total_segments > 10:
            pattern_type = 'medium'
        else:
            pattern_type = 'simple'

        return {
            'pattern_type': pattern_type,
            'total_line_segments': total_segments,
            'total_line_points': total_points
        }

    def _alternative_feature_extraction(self) -> Dict:
        """Alternative feature extraction when primary method fails"""
        print("   🔄 Trying alternative feature extraction...")

        # Create synthetic dots based on pattern structure
        synthetic_dots = self._create_synthetic_dots()

        # Re-extract lines with different parameters
        lines, skeleton = self._improved_line_tracing()

        # Analyze spatial relationships
        spatial_info = self._enhanced_spatial_analysis(synthetic_dots, lines)

        return {
            'dots': synthetic_dots,
            'lines': lines,
            'skeleton': skeleton,
            'spatial_info': spatial_info,
            'image_shape': self.processed_image.shape,
            'extraction_method': 'alternative_synthetic'
        }

    def _create_synthetic_dots(self) -> List[Tuple[float, float]]:
        """Create synthetic dots based on image structure"""
        if not self.config['pattern_reconstruction']['enable_synthetic_dots']:
            return []

        synthetic_dots = []

        try:
            # Method 1: Based on image intersections
            h, w = self.processed_image.shape

            # Create a grid of potential dot positions
            step_x = w // 8  # Rough grid
            step_y = h // 8

            for i in range(1, 8):
                for j in range(1, 8):
                    x = j * step_x
                    y = i * step_y

                    # Check if this location has some pattern activity
                    if self._has_pattern_activity(y, x):
                        synthetic_dots.append((float(x), float(y)))

        except Exception as e:
            print(f"   ⚠️ Synthetic dot creation failed: {e}")

        return synthetic_dots

    def _has_pattern_activity(self, y: int, x: int, radius: int = 20) -> bool:
        """Check if there's pattern activity around a point"""
        try:
            h, w = self.processed_image.shape

            # Define region around point
            y_min = max(0, y - radius)
            y_max = min(h, y + radius)
            x_min = max(0, x - radius)
            x_max = min(w, x + radius)

            region = self.processed_image[y_min:y_max, x_min:x_max]

            # Check if there's sufficient pattern density
            pattern_density = np.sum(region) / region.size

            return pattern_density > 0.1  # Threshold for pattern activity

        except:
            return False

    def _create_synthetic_structure(self, features: Dict) -> Tuple[nx.Graph, Dict]:
        """Create synthetic structure when no dots are detected"""
        print("   🔧 Creating synthetic structure...")

        graph = nx.Graph()

        # Create a simple grid structure based on image
        h, w = features.get('image_shape', (100, 100))

        # Create a 5x5 synthetic grid
        grid_size = 5
        spacing_x = w // (grid_size + 1)
        spacing_y = h // (grid_size + 1)

        synthetic_dots = []
        for i in range(grid_size):
            for j in range(grid_size):
                x = (j + 1) * spacing_x
                y = (i + 1) * spacing_y
                synthetic_dots.append((float(x), float(y)))

                # Add node to graph
                node_id = i * grid_size + j
                graph.add_node(node_id, pos=(x, y), type='synthetic_dot')

        # Add edges to create basic grid connectivity
        for i in range(grid_size):
            for j in range(grid_size):
                node_id = i * grid_size + j

                # Connect to right neighbor
                if j < grid_size - 1:
                    right_neighbor = i * grid_size + (j + 1)
                    graph.add_edge(node_id, right_neighbor, connection_type='synthetic')

                # Connect to bottom neighbor
                if i < grid_size - 1:
                    bottom_neighbor = (i + 1) * grid_size + j
                    graph.add_edge(node_id, bottom_neighbor, connection_type='synthetic')

        # Update features
        updated_features = features.copy()
        updated_features['dots'] = synthetic_dots
        updated_features['extraction_method'] = 'synthetic_fallback'

        print(f"   ✅ Created synthetic structure: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

        return graph, updated_features

    def _build_graph_from_features(self, features: Dict) -> nx.Graph:
        """Build graph with better error handling"""
        print("   📊 Creating graph representation...")

        graph = nx.Graph()
        dots = features.get('dots', [])
        lines = features.get('lines', [])

        if not dots:
            print("   ⚠️ No dots found - graph will be empty")
            return graph

        # Add nodes for dots
        for i, (x, y) in enumerate(dots):
            graph.add_node(i, pos=(x, y), type='dot')

        # Add edges based on lines or proximity
        if lines:
            self._add_edges_from_lines(graph, lines, dots)
        else:
            self._add_edges_from_proximity(graph, dots)

        self.pattern_graph = graph
        print(f"   ✅ Graph created: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

        return graph

    def _add_edges_from_lines(self, graph: nx.Graph, lines: List, dots: List):
        """Add edges based on detected lines"""
        connection_threshold = self.config['graph_analysis']['connection_threshold']

        for line in lines:
            if len(line) < 2:
                continue

            start_dots = self._find_nearby_dots(line[0], dots, connection_threshold)
            end_dots = self._find_nearby_dots(line[-1], dots, connection_threshold)

            for start_dot in start_dots:
                for end_dot in end_dots:
                    if start_dot != end_dot:
                        graph.add_edge(start_dot, end_dot, connection_type='detected')

    def _add_edges_from_proximity(self, graph: nx.Graph, dots: List):
        """Add edges based on proximity"""
        threshold = self.config['graph_analysis']['connection_threshold']

        for i, dot1 in enumerate(dots):
            for j, dot2 in enumerate(dots):
                if i < j:
                    dist = math.sqrt((dot1[0] - dot2[0])**2 + (dot1[1] - dot2[1])**2)
                    if dist < threshold:
                        graph.add_edge(i, j, weight=dist, connection_type='proximity')

    def _find_nearby_dots(self, point: Tuple[float, float], dots: List[Tuple[float, float]],
                         threshold: float) -> List[int]:
        """Find dots near a point"""
        nearby = []
        for i, dot in enumerate(dots):
            distance = math.sqrt((point[0] - dot[0])**2 + (point[1] - dot[1])**2)
            if distance < threshold:
                nearby.append(i)
        return nearby

    def _analyze_extracted_principles(self, graph: nx.Graph, features: Dict) -> Dict:
        """Analyze principles with proper error handling"""
        print("   🧠 Analyzing design principles...")

        principles = {
            'basic_properties': self._safe_analyze_basic_properties(graph),
            'symmetries': self._detect_image_symmetries(graph, features),
            'topology': self._safe_analyze_topology(graph),
            'mathematical_features': self._safe_calculate_mathematical_features(graph),
            'spatial_properties': features.get('spatial_info', {}),
            'pattern_classification': self._classify_extracted_pattern(graph, features)
        }

        return principles

    def _safe_analyze_basic_properties(self, graph: nx.Graph) -> Dict:
        """Safely analyze basic properties"""
        node_count = graph.number_of_nodes()
        edge_count = graph.number_of_edges()

        return {
            'node_count': node_count,
            'edge_count': edge_count,
            'density': nx.density(graph) if node_count > 0 else 0,
            'average_degree': (2 * edge_count / node_count) if node_count > 0 else 0
        }

    def _safe_analyze_topology(self, graph: nx.Graph) -> Dict:
        """Safely analyze topology"""
        if graph.number_of_nodes() == 0:
            return {
                'is_connected': False,
                'number_of_components': 0,
                'clustering_coefficient': 0
            }

        try:
            return {
                'is_connected': nx.is_connected(graph),
                'number_of_components': nx.number_connected_components(graph),
                'clustering_coefficient': nx.average_clustering(graph)
            }
        except Exception as e:
            return {
                'is_connected': False,
                'number_of_components': 0,
                'clustering_coefficient': 0,
                'error': str(e)
            }

    def _safe_calculate_mathematical_features(self, graph: nx.Graph) -> Dict:
        """Safely calculate mathematical features"""
        if graph.number_of_nodes() == 0:
            return {'cycle_count': 0}

        try:
            cycle_basis = nx.cycle_basis(graph)
            return {
                'cycle_count': len(cycle_basis),
                'cycle_lengths': [len(cycle) for cycle in cycle_basis]
            }
        except Exception as e:
            return {
                'cycle_count': 0,
                'error': str(e)
            }

    def _detect_image_symmetries(self, graph: nx.Graph, features: Dict) -> Dict:
        """Detect symmetries from extracted features"""
        symmetries = {}

        # Basic symmetry detection based on pattern structure
        spatial_info = features.get('spatial_info', {})

        if spatial_info.get('grid_detected', False):
            grid_dims = spatial_info.get('grid_dimensions', (0, 0))

            if grid_dims[0] > 0 and grid_dims[1] > 0:
                if grid_dims[0] == grid_dims[1]:
                    symmetries.update({
                        'rotational_90': True,
                        'rotational_180': True,
                        'horizontal_reflection': True,
                        'vertical_reflection': True
                    })
                else:
                    symmetries.update({
                        'rotational_180': True,
                        'horizontal_reflection': True,
                        'vertical_reflection': True
                    })

        # Additional pattern-based symmetries
        pattern_type = spatial_info.get('pattern_type', 'unknown')
        if pattern_type == 'complex':
            symmetries['complex_pattern'] = True

        return symmetries

    def _classify_extracted_pattern(self, graph: nx.Graph, features: Dict) -> Dict:
        """Classify the extracted pattern"""
        node_count = graph.number_of_nodes()
        extraction_method = features.get('extraction_method', 'unknown')

        # Determine quality based on extraction success
        if node_count == 0:
            quality = 'poor'
        elif extraction_method == 'synthetic_fallback':
            quality = 'reconstructed'
        elif node_count < 5:
            quality = 'limited'
        else:
            quality = 'good'

        return {
            'extraction_quality': quality,
            'pattern_type': features.get('spatial_info', {}).get('pattern_type', 'unknown'),
            'complexity': 'simple' if node_count <= 9 else 'medium' if node_count <= 25 else 'complex',
            'authenticity': 'traditional_kolam',  # Assume traditional for now
            'extraction_method': extraction_method
        }

    def _generate_recreation_from_analysis(self, principles: Dict, features: Dict) -> Dict:
        """Generate recreation data"""
        print("   🎨 Planning recreation...")

        return {
            'recreation_method': 'enhanced_analysis_based',
            'source_analysis': {
                'dots_found': len(features.get('dots', [])),
                'lines_found': len(features.get('lines', [])),
                'quality': principles.get('pattern_classification', {}).get('extraction_quality', 'unknown')
            },
            'recreation_strategy': self._determine_recreation_strategy(principles, features),
            'enhancements': self._suggest_enhancements(principles, features),
            'cultural_authenticity': {
                'traditional_elements': True,
                'color_scheme': 'preserve_original_colors',
                'decorative_style': 'traditional_tamil_kolam'
            }
        }

    def _determine_recreation_strategy(self, principles: Dict, features: Dict) -> List[str]:
        """Determine recreation strategy"""
        strategy = []

        quality = principles.get('pattern_classification', {}).get('extraction_quality', 'poor')

        if quality == 'good':
            strategy.extend([
                "Use detected dot positions",
                "Follow extracted line patterns",
                "Apply detected symmetries"
            ])
        elif quality == 'reconstructed':
            strategy.extend([
                "Use synthetic grid structure",
                "Apply traditional Kolam patterns",
                "Enhance with cultural elements"
            ])
        else:
            strategy.extend([
                "Create idealized version",
                "Apply symmetry completion",
                "Add traditional decorative elements"
            ])

        strategy.extend([
            "Preserve color scheme from original",
            "Add authentic Kolam styling",
            "Ensure cultural accuracy"
        ])

        return strategy

    def _suggest_enhancements(self, principles: Dict, features: Dict) -> List[str]:
        """Suggest enhancements"""
        return [
            "Enhance symmetrical elements",
            "Add decorative loops around dots",
            "Use traditional color palette",
            "Apply authentic line styling",
            "Complete partial patterns using symmetry"
        ]

    def _create_error_result(self, error_message: str) -> Dict:
        """Create error result structure"""
        return {
            'error': True,
            'error_message': error_message,
            'timestamp': datetime.now().isoformat(),
            'partial_results': self.analysis_results if self.analysis_results else None
        }

    def _safe_graph_to_dict(self, graph: nx.Graph) -> Dict:
        """Safely convert graph to dictionary"""
        try:
            return {
                'nodes': list(graph.nodes(data=True)),
                'edges': list(graph.edges(data=True)),
                'number_of_nodes': graph.number_of_nodes(),
                'number_of_edges': graph.number_of_edges()
            }
        except Exception as e:
            return {
                'error': f'Graph conversion failed: {str(e)}',
                'number_of_nodes': 0,
                'number_of_edges': 0
            }

    def create_comprehensive_visualization(self, save_path: Optional[str] = None) -> plt.Figure:
        """Create comprehensive visualization"""
        if not self.analysis_results:
            raise ValueError("No analysis results available")

        print("\n📊 Creating comprehensive visualization...")

        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
        fig.suptitle('Enhanced Kolam Analysis - Your Beautiful Pattern!',
                    fontsize=18, fontweight='bold')

        # Row 1: Image processing stages
        if self.original_image is not None:
            axes[0, 0].imshow(self.original_image)
            axes[0, 0].set_title('Original Kolam Image', fontsize=14, fontweight='bold')
            axes[0, 0].axis('off')

        if self.processed_image is not None:
            axes[0, 1].imshow(self.processed_image, cmap='gray')
            axes[0, 1].set_title('Processed Image', fontsize=14, fontweight='bold')
            axes[0, 1].axis('off')

        self._plot_detected_features(axes[0, 2])

        # Row 2: Analysis results
        self._plot_pattern_graph(axes[1, 0])
        self._plot_analysis_summary(axes[1, 1])
        self._plot_recreation_plan(axes[1, 2])

        plt.tight_layout()

        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f'data/output_results/kolam_analysis_{timestamp}.png'

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📁 Comprehensive visualization saved to {save_path}")

        return fig

    def _plot_detected_features(self, ax):
        """Plot detected features overlay"""
        if self.processed_image is not None:
            ax.imshow(self.processed_image, cmap='gray', alpha=0.7)

        features = self.analysis_results.get('extracted_features', {})

        # Plot dots
        dots = features.get('dots', [])
        if dots:
            dot_x, dot_y = zip(*dots)
            ax.scatter(dot_x, dot_y, c='red', s=150, alpha=0.8,
                      edgecolors='white', linewidths=3, label=f'{len(dots)} Dots')

        # Plot lines
        lines = features.get('lines', [])
        for line in lines:
            if len(line) > 1:
                line_x, line_y = zip(*line)
                ax.plot(line_x, line_y, 'blue', linewidth=2, alpha=0.7)

        if lines:
            ax.plot([], [], 'blue', linewidth=2, label=f'{len(lines)} Line Segments')

        ax.set_title('Detected Features', fontsize=14, fontweight='bold')
        if dots or lines:
            ax.legend()
        ax.axis('equal')

    def _plot_pattern_graph(self, ax):
        """Plot the pattern graph"""
        if self.pattern_graph.number_of_nodes() == 0:
            ax.text(0.5, 0.5, 'No Graph Structure\nDetected',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=16, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
        else:
            pos = nx.get_node_attributes(self.pattern_graph, 'pos')

            # Draw edges
            nx.draw_networkx_edges(self.pattern_graph, pos, ax=ax,
                                 edge_color='darkblue', width=2, alpha=0.7)

            # Draw nodes
            nx.draw_networkx_nodes(self.pattern_graph, pos, ax=ax,
                                 node_color='red', node_size=300,
                                 edgecolors='darkred', linewidths=2)

            ax.set_aspect('equal')

        ax.set_title('Graph Structure', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

    def _plot_analysis_summary(self, ax):
        """Plot analysis summary"""
        ax.axis('off')

        summary = self._format_comprehensive_summary()

        ax.text(0.05, 0.95, summary, transform=ax.transAxes,
               fontsize=12, verticalalignment='top',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))

        ax.set_title('Analysis Summary', fontsize=14, fontweight='bold')

    def _plot_recreation_plan(self, ax):
        """Plot recreation plan"""
        ax.axis('off')

        recreation_data = self.analysis_results.get('recreation_data', {})
        strategy = recreation_data.get('recreation_strategy', [])

        plan_text = "🎨 RECREATION PLAN\n"
        plan_text += "=" * 25 + "\n\n"

        for i, step in enumerate(strategy[:6], 1):  # Show first 6 steps
            plan_text += f"{i}. {step}\n"

        if len(strategy) > 6:
            plan_text += f"\n... and {len(strategy) - 6} more steps"

        ax.text(0.05, 0.95, plan_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

        ax.set_title('Recreation Strategy', fontsize=14, fontweight='bold')

    def _format_comprehensive_summary(self) -> str:
        """Format comprehensive analysis summary"""
        features = self.analysis_results.get('extracted_features', {})
        principles = self.analysis_results.get('design_principles', {})

        summary = "🔍 ANALYSIS RESULTS\n"
        summary += "=" * 30 + "\n"

        # Feature extraction results
        dots_found = len(features.get('dots', []))
        lines_found = len(features.get('lines', []))
        extraction_method = features.get('extraction_method', 'unknown')

        summary += f"📷 Features Extracted:\n"
        summary += f"   • Dots: {dots_found}\n"
        summary += f"   • Lines: {lines_found}\n"
        summary += f"   • Method: {extraction_method}\n\n"

        # Pattern analysis
        basic = principles.get('basic_properties', {})
        classification = principles.get('pattern_classification', {})

        summary += f"📊 Pattern Analysis:\n"
        summary += f"   • Quality: {classification.get('extraction_quality', 'unknown')}\n"
        summary += f"   • Type: {classification.get('pattern_type', 'unknown')}\n"
        summary += f"   • Complexity: {classification.get('complexity', 'unknown')}\n\n"

        # Graph properties
        graph_data = self.analysis_results.get('graph_data', {})
        node_count = graph_data.get('number_of_nodes', 0)
        edge_count = graph_data.get('number_of_edges', 0)

        summary += f"🔗 Graph Structure:\n"
        summary += f"   • Nodes: {node_count}\n"
        summary += f"   • Edges: {edge_count}\n"

        # Symmetries
        symmetries = principles.get('symmetries', {})
        if symmetries:
            detected_symmetries = [k for k, v in symmetries.items() if v]
            summary += f"   • Symmetries: {len(detected_symmetries)}\n"

        summary += f"\n🏛️ Cultural: Traditional Kolam"

        return summary

    def export_results(self, file_path: Optional[str] = None) -> str:
        """Export analysis results"""
        if not self.analysis_results:
            raise ValueError("No results to export")

        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f'data/output_results/enhanced_kolam_analysis_{timestamp}.json'

        try:
            with open(file_path, 'w') as f:
                json.dump(self.analysis_results, f, indent=2, default=str)

            print(f"📁 Enhanced analysis results exported to {file_path}")
            return file_path

        except Exception as e:
            print(f"❌ Export failed: {e}")
            return ""

def main():
    """Enhanced main function"""
    print("🎯 ENHANCED KOLAM PROCESSOR - FIXED VERSION")
    print("=" * 60)
    print("🎨 Now handles colored Kolams with better error handling!")

    processor = EnhancedKolamProcessor()

    print("\n📋 ENHANCED FEATURES:")
    print("✅ Improved colored image processing")
    print("✅ Multiple dot detection methods")
    print("✅ Synthetic structure fallback")
    print("✅ Comprehensive error handling")
    print("✅ Better visualization")

    # Check for images
    input_dir = 'data/input_images'
    if os.path.exists(input_dir):
        image_files = [f for f in os.listdir(input_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]

        if image_files:
            print(f"\n🔍 Processing {len(image_files)} image(s):")

            for image_file in image_files:
                image_path = os.path.join(input_dir, image_file)
                print(f"\n📷 Processing: {image_file}")

                try:
                    # Analyze the image
                    result = processor.analyze_kolam_from_image(image_path)

                    # Create enhanced visualization
                    fig = processor.create_comprehensive_visualization()
                    plt.show()

                    # Export results
                    processor.export_results()

                    print(f"✅ Successfully processed {image_file}!")

                except Exception as e:
                    print(f"❌ Failed to process {image_file}: {str(e)}")
        else:
            print("\n📂 No images found - please add images to 'data/input_images/'")
    else:
        print("\n📁 Creating directories - please add your Kolam images!")

if __name__ == "__main__":
    main()
