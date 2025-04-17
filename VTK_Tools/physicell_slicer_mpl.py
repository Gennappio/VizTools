#!/usr/bin/env python3
"""
PhysiCell Slicer (Matplotlib version): A utility to create slices of PhysiCell output data.

Usage:
    Simply run the script: python physicell_slicer_mpl.py
    
    All parameters are configured in the script itself.
"""

import os
import sys
import argparse
import numpy as np
import scipy.io as sio
from scipy.interpolate import griddata
import xml.etree.ElementTree as ET
import logging
import datetime

# Use backend selection based on command line args - we'll set this properly later
import matplotlib
# Don't set backend yet - we'll do it after setting parameters

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# =============================================================================
# CONFIGURATION - MODIFY PARAMETERS HERE
# =============================================================================

# Main parameters
FILENAME = "../../PhysiCell_micro/PhysiCell/output/output00000002"  # Root name of PhysiCell output files
POSITION = [0, 0, 0]                       # Position of the slice plane [x,y,z]
NORMAL = [0, 0, 1]                         # Normal vector of the slice plane [x,y,z]
SUBSTRATE_INDEX = 11                        # Index of substrate to visualize
CONTOUR_VALUE = None                       # Value to draw contour at (None for no contour)
VALUE_RANGE = None                         # Value range [min,max] for the colormap (None for auto-range)
COLORMAP = 'jet'                           # Colormap to use: jet, viridis, plasma, inferno, magma, etc.
OUTPUT_FILE = None                         # Save image to file instead of displaying (None to auto-generate)

# Advanced parameters
ENABLE_LOGGING = True                      # Enable logging to output_mpl.log
DISPLAY_FIGURE = True                      # Show interactive display
FIGURE_DPI = 100                           # DPI for the output image
FIGURE_SIZE = (10, 8)                      # Figure size in inches (width, height)
GRID_POINTS = 100                          # Number of grid points for interpolation (lower = faster)
DOWNSAMPLE_FACTOR = 1                      # Downsample factor for the input data (higher = faster)

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_enabled):
    """Configure logging to file and console."""
    logger = logging.getLogger('physicell_slicer_mpl')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    if logger.handlers:
        logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    if log_enabled:
        # Create file handler
        file_handler = logging.FileHandler('output_mpl.log', mode='w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Log session start
        logger.info('-' * 70)
        logger.info(f"PhysiCell Slicer (Matplotlib version) started at {datetime.datetime.now()}")
        logger.info('-' * 70)
    
    return logger

# =============================================================================
# FILE HANDLING
# =============================================================================

def find_output_files(filename, logger=None):
    """Find all PhysiCell output files with the given root name."""
    # Get directory and base filename
    base_dir = os.path.dirname(filename)
    base_name = os.path.basename(filename)
    
    # If base_dir is empty, use current directory
    if not base_dir:
        base_dir = "."
    
    # Define expected file patterns with full paths
    xml_file = f"{filename}.xml"
    cells_file = f"{filename}_cells.mat"
    
    # Try different naming patterns for microenvironment files
    microenv_file_candidates = [
        f"{filename}_microenvironment0.mat",
        f"{filename}_microenvironment.mat"
    ]
    
    # Use the first microenvironment file that exists
    microenv_file = next((f for f in microenv_file_candidates if os.path.exists(f)), None)
    
    # Check if files exist
    files = {
        "xml_file": xml_file if os.path.exists(xml_file) else None,
        "cells_file": cells_file if os.path.exists(cells_file) else None,
        "microenv_file": microenv_file
    }
    
    if all(value is None for value in files.values()):
        # If no files found, try checking if user provided path without the output prefix
        # This handles case where user gives path ending with frame number
        output_prefix = os.path.join(base_dir, f"output{base_name}")
        
        # Try alternative file patterns
        xml_file = f"{output_prefix}.xml"
        cells_file = f"{output_prefix}_cells.mat"
        
        microenv_file_candidates = [
            f"{output_prefix}_microenvironment0.mat",
            f"{output_prefix}_microenvironment.mat"
        ]
        
        microenv_file = next((f for f in microenv_file_candidates if os.path.exists(f)), None)
        
        files = {
            "xml_file": xml_file if os.path.exists(xml_file) else None,
            "cells_file": cells_file if os.path.exists(cells_file) else None,
            "microenv_file": microenv_file
        }
    
    if logger:
        logger.info(f"Found files: {files}")
    
    return files

def count_substrates_from_data(microenv_data, logger=None):
    """Count the number of substrates in the microenvironment data."""
    if microenv_data is None:
        if logger:
            logger.error("No microenvironment data available")
        return 0
    
    # First 4 rows are x,y,z positions and time
    num_substrates = microenv_data.shape[0] - 4
    
    if logger:
        logger.info(f"Detected {num_substrates} substrates in microenvironment data")
    
    return num_substrates

def load_microenvironment_data(microenv_file, logger=None):
    """Load microenvironment data from a .mat file."""
    if logger:
        logger.info(f"Loading microenvironment data from {microenv_file}")
    
    try:
        mat_contents = sio.loadmat(microenv_file)
        mat_contents = {k:v for k, v in mat_contents.items() 
                       if not k.startswith('__')}
        
        if 'multiscale_microenvironment' in mat_contents:
            return mat_contents['multiscale_microenvironment']
        
        # For older versions or different naming
        for key in mat_contents:
            if 'microenvironment' in key.lower():
                if logger:
                    logger.info(f"Found microenvironment data in key: {key}")
                return mat_contents[key]
                
        if logger:
            logger.error(f"No microenvironment data found in file: {microenv_file}")
            logger.info(f"Available keys: {list(mat_contents.keys())}")
        
        return None
        
    except Exception as e:
        if logger:
            logger.error(f"Error loading microenvironment data: {e}")
            import traceback
            logger.error(traceback.format_exc())
        return None

def parse_substrates_from_xml(xml_file, logger=None):
    """Parse substrate names from PhysiCell XML configuration file."""
    if not xml_file or not os.path.exists(xml_file):
        if logger:
            logger.warning(f"XML file not found: {xml_file}")
        return []
    
    try:
        if logger:
            logger.info(f"Parsing XML file: {xml_file}")
        
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Log XML structure for debugging
        if logger:
            logger.info(f"XML root tag: {root.tag}")
            logger.info(f"First level tags: {[child.tag for child in root]}")
        
        # Try multiple possible paths for microenvironment node
        possible_paths = [
            ".//microenvironment",
            ".//microenvironment_setup",
            ".//domain/microenvironment",
            ".//PhysiCell_settings/microenvironment_setup"
        ]
        
        microenv_node = None
        for path in possible_paths:
            if logger:
                logger.info(f"Trying microenvironment path: {path}")
            node = root.find(path)
            if node is not None:
                microenv_node = node
                if logger:
                    logger.info(f"Found microenvironment node with path: {path}")
                break
        
        if microenv_node is None:
            if logger:
                logger.warning("No microenvironment node found in XML using common paths")
                logger.info("Trying to find any node containing variable definitions...")
            
            # Last resort: look for any nodes that might contain variable definitions
            variable_nodes = root.findall(".//variable")
            if variable_nodes:
                if logger:
                    logger.info(f"Found {len(variable_nodes)} variable nodes directly in the XML")
                # Process these nodes directly without a microenvironment parent
                substrate_names = []
                for var_node in variable_nodes:
                    name_node = var_node.find("name")
                    if name_node is not None and name_node.text:
                        substrate_names.append(name_node.text.strip())
                        if logger:
                            logger.info(f"Found substrate: {name_node.text.strip()}")
                
                return substrate_names
            else:
                if logger:
                    logger.warning("No variable nodes found in the XML")
                return []
        
        # Extract variable nodes from the found microenvironment node
        # Try different possible variable container paths
        variable_containers = [
            microenv_node,  # Variables directly in microenvironment
            microenv_node.find("domain"),  # Variables in domain
            microenv_node.find("variables"),  # Variables in variables container
        ]
        
        variable_nodes = []
        for container in variable_containers:
            if container is not None:
                vars_found = container.findall(".//variable")
                if not vars_found:
                    # Try direct children
                    vars_found = container.findall("variable")
                
                if vars_found:
                    variable_nodes = vars_found
                    break
        
        if not variable_nodes:
            # One more attempt with a direct search in the microenvironment node
            variable_nodes = microenv_node.findall(".//variable")
        
        if logger:
            logger.info(f"Found {len(variable_nodes) if variable_nodes else 0} variable nodes")
        
        substrate_names = []
        
        # Look for name in different possible locations
        for var_node in variable_nodes:
            # Try finding name node as direct child
            name_node = var_node.find("name")
            
            # If not found, try id node
            if name_node is None or not name_node.text:
                name_node = var_node.find("ID")
            
            # If still not found, try attribute
            if (name_node is None or not name_node.text) and 'name' in var_node.attrib:
                substrate_names.append(var_node.attrib['name'])
                if logger:
                    logger.info(f"Found substrate from attribute: {var_node.attrib['name']}")
            elif name_node is not None and name_node.text:
                substrate_names.append(name_node.text.strip())
                if logger:
                    logger.info(f"Found substrate: {name_node.text.strip()}")
        
        # If we still don't have names, look for any text in the variable nodes
        if not substrate_names:
            for i, var_node in enumerate(variable_nodes):
                # Use any text content as name
                if var_node.text and var_node.text.strip():
                    substrate_names.append(var_node.text.strip())
                    if logger:
                        logger.info(f"Found substrate from text content: {var_node.text.strip()}")
                else:
                    # As a last resort, use the tag with index
                    substrate_names.append(f"variable_{i}")
                    if logger:
                        logger.info(f"Using default name: variable_{i}")
        
        if logger:
            if substrate_names:
                logger.info(f"Found {len(substrate_names)} substrates: {substrate_names}")
            else:
                logger.warning("No substrate names found in XML file")
        
        return substrate_names
    
    except Exception as e:
        if logger:
            logger.error(f"Error parsing XML file: {e}")
            import traceback
            logger.error(traceback.format_exc())
        return []

def find_physicell_settings_xml(base_dir, logger=None):
    """
    Try to find the PhysiCell_settings.xml file that might contain substrate names.
    
    Args:
        base_dir: Base directory to start the search
        logger: Logger object
        
    Returns:
        Path to PhysiCell_settings.xml if found, None otherwise
    """
    if logger:
        logger.info(f"Looking for PhysiCell_settings.xml in/near {base_dir}")
    
    # First check the immediate directory
    settings_file = os.path.join(base_dir, "PhysiCell_settings.xml")
    if os.path.exists(settings_file):
        if logger:
            logger.info(f"Found settings file at {settings_file}")
        return settings_file
    
    # Try parent directory
    parent_dir = os.path.dirname(base_dir)
    settings_file = os.path.join(parent_dir, "PhysiCell_settings.xml")
    if os.path.exists(settings_file):
        if logger:
            logger.info(f"Found settings file at {settings_file}")
        return settings_file
    
    # Try config subdirectory of parent
    config_dir = os.path.join(parent_dir, "config")
    settings_file = os.path.join(config_dir, "PhysiCell_settings.xml")
    if os.path.exists(settings_file):
        if logger:
            logger.info(f"Found settings file at {settings_file}")
        return settings_file
    
    # Try a few other common locations
    common_locations = [
        os.path.join(parent_dir, ".."), # Grandparent directory
        os.path.join(parent_dir, "..", "config"),
        os.path.join(base_dir, "config")
    ]
    
    for location in common_locations:
        settings_file = os.path.join(location, "PhysiCell_settings.xml")
        if os.path.exists(settings_file):
            if logger:
                logger.info(f"Found settings file at {settings_file}")
            return settings_file
    
    if logger:
        logger.warning("Could not find PhysiCell_settings.xml file")
    return None

def get_substrate_name(substrate_index, substrates, logger=None):
    """Get substrate name for the given index."""
    if not substrates:
        if logger:
            logger.warning(f"Using numeric substrate index {substrate_index} (no substrate names available)")
        return f"substrate_{substrate_index}"
    
    # Adjust for the position and orientation data in the microenvironment
    data_offset = 4  # Typically: x,y,z positions and timestamp
    
    # Check if the substrate index is valid
    if substrate_index >= 0 and substrate_index < len(substrates):
        name = substrates[substrate_index]
        if logger:
            logger.info(f"Using substrate '{name}' (index {substrate_index})")
        return name
    else:
        if logger:
            logger.warning(f"Substrate index {substrate_index} out of range, max index is {len(substrates)-1}")
            logger.warning(f"Using numeric substrate index {substrate_index}")
        return f"substrate_{substrate_index}"

# =============================================================================
# SLICE CREATION
# =============================================================================

def compute_plane_points(position, normal, bounds, num_points=100, logger=None):
    """
    Compute points on a plane defined by a position and normal vector,
    constrained by the given bounds.
    
    Args:
        position: [x, y, z] position of a point on the plane
        normal: [nx, ny, nz] normal vector of the plane
        bounds: [xmin, xmax, ymin, ymax, zmin, zmax] bounds of the domain
        num_points: number of points to generate along each dimension
        
    Returns:
        x_grid, y_grid, z_grid: 2D arrays of points on the plane
    """
    if logger:
        logger.info(f"Computing plane points at position {position} with normal {normal}")
    
    # Normalize the normal vector - add safety check for zero vector
    normal = np.array(normal)
    norm = np.linalg.norm(normal)
    
    # Safety check for zero normal vector
    if norm < 1e-10:
        if logger:
            logger.warning("Normal vector is too close to zero, using default [0,0,1]")
        normal = np.array([0, 0, 1])
    else:
        normal = normal / norm
    
    # Create two orthogonal vectors in the plane - with numeric stability checks
    if np.abs(normal[0]) < np.abs(normal[1]) and np.abs(normal[0]) < np.abs(normal[2]):
        v1 = np.array([0, -normal[2], normal[1]])
    elif np.abs(normal[1]) < np.abs(normal[0]) and np.abs(normal[1]) < np.abs(normal[2]):
        v1 = np.array([-normal[2], 0, normal[0]])
    else:
        v1 = np.array([-normal[1], normal[0], 0])
    
    # Safety check for zero v1 vector
    v1_norm = np.linalg.norm(v1)
    if v1_norm < 1e-10:
        # Try a different approach to get perpendicular vector
        if np.abs(normal[2]) > 0.9:
            # If normal is close to Z axis, use X axis for v1
            v1 = np.array([1, 0, 0])
        else:
            # Otherwise use Z axis
            v1 = np.array([0, 0, 1])
        # Make v1 perpendicular to normal
        v1 = v1 - np.dot(v1, normal) * normal
        v1_norm = np.linalg.norm(v1)
        if v1_norm < 1e-10:
            if logger:
                logger.warning("Could not compute orthogonal vector, using default axes")
            # Last resort - use standard axes
            v1 = np.array([1, 0, 0])
            v2 = np.array([0, 1, 0])
            
            # Make sure v1 is not parallel to normal
            v1_dot = np.abs(np.dot(v1, normal))
            v2_dot = np.abs(np.dot(v2, normal))
            
            if v1_dot > 0.9:
                # v1 is too close to normal, use v2
                v1 = v2
            
            # Make v1 perpendicular to normal
            v1 = v1 - np.dot(v1, normal) * normal
            v1_norm = np.linalg.norm(v1)
            
            if v1_norm < 1e-10:
                # Still have problems - just use a fallback plane
                if logger:
                    logger.warning("Using fallback XY plane for visualization")
                x = np.linspace(bounds[0], bounds[1], num_points)
                y = np.linspace(bounds[2], bounds[3], num_points)
                x_grid, y_grid = np.meshgrid(x, y)
                z_grid = np.ones_like(x_grid) * position[2]
                return x_grid, y_grid, z_grid
    
    v1 = v1 / v1_norm
    v2 = np.cross(normal, v1)
    
    # Determine appropriate scale based on bounds
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    scale = max(xmax - xmin, ymax - ymin, zmax - zmin) / 2
    
    # Generate points in the plane - reduce density for performance
    u = np.linspace(-scale, scale, num_points)
    v = np.linspace(-scale, scale, num_points)
    u_grid, v_grid = np.meshgrid(u, v)
    
    # Convert to 3D coordinates
    position = np.array(position)
    x_grid = position[0] + u_grid * v1[0] + v_grid * v2[0]
    y_grid = position[1] + u_grid * v1[1] + v_grid * v2[1]
    z_grid = position[2] + u_grid * v1[2] + v_grid * v2[2]
    
    # Final safety check - make sure there are no NaN values
    if np.any(np.isnan(x_grid)) or np.any(np.isnan(y_grid)) or np.any(np.isnan(z_grid)):
        if logger:
            logger.warning("NaN values detected in grid, using fallback slice")
        # Use a simple slice in a standard plane
        if np.abs(normal[2]) > np.abs(normal[0]) and np.abs(normal[2]) > np.abs(normal[1]):
            # Normal is primarily along Z, use XY plane
            x = np.linspace(bounds[0], bounds[1], num_points)
            y = np.linspace(bounds[2], bounds[3], num_points)
            x_grid, y_grid = np.meshgrid(x, y)
            z_grid = np.ones_like(x_grid) * position[2]
        elif np.abs(normal[1]) > np.abs(normal[0]) and np.abs(normal[1]) > np.abs(normal[2]):
            # Normal is primarily along Y, use XZ plane
            x = np.linspace(bounds[0], bounds[1], num_points)
            z = np.linspace(bounds[4], bounds[5], num_points)
            x_grid, z_grid = np.meshgrid(x, z)
            y_grid = np.ones_like(x_grid) * position[1]
        else:
            # Normal is primarily along X, use YZ plane
            y = np.linspace(bounds[2], bounds[3], num_points)
            z = np.linspace(bounds[4], bounds[5], num_points)
            y_grid, z_grid = np.meshgrid(y, z)
            x_grid = np.ones_like(y_grid) * position[0]
    
    return x_grid, y_grid, z_grid

def create_slice(microenv_data, position, normal, substrate_idx, logger=None):
    """
    Create a slice of the microenvironment data using interpolation.
    
    Args:
        microenv_data: PhysiCell microenvironment data matrix
        position: [x, y, z] position of a point on the slice plane
        normal: [nx, ny, nz] normal vector of the slice plane
        substrate_idx: index of the substrate to visualize
        
    Returns:
        dict with slice data, including coordinates, values, and metadata
    """
    if microenv_data is None:
        if logger:
            logger.error("No microenvironment data to slice")
        return None
    
    if logger:
        logger.info(f"Creating slice at position {position} with normal {normal}")
    
    # Extract positions (first 3 rows) and substrate data
    positions = microenv_data[0:3, :]
    substrate = microenv_data[substrate_idx + 4, :]  # +4 to skip x,y,z and time
    
    # Get bounds of the data
    x_min, x_max = np.min(positions[0, :]), np.max(positions[0, :])
    y_min, y_max = np.min(positions[1, :]), np.max(positions[1, :])
    z_min, z_max = np.min(positions[2, :]), np.max(positions[2, :])
    bounds = [x_min, x_max, y_min, y_max, z_min, z_max]
    
    # Get min and max values for the substrate
    data_min, data_max = np.min(substrate), np.max(substrate)
    
    if logger:
        logger.info(f"Grid bounds: x: [{x_min:.1f}, {x_max:.1f}], y: [{y_min:.1f}, {y_max:.1f}], z: [{z_min:.1f}, {z_max:.1f}]")
        logger.info(f"Substrate range: {data_min} to {data_max}")
        logger.info(f"Data points: {len(positions[0])} (may need to reduce for large datasets)")
    
    # Check if dataset is very large - downsampling if needed
    MAX_POINTS = 100000  # Maximum number of points to use for interpolation
    if positions.shape[1] > MAX_POINTS:
        if logger:
            logger.warning(f"Large dataset detected ({positions.shape[1]} points), downsampling to {MAX_POINTS} points")
        
        # Downsample the data by taking every nth point
        n = positions.shape[1] // MAX_POINTS + 1
        positions_ds = positions[:, ::n]
        substrate_ds = substrate[::n]
        
        if logger:
            logger.info(f"Downsampled to {positions_ds.shape[1]} points")
            
        points_3d = positions_ds.T
        substrate_values = substrate_ds
    else:
        points_3d = positions.T
        substrate_values = substrate
    
    # Use fewer points for the output grid to improve performance
    grid_resolution = max(50, min(100, 2000000 // len(positions[0])))  # Adjust based on input size
    
    if logger:
        logger.info(f"Using grid resolution of {grid_resolution}x{grid_resolution} for slice")
    
    # Create a grid of points on the slice plane (reduced resolution)
    x_grid, y_grid, z_grid = compute_plane_points(position, normal, bounds, num_points=grid_resolution, logger=logger)
    
    # Interpolate using optimized method
    if logger:
        logger.info("Starting interpolation (this might take a moment)...")
    
    # Use faster 'nearest' interpolation for extremely large datasets
    method = 'linear'
    if len(points_3d) > 500000:
        method = 'nearest'
        logger.info(f"Using faster '{method}' interpolation due to large dataset size")
    
    # Do the interpolation
    slice_values = griddata(points_3d, substrate_values, (x_grid, y_grid, z_grid), method=method)
    
    if logger:
        logger.info("Interpolation complete")
    
    # Create slice data object
    slice_data = {
        'x_grid': x_grid,
        'y_grid': y_grid,
        'z_grid': z_grid,
        'values': slice_values,
        'scalar_range': (data_min, data_max),
        'bounds': bounds
    }
    
    return slice_data

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_slice_visualization(filename, position, normal, substrate_index, 
                              contour_value=None, colormap='jet', value_range=None, 
                              output_file=None, fig_size=(10, 8), dpi=100, logger=None):
    """
    Create a matplotlib visualization of a slice through the microenvironment data.
    
    Args:
        filename: root name of PhysiCell output files
        position: [x, y, z] position of a point on the slice plane
        normal: [nx, ny, nz] normal vector of the slice plane
        substrate_index: index of the substrate to visualize
        contour_value: value to draw contour at (or None for no contour)
        colormap: colormap name (e.g., 'viridis', 'jet')
        value_range: [min, max] range for the colormap (or None for auto-range)
        output_file: file to save image to (or None to display)
        fig_size: figure size in inches (width, height)
        dpi: dots per inch for the figure
        logger: logger object
        
    Returns:
        matplotlib figure with the visualization
    """
    # Ensure logger is initialized
    if logger is None:
        logger = setup_logging(False)
    
    # Find output files
    files = find_output_files(filename, logger)
    
    # Check if microenvironment file exists
    if not files['microenv_file']:
        logger.error(f"No microenvironment file found for {filename}")
        return None
    
    # Parse substrate names from XML file
    substrates = parse_substrates_from_xml(files['xml_file'], logger)
    
    # Load microenvironment data
    microenv_data = load_microenvironment_data(files['microenv_file'], logger)
    
    if microenv_data is None:
        logger.error("Failed to load microenvironment data")
        return None
    
    # Get substrate name
    substrate_name = get_substrate_name(substrate_index, substrates, logger)
    
    # Create a slice through the data
    slice_data = create_slice(microenv_data, position, normal, substrate_index, logger)
    
    if slice_data is None:
        logger.error("Failed to create slice")
        return None
    
    # Extract slice data
    x_grid = slice_data['x_grid']
    y_grid = slice_data['y_grid']
    z_grid = slice_data['z_grid']
    slice_values = slice_data['values']
    scalar_range = slice_data['scalar_range']
    
    # Check for non-finite values that will cause plotting errors
    non_finite_mask = ~np.isfinite(slice_values)
    if np.any(non_finite_mask):
        count_nans = np.sum(np.isnan(slice_values))
        count_infs = np.sum(np.isinf(slice_values))
        logger.warning(f"Found {count_nans} NaN values and {count_infs} Inf values in slice data")
        
        # Replace non-finite values with a default value (e.g., 0)
        logger.info("Replacing non-finite values with 0")
        slice_values = np.nan_to_num(slice_values, nan=0.0, posinf=scalar_range[1], neginf=scalar_range[0])
    
    # Check for non-finite values in coordinates
    if np.any(~np.isfinite(x_grid)) or np.any(~np.isfinite(y_grid)) or np.any(~np.isfinite(z_grid)):
        logger.warning("Found non-finite values in coordinate grids, attempting to fix")
        
        # Filter out points where coordinates are non-finite
        x_grid = np.nan_to_num(x_grid, nan=0.0, posinf=0.0, neginf=0.0)
        y_grid = np.nan_to_num(y_grid, nan=0.0, posinf=0.0, neginf=0.0)
        z_grid = np.nan_to_num(z_grid, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Use custom range if provided, otherwise use auto-range
    if value_range:
        min_val, max_val = value_range
    else:
        min_val, max_val = scalar_range
        # Ensure range is valid (no NaN values)
        if not np.isfinite(min_val) or not np.isfinite(max_val):
            logger.warning("Range contains non-finite values, using default range [0, 1]")
            min_val, max_val = 0, 1
        logger.info(f"Using auto range: {min_val} to {max_val}")
        
    # Create a new figure
    fig = plt.figure(figsize=fig_size, dpi=dpi, facecolor='white')
    
    # Determine which dimensions to show based on the normal vector
    norm_abs = [abs(n) for n in normal]
    if norm_abs[0] > norm_abs[1] and norm_abs[0] > norm_abs[2]:
        # Normal is primarily in X direction, show Y-Z plane
        h_axis = y_grid
        v_axis = z_grid
        h_label = 'Y Position'
        v_label = 'Z Position'
    elif norm_abs[1] > norm_abs[0] and norm_abs[1] > norm_abs[2]:
        # Normal is primarily in Y direction, show X-Z plane
        h_axis = x_grid
        v_axis = z_grid
        h_label = 'X Position'
        v_label = 'Z Position'
    else:
        # Normal is primarily in Z direction, show X-Y plane
        h_axis = x_grid
        v_axis = y_grid
        h_label = 'X Position'
        v_label = 'Y Position'
    
    # Create the main plot
    ax = fig.add_subplot(111)
    
    try:
        # Try to use pcolormesh with error checking
        logger.info("Creating visualization...")
        
        # Check for invalid input to pcolormesh
        if (np.any(~np.isfinite(h_axis)) or np.any(~np.isfinite(v_axis)) or 
            np.any(~np.isfinite(slice_values))):
            logger.warning("Plotting data contains non-finite values, replacing them")
            h_axis = np.nan_to_num(h_axis)
            v_axis = np.nan_to_num(v_axis)
            slice_values = np.nan_to_num(slice_values)
        
        # Handle masked arrays (convert to regular arrays)
        if isinstance(h_axis, np.ma.MaskedArray):
            h_axis = h_axis.filled(0)
        if isinstance(v_axis, np.ma.MaskedArray):
            v_axis = v_axis.filled(0)
        if isinstance(slice_values, np.ma.MaskedArray):
            slice_values = slice_values.filled(0)
        
        # If data is too sparse, try using imshow instead
        if np.sum(~np.isnan(slice_values)) < 10:
            logger.warning("Too few valid data points for pcolormesh, using imshow instead")
            im = ax.imshow(slice_values, 
                          cmap=colormap, 
                          vmin=min_val, vmax=max_val,
                          origin='lower', 
                          extent=[np.min(h_axis), np.max(h_axis), np.min(v_axis), np.max(v_axis)])
        else:
            # Plot the slice data as a filled contour plot
            im = ax.pcolormesh(h_axis, v_axis, slice_values, 
                              cmap=colormap, 
                              vmin=min_val, vmax=max_val,
                              shading='auto')
    except Exception as e:
        logger.error(f"Error creating pcolormesh plot: {str(e)}")
        logger.info("Falling back to simpler plotting method...")
        
        # Fall back to imshow which is more robust
        valid_values = np.nan_to_num(slice_values)
        im = ax.imshow(valid_values, 
                      cmap=colormap, 
                      vmin=min_val, vmax=max_val,
                      origin='lower',
                      extent=[np.min(h_axis), np.max(h_axis), np.min(v_axis), np.max(v_axis)])
    
    # Add contour lines if requested
    if contour_value is not None:
        try:
            contour = ax.contour(h_axis, v_axis, slice_values, 
                               levels=[contour_value], 
                               colors='black', 
                               linewidths=2)
            logger.info(f"Added contour at value {contour_value}")
        except Exception as e:
            logger.error(f"Error adding contour: {str(e)}")
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(substrate_name)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Set labels and title
    ax.set_xlabel(h_label)
    ax.set_ylabel(v_label)
    ax.set_title(f"PhysiCell Slice - {substrate_name} - {os.path.basename(filename)}")
    
    # Add text with information
    info_text = f"File: {os.path.basename(filename)}\n"
    info_text += f"Position: [{position[0]}, {position[1]}, {position[2]}]\n"
    info_text += f"Normal: [{normal[0]}, {normal[1]}, {normal[2]}]\n"
    info_text += f"Substrate: {substrate_name}\n"
    info_text += f"Range: {min_val:.2f} to {max_val:.2f}"
    if contour_value is not None:
        info_text += f"\nContour at: {contour_value:.3f}"
    
    # Add text box for information
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', bbox=props)
    
    # Adjust layout
    plt.tight_layout()
    
    # Log summary
    if logger:
        logger.info("\nVisualization Summary:")
        logger.info("-" * 40)
        logger.info(f"File: {os.path.basename(filename)}")
        logger.info(f"Position: {position}")
        logger.info(f"Normal: {normal}")
        logger.info(f"Substrate: {substrate_name} (index {substrate_index})")
        logger.info(f"Value range: {min_val} to {max_val}")
        if contour_value is not None:
            logger.info(f"Contour at value: {contour_value}")
        logger.info("-" * 40)
    
    # Save or display the figure
    if output_file:
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        logger.info(f"Figure saved to {output_file}")
    
    return fig

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def log_substrate_details(logger, substrate_names, microenv_data):
    """Log detailed substrate information with indices, names, and value ranges."""
    if logger and microenv_data is not None:
        logger.info("\n" + "-" * 60)
        logger.info("Index  Name                           Min Value    Max Value   ")
        logger.info("-" * 60)
        
        # Calculate how many substrates are in the data
        num_substrates = microenv_data.shape[0] - 4  # First 4 rows are x,y,z,time
        
        # Ensure we have names for all substrates
        if len(substrate_names) < num_substrates:
            # Add generic names for missing substrates
            for i in range(len(substrate_names), num_substrates):
                substrate_names.append(f"substrate_{i}")
        
        # Log each substrate with its min and max values
        for i in range(num_substrates):
            substrate_data = microenv_data[i + 4, :]  # +4 to skip x,y,z,time
            min_val = np.min(substrate_data)
            max_val = np.max(substrate_data)
            name = substrate_names[i] if i < len(substrate_names) else f"substrate_{i}"
            
            # Format nicely
            logger.info(f"{i:<6} {name:<30} {min_val:<11.6f} {max_val:<11.6f}")
        
        logger.info("-" * 60)
        logger.info("")  # Empty line at the end

def main():
    """Main function to run the PhysiCell slicer."""
    # Configure the matplotlib backend based on display settings
    if not DISPLAY_FIGURE:
        matplotlib.use('Agg')  # Non-interactive backend for saving only
        print("Using non-interactive Agg backend for saving without display")
    else:
        # Use an appropriate interactive backend based on platform
        if sys.platform == 'darwin':  # macOS
            matplotlib.use('MacOSX')
            print("Using MacOSX backend for interactive display")
        else:
            try:
                matplotlib.use('TkAgg')
                print("Using TkAgg backend for interactive display")
            except ImportError:
                print("TkAgg not available, trying Qt5Agg")
                try:
                    matplotlib.use('Qt5Agg')
                    print("Using Qt5Agg backend for interactive display")
                except ImportError:
                    print("No interactive backend available, falling back to Agg")
                    matplotlib.use('Agg')
                    print("Using non-interactive Agg backend (interactive display not available)")
    
    # Set up logging
    logger = setup_logging(ENABLE_LOGGING)
    
    logger.info(f"Processing PhysiCell output with root: {FILENAME}")
    
    # Find PhysiCell output files
    files = find_output_files(FILENAME, logger)
    
    # Load microenvironment data
    microenv_data = load_microenvironment_data(files['microenv_file'], logger)
    if microenv_data is None:
        logger.error("Failed to load microenvironment data")
        return 1
    
    # Count substrates from microenvironment data
    num_substrates = count_substrates_from_data(microenv_data, logger)
    
    # Try to get substrate names from XML file
    substrates = parse_substrates_from_xml(files['xml_file'], logger)
    
    # If no substrates found in output XML, try looking for PhysiCell_settings.xml
    if not substrates or len(substrates) == 0:
        logger.info("No substrate names found in output XML, looking for PhysiCell_settings.xml")
        base_dir = os.path.dirname(files['microenv_file']) if files['microenv_file'] else os.path.dirname(FILENAME)
        settings_file = find_physicell_settings_xml(base_dir, logger)
        if settings_file:
            logger.info(f"Trying to parse substrate names from {settings_file}")
            substrates = parse_substrates_from_xml(settings_file, logger)
    
    # Check if we found substrate names that match the count in the data
    if not substrates:
        logger.warning("No substrate names found in XML file")
        substrates = [f"substrate_{i}" for i in range(num_substrates)]
        logger.info(f"Using default substrate names")
    elif len(substrates) != num_substrates:
        logger.warning(f"Number of substrate names from XML ({len(substrates)}) doesn't match number of substrates in data ({num_substrates})")
        # Expand or truncate the list as needed
        if len(substrates) < num_substrates:
            # Add generic names for missing substrates
            for i in range(len(substrates), num_substrates):
                substrates.append(f"substrate_{i}")
            logger.info(f"Added default names for missing substrates")
        else:
            # Truncate extra names
            substrates = substrates[:num_substrates]
            logger.info(f"Truncated extra substrate names")
    
    # Log substrate details exactly like physicell_slicer.py does
    log_substrate_details(logger, substrates, microenv_data)
    
    # Verify the selected substrate index is valid
    if SUBSTRATE_INDEX < 0 or SUBSTRATE_INDEX >= num_substrates:
        logger.warning(f"Selected substrate index {SUBSTRATE_INDEX} is out of range (0-{num_substrates-1})!")
        logger.warning("Using substrate 0 instead")
        selected_substrate_idx = 0
    else:
        selected_substrate_idx = SUBSTRATE_INDEX
    
    # Get substrate name for the selected index
    selected_substrate_name = substrates[selected_substrate_idx] if selected_substrate_idx < len(substrates) else f"substrate_{selected_substrate_idx}"
    
    # Get substrate range
    substrate_values = microenv_data[selected_substrate_idx + 4, :]
    min_val = np.min(substrate_values)
    max_val = np.max(substrate_values)
    logger.info(f"Using substrate: {selected_substrate_idx} - {selected_substrate_name} (range: {min_val:.4f} to {max_val:.4f})")
    
    # Also print to console for user convenience
    print("\nAvailable substrates:")
    print("-" * 40)
    print(f"{'Index':<6} {'Substrate Name':<30} {'Range':<20}")
    print("-" * 40)
    
    for idx in range(num_substrates):
        substrate_values = microenv_data[idx + 4, :]
        min_val = np.min(substrate_values)
        max_val = np.max(substrate_values)
        name = substrates[idx] if idx < len(substrates) else f"substrate_{idx}"
        
        # Print with highlighting for non-zero substrates
        has_data = max_val > 0
        range_info = f"{min_val:.4f} to {max_val:.4f}"
        
        # Add an indicator for the currently selected substrate
        is_selected = (idx == SUBSTRATE_INDEX)
        selector = "=>" if is_selected else "  "
        
        if has_data:
            print(f"{selector} {idx:<3} {name:<30} {range_info:<20} *HAS DATA*")
        else:
            print(f"{selector} {idx:<3} {name:<30} {range_info:<20}")
    
    print("-" * 40)
    print(f"Using substrate: {selected_substrate_idx} - {selected_substrate_name} (range: {min_val:.4f} to {max_val:.4f})\n")
    
    # Generate default output filename if needed
    output_file = OUTPUT_FILE
    if not output_file:
        base_name = os.path.basename(FILENAME)
        output_file = f"{base_name}_s{SUBSTRATE_INDEX}_slice.png"
        logger.info(f"Using default output filename: {output_file}")
    
    # Create the visualization
    try:
        fig = create_slice_visualization(
            filename=FILENAME,
            position=POSITION,
            normal=NORMAL,
            substrate_index=SUBSTRATE_INDEX,
            contour_value=CONTOUR_VALUE,
            colormap=COLORMAP,
            value_range=VALUE_RANGE,
            output_file=output_file,
            fig_size=FIGURE_SIZE,
            dpi=FIGURE_DPI,
            logger=logger
        )
        
        if fig:
            logger.info(f"Figure saved to {output_file}")
            logger.info("Visualization complete")
            
            # Only try to display if requested
            if DISPLAY_FIGURE:
                logger.info("Displaying figure interactively (close window to exit)...")
                plt.show()
                logger.info("Figure window closed")
        else:
            logger.error("Failed to create visualization")
    except Exception as e:
        logger.error(f"Error during visualization: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        print(f"Error: {str(e)}. See output_mpl.log for details.")
        return 1
    
    return 0

if __name__ == "__main__":
    main() 