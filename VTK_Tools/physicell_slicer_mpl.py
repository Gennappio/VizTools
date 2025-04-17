#!/usr/bin/env python3
"""
PhysiCell Slicer (Matplotlib version): A utility to create slices of PhysiCell output data.

Usage:
    python physicell_slicer_mpl.py --filename <filename> [--log] [--position x,y,z] [--normal x,y,z] 
                              [--substrate index] [--contour value] [--range min,max] [--colormap name] 
                              [--output output_file.png]

Parameters:
    --filename:     Root name of PhysiCell output files (without extension)
                    (e.g., 'output0000060' will look for output0000060.xml, output0000060_cells.mat, etc.)
    --log:          Enable logging to output_mpl.log
    --position:     Position of the slice plane [x,y,z] (default: [0,0,0])
    --normal:       Normal vector of the slice plane [x,y,z] (default: [0,0,1])
    --substrate:    Index of substrate to visualize (default: 0)
    --contour:      Value to draw contour at (optional)
    --range:        Value range [min,max] for the colormap (optional)
    --colormap:     Colormap to use: jet, viridis, plasma, inferno, magma, etc. (default: jet)
    --output:       Save image to file instead of displaying (optional)
    --no-display:   Don't show interactive display, only save to file (default: False)
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
# Don't set backend yet - we'll do it after parsing arguments

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
        return None
    
    if logger:
        logger.info(f"Parsing substrate names from {xml_file}")
    
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        substrates = []
        
        # Try to find substrates in different possible XML structures
        microenv_nodes = root.findall('.//microenvironment_setup')
        if microenv_nodes:
            for microenv in microenv_nodes:
                for variable in microenv.findall('.//variable'):
                    name = variable.find('name')
                    if name is not None and name.text:
                        substrates.append(name.text.strip())
        
        # If no substrates found in the primary location, try alternative locations
        if not substrates:
            for variable in root.findall('.//variable'):
                name = variable.find('name')
                if name is not None and name.text:
                    substrates.append(name.text.strip())
        
        if logger:
            if substrates:
                logger.info(f"Found {len(substrates)} substrates: {substrates}")
            else:
                logger.warning("No substrate names found in XML file")
        
        return substrates
    
    except Exception as e:
        if logger:
            logger.error(f"Error parsing XML file: {e}")
            import traceback
            logger.error(traceback.format_exc())
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

def main():
    """Main function to run the PhysiCell slicer."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='PhysiCell Slicer (Matplotlib version)')
    parser.add_argument('--filename', required=True, help='Root name of PhysiCell output files')
    parser.add_argument('--log', action='store_true', help='Enable logging to output_mpl.log')
    parser.add_argument('--position', default='0,0,0', help='Position of the slice plane [x,y,z]')
    parser.add_argument('--normal', default='0,0,1', help='Normal vector of the slice plane [x,y,z]')
    parser.add_argument('--substrate', type=int, default=0, help='Index of substrate to visualize')
    parser.add_argument('--contour', type=float, help='Value to draw contour at')
    parser.add_argument('--range', help='Value range [min,max] for the colormap')
    parser.add_argument('--colormap', default='jet', help='Colormap to use')
    parser.add_argument('--output', help='Save image to file instead of displaying')
    parser.add_argument('--no-display', action='store_true', help='Don\'t show interactive display, only save to file')
    parser.add_argument('--dpi', type=int, default=100, help='DPI for the output image')
    parser.add_argument('--grid-points', type=int, default=None, help='Number of grid points for interpolation (lower = faster)')
    parser.add_argument('--downsample', type=int, default=1, help='Downsample factor for the input data (higher = faster)')
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(args.log)
    
    # IMPORTANT: Configure matplotlib backend BEFORE importing pyplot functionality
    # This avoids the backend switching issue
    if args.no_display:
        matplotlib.use('Agg')  # Non-interactive backend for saving only
        logger.info("Using non-interactive Agg backend for saving without display")
    else:
        # Use an appropriate interactive backend based on platform
        if sys.platform == 'darwin':  # macOS
            matplotlib.use('MacOSX')
            logger.info("Using MacOSX backend for interactive display")
        else:
            try:
                matplotlib.use('TkAgg')
                logger.info("Using TkAgg backend for interactive display")
            except ImportError:
                logger.warning("TkAgg not available, trying Qt5Agg")
                try:
                    matplotlib.use('Qt5Agg')
                    logger.info("Using Qt5Agg backend for interactive display")
                except ImportError:
                    logger.warning("No interactive backend available, falling back to Agg")
                    matplotlib.use('Agg')
                    logger.info("Using non-interactive Agg backend (interactive display not available)")
                    args.no_display = True  # Force no-display mode since we can't show interactively
    
    # Process arguments
    logger.info(f"Processing PhysiCell output with root: {args.filename}")
    
    # Parse position
    try:
        position = [float(x) for x in args.position.split(',')]
        if len(position) != 3:
            logger.error("Position must be specified as x,y,z")
            return
    except ValueError:
        logger.error("Invalid position format. Must be x,y,z (e.g., 100,100,50)")
        return
    
    # Parse normal
    try:
        normal = [float(x) for x in args.normal.split(',')]
        if len(normal) != 3:
            logger.error("Normal must be specified as x,y,z")
            return
    except ValueError:
        logger.error("Invalid normal format. Must be x,y,z (e.g., 0,0,1)")
        return
    
    # Parse range if provided
    value_range = None
    if args.range:
        try:
            value_range = [float(x) for x in args.range.split(',')]
            if len(value_range) != 2:
                logger.error("Range must be specified as min,max")
                return
            logger.info(f"Using custom range: {value_range[0]} to {value_range[1]}")
        except ValueError:
            logger.error("Invalid range format. Must be min,max (e.g., 0,1)")
            return
    
    # Generate default output filename if no output specified
    output_file = args.output
    if not output_file:
        base_name = os.path.basename(args.filename)
        output_file = f"{base_name}_s{args.substrate}_slice.png"
        logger.info(f"Using default output filename: {output_file}")
    
    # Create the visualization
    try:
        fig = create_slice_visualization(
            filename=args.filename,
            position=position,
            normal=normal,
            substrate_index=args.substrate,
            contour_value=args.contour,
            colormap=args.colormap,
            value_range=value_range,
            output_file=output_file,
            fig_size=(10, 8),
            dpi=args.dpi,
            logger=logger
        )
        
        if fig:
            logger.info(f"Figure saved to {output_file}")
            logger.info("Visualization complete")
            
            # Only try to display if explicitly requested (not the default)
            if not args.no_display:
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