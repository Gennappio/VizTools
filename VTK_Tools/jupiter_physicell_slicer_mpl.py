#!/usr/bin/env python3
"""
Jupiter PhysiCell Slicer (Matplotlib version): A Jupyter-friendly utility to create slices 
of PhysiCell output data using matplotlib.

Usage in Jupyter notebook:
    %run jupiter_physicell_slicer_mpl.py
    # or
    from jupiter_physicell_slicer_mpl import create_slice_visualization
    fig = create_slice_visualization()
    
All parameters are set at the top of this file.
"""

import os
import sys
import numpy as np
import scipy.io as sio
import xml.etree.ElementTree as ET
import logging
import datetime
import math
from scipy.interpolate import griddata

# Matplotlib imports
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# IPython and Jupyter imports
from IPython.display import display
import ipywidgets as widgets

# =============================================================================
# SET PARAMETERS HERE
# =============================================================================

# Main file parameters
FILENAME = "./sample/output00000000"  # Path to PhysiCell output (will be overridden by file browser)
ENABLE_LOGGING = True                          # Enable logging to output_jupiter_mpl.log

# Slice parameters
POSITION = [500, 0, 0]                         # Position of the slice plane [x, y, z]
NORMAL = [0, 0, 1]                             # Normal vector of the slice plane [x, y, z]
SUBSTRATE_INDEX = 11                           # Index of substrate to visualize
CONTOUR_VALUE = 0.02                           # Value to draw contour at (or None for no contour)

# Visualization parameters
COLORMAP = "jet"                               # Colormap: "viridis", "jet", "rainbow", "plasma"
RANGE = None                                   # Value range [min, max] or None for auto-range
SHOW_GRID = True                               # Show grid lines on the slice
SHOW_AXES = True                               # Show orientation axes
SHOW_COLORBAR = True                           # Show colorbar (colormap legend)

# Output parameters
FIG_WIDTH = 10                                 # Width of the figure in inches
FIG_HEIGHT = 8                                 # Height of the figure in inches
FIG_DPI = 100                                  # DPI of the figure
BACKGROUND_COLOR = 'white'                     # Background color

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_enabled):
    """Configure logging to file and console."""
    logger = logging.getLogger('jupiter_physicell_mpl')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    if logger.handlers:
        logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    if log_enabled:
        # Create file handler
        file_handler = logging.FileHandler('output_jupiter_mpl.log', mode='w')
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
        logger.info(f"Jupiter PhysiCell Slicer (Matplotlib) started at {datetime.datetime.now()}")
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
    
    # Normalize the normal vector
    normal = np.array(normal)
    normal = normal / np.linalg.norm(normal)
    
    # Create two orthogonal vectors in the plane
    if np.abs(normal[0]) < np.abs(normal[1]) and np.abs(normal[0]) < np.abs(normal[2]):
        v1 = np.array([0, -normal[2], normal[1]])
    elif np.abs(normal[1]) < np.abs(normal[0]) and np.abs(normal[1]) < np.abs(normal[2]):
        v1 = np.array([-normal[2], 0, normal[0]])
    else:
        v1 = np.array([-normal[1], normal[0], 0])
    
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(normal, v1)
    
    # Determine appropriate scale based on bounds
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    scale = max(xmax - xmin, ymax - ymin, zmax - zmin) / 2
    
    # Generate points in the plane
    u = np.linspace(-scale, scale, num_points)
    v = np.linspace(-scale, scale, num_points)
    u_grid, v_grid = np.meshgrid(u, v)
    
    # Convert to 3D coordinates
    position = np.array(position)
    x_grid = position[0] + u_grid * v1[0] + v_grid * v2[0]
    y_grid = position[1] + u_grid * v1[1] + v_grid * v2[1]
    z_grid = position[2] + u_grid * v1[2] + v_grid * v2[2]
    
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
        logger.info(f"Grid bounds: x: [{x_min}, {x_max}], y: [{y_min}, {y_max}], z: [{z_min}, {z_max}]")
        logger.info(f"Substrate range: {data_min} to {data_max}")
    
    # Create a grid of points on the slice plane
    x_grid, y_grid, z_grid = compute_plane_points(position, normal, bounds, num_points=200, logger=logger)
    
    # Create a list of all 3D points in the data
    points_3d = positions.T
    
    # Interpolate the substrate values onto the slice plane
    slice_values = griddata(points_3d, substrate, (x_grid, y_grid, z_grid), method='linear')
    
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

def create_slice_visualization(filename=None, position=None, normal=None, substrate_index=None, 
                              contour_value=None, colormap=None, value_range=None, logger=None):
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
        logger: logger object
        
    Returns:
        matplotlib figure with the visualization
    """
    # Use parameters from function call if provided, else use global defaults
    filename = filename or FILENAME
    position = position or POSITION
    normal = normal or NORMAL
    substrate_index = substrate_index if substrate_index is not None else SUBSTRATE_INDEX
    contour_value = contour_value if contour_value is not None else CONTOUR_VALUE
    colormap = colormap or COLORMAP
    value_range = value_range or RANGE
    
    # Ensure logger is initialized
    if logger is None:
        logger = setup_logging(ENABLE_LOGGING)
    
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
    
    # Use custom range if provided, otherwise use auto-range
    if value_range:
        min_val, max_val = value_range
    else:
        min_val, max_val = scalar_range
        logger.info(f"Using auto range: {min_val} to {max_val}")
        
    # Create a new figure
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=FIG_DPI, facecolor=BACKGROUND_COLOR)
    
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
    
    # Plot the slice data as a filled contour plot
    im = ax.pcolormesh(h_axis, v_axis, slice_values, 
                      cmap=colormap, 
                      vmin=min_val, vmax=max_val,
                      shading='auto')
    
    # Add contour lines if requested
    if contour_value is not None:
        contour = ax.contour(h_axis, v_axis, slice_values, 
                           levels=[contour_value], 
                           colors='black', 
                           linewidths=2)
        logger.info(f"Added contour at value {contour_value}")
    
    # Add colorbar
    if SHOW_COLORBAR:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label(substrate_name)
    
    # Add grid if requested
    if SHOW_GRID:
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
    
    return fig

# =============================================================================
# FILE BROWSER WIDGET
# =============================================================================

def create_file_browser():
    """Create a file browser widget to select PhysiCell output files."""
    import os
    
    # Create file browser widget
    file_browser = widgets.Text(
        value=FILENAME,
        placeholder='Enter path to PhysiCell output file (without extension)',
        description='File path:',
        disabled=False,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='100%')
    )
    
    # Create a button to open a dialog
    browse_button = widgets.Button(
        description='Refresh',
        disabled=False,
        button_style='info',
        tooltip='Check if files exist at the specified path',
        icon='refresh'
    )
    
    # Create output area for status messages
    status_output = widgets.Output()
    
    # Function to update status
    def update_status(b=None):
        status_output.clear_output()
        with status_output:
            path = file_browser.value
            print(f"Checking for PhysiCell files at: {path}")
            
            # Check if the files exist
            files = find_output_files(path)
            
            if not any(files.values()):
                print("❌ No PhysiCell output files found at this path")
                print("Expected files: .xml, _cells.mat, _microenvironment.mat")
                if os.path.exists(os.path.dirname(path)):
                    print("\nFiles in directory:")
                    try:
                        for f in os.listdir(os.path.dirname(path) or '.'):
                            if 'output' in f:
                                print(f"  {f}")
                    except:
                        pass
            else:
                if files['xml_file']:
                    print(f"✓ Found XML file: {os.path.basename(files['xml_file'])}")
                else:
                    print(f"❌ XML file not found")
                    
                if files['cells_file']:
                    print(f"✓ Found cells file: {os.path.basename(files['cells_file'])}")
                else:
                    print(f"❌ Cells file not found")
                    
                if files['microenv_file']:
                    print(f"✓ Found microenvironment file: {os.path.basename(files['microenv_file'])}")
                else:
                    print(f"❌ Microenvironment file not found")
    
    # Connect button click to function
    browse_button.on_click(update_status)
    
    # Run initial check
    update_status()
    
    # Create help text
    help_text = widgets.HTML(
        value="""<div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 10px; font-size: 0.9em;">
        <p><b>How to specify PhysiCell files:</b></p>
        <p>Enter the path to your PhysiCell output file <b>without extension</b>.</p>
        <p>Example: If you have files like "output00000060.xml", "output00000060_cells.mat", enter <code>path/to/output00000060</code></p>
        </div>"""
    )
    
    return widgets.VBox([
        widgets.HBox([file_browser, browse_button]),
        status_output,
        help_text
    ]), file_browser

# =============================================================================
# INTERACTIVE FUNCTIONS FOR JUPYTER
# =============================================================================

def interactive_visualization(filename=None):
    """
    Create interactive widgets for the visualization in a notebook.
    
    Returns:
        ipywidgets layout for interactive visualization
    """
    # Create file browser
    file_browser_widget, file_path_input = create_file_browser()
    
    # Set initial filename
    filename = filename or FILENAME
    file_path_input.value = filename
    
    # Create substrate dropdown with a placeholder (will be populated when files are found)
    substrate_dropdown = widgets.Dropdown(
        options=[("No substrates found", 0)],
        value=0,
        description='Substrate:',
        disabled=True
    )
    
    # Create sliders for position
    pos_x = widgets.FloatSlider(value=POSITION[0], min=-1000, max=1000, step=10, description='Position X:')
    pos_y = widgets.FloatSlider(value=POSITION[1], min=-1000, max=1000, step=10, description='Position Y:')
    pos_z = widgets.FloatSlider(value=POSITION[2], min=-1000, max=1000, step=10, description='Position Z:')
    
    # Create dropdown for normal vector
    normal_dropdown = widgets.Dropdown(
        options=[('X', [1, 0, 0]), ('Y', [0, 1, 0]), ('Z', [0, 0, 1])],
        value=NORMAL,
        description='Normal:'
    )
    
    # Create slider for contour value
    contour_slider = widgets.FloatSlider(
        value=CONTOUR_VALUE if CONTOUR_VALUE is not None else 0.02, 
        min=0, max=1, step=0.01, 
        description='Contour:'
    )
    
    # Checkbox to enable/disable contour
    contour_checkbox = widgets.Checkbox(
        value=CONTOUR_VALUE is not None,
        description='Show Contour'
    )
    
    # Create button to update the visualization
    update_button = widgets.Button(description='Update Visualization', 
                                 button_style='success',
                                 icon='image')
    output = widgets.Output()
    
    # Function to update substrate dropdown when a valid file is selected
    def update_substrate_dropdown(change=None):
        # Get current filename
        current_filename = file_path_input.value
        
        # Find files
        files = find_output_files(current_filename)
        
        # Only update if we found a microenvironment file
        if files['microenv_file']:
            # Parse substrate names
            substrates = parse_substrates_from_xml(files['xml_file']) or []
            
            # If no substrates found, use numeric indices
            if not substrates:
                max_idx = max(20, SUBSTRATE_INDEX + 5)
                substrates = [f"substrate_{i}" for i in range(max_idx)]
            
            # Create substrate options
            substrate_options = [(s, i) for i, s in enumerate(substrates)]
            
            # Get default value
            default_value = min(SUBSTRATE_INDEX, len(substrates) - 1)
            
            # Update dropdown
            substrate_dropdown.options = substrate_options
            substrate_dropdown.value = default_value
            substrate_dropdown.disabled = False
    
    # Connect file input to substrate update
    file_path_input.observe(update_substrate_dropdown, names='value')
    
    # Define update function
    def update_viz(b):
        with output:
            output.clear_output()
            
            try:
                # Get current filename
                current_filename = file_path_input.value
                
                # Show progress
                print("Loading data and creating visualization...")
                
                # Create visualization with selected parameters
                position = [pos_x.value, pos_y.value, pos_z.value]
                contour_value = contour_slider.value if contour_checkbox.value else None
                fig = create_slice_visualization(
                    filename=current_filename,
                    position=position,
                    normal=normal_dropdown.value,
                    substrate_index=substrate_dropdown.value,
                    contour_value=contour_value
                )
                
                if fig:
                    plt.show()
                else:
                    print("❌ Error creating visualization.")
                    print("Check that all required files exist:")
                    files = find_output_files(current_filename)
                    if not files['microenv_file']:
                        print(f"  - Missing microenvironment file for {current_filename}")
                    print("\nSee output_jupiter_mpl.log for detailed error information.")
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                print("See output_jupiter_mpl.log for detailed error information.")
    
    update_button.on_click(update_viz)
    
    # Create the UI layout
    position_box = widgets.HBox([pos_x, pos_y, pos_z])
    contour_box = widgets.HBox([contour_checkbox, contour_slider])
    
    # Organize controls in sections
    file_section = widgets.VBox([
        widgets.HTML("<h3>PhysiCell File Selection</h3>"),
        file_browser_widget
    ])
    
    viz_controls = widgets.VBox([
        widgets.HTML("<h3>Visualization Controls</h3>"),
        substrate_dropdown,
        widgets.HTML("<b>Slice Position and Orientation:</b>"),
        position_box, 
        normal_dropdown, 
        widgets.HTML("<b>Contour Settings:</b>"),
        contour_box,
        update_button
    ])
    
    # Return complete layout
    return widgets.VBox([file_section, viz_controls, output])

# =============================================================================
# MAIN FUNCTION - WILL RUN IF SCRIPT IS EXECUTED DIRECTLY
# =============================================================================

def main():
    """Main function to run the Jupiter PhysiCell slicer with matplotlib."""
    logger = setup_logging(ENABLE_LOGGING)
    
    logger.info("Starting Jupiter PhysiCell Slicer (Matplotlib version)")
    logger.info(f"Parameters:")
    logger.info(f"  Filename: {FILENAME}")
    logger.info(f"  Position: {POSITION}")
    logger.info(f"  Normal: {NORMAL}")
    logger.info(f"  Substrate index: {SUBSTRATE_INDEX}")
    logger.info(f"  Contour value: {CONTOUR_VALUE}")
    
    # Create the visualization
    fig = create_slice_visualization(logger=logger)
    
    if fig:
        # If running in a Jupyter environment, display the figure
        try:
            from IPython import get_ipython
            if get_ipython() is not None:
                plt.show()
            else:
                # If running from command line, save the figure
                plt.savefig('physicell_slice.png')
                logger.info("Saved visualization to physicell_slice.png")
                plt.show()
        except ImportError:
            # Not in an IPython environment, save the figure
            plt.savefig('physicell_slice.png')
            logger.info("Saved visualization to physicell_slice.png")
            plt.show()
    else:
        logger.error("Failed to create visualization")

if __name__ == "__main__":
    main()

# When running in Jupyter with the "play" button, auto-generate a visualization
try:
    # Check if we're in a Jupyter environment
    from IPython import get_ipython
    jupyter_env = get_ipython() is not None
    if jupyter_env:
        # Try to detect if this is a fresh execution (not an import)
        import inspect
        if not inspect.stack()[1].filename.endswith('importlib/_bootstrap.py'):
            print("Jupyter environment detected. Generating visualization...")
            
            # Create interactive controls for visualization
            vis_widget = interactive_visualization()
            
            # Display the controls
            display(vis_widget)
            print("Use the controls above to select a file and create a visualization.")
            print("If you don't see any visualization, check if the file path is correct.")
except ImportError:
    # Not running in IPython/Jupyter
    pass 