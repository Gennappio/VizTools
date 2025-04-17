#!/usr/bin/env python3
"""
Jupiter PhysiCell Slicer: A Jupyter-friendly utility to create slices of PhysiCell output data.

Usage in Jupyter notebook:
    %run jupiter_physicell_slicer.py
    # or
    from jupiter_physicell_slicer import create_slice_visualization
    renderer, window = create_slice_visualization()
    
All parameters are set at the top of this file.
"""

import os
import sys
import numpy as np
import scipy.io as sio
import vtk
import xml.etree.ElementTree as ET
import logging
import datetime
import math
from IPython.display import Image
import ipywidgets as widgets

# =============================================================================
# SET PARAMETERS HERE
# =============================================================================

# Main file parameters
FILENAME = "../../../PhysiCell_micro/PhysiCell/output/output00000007"  # Path to PhysiCell output
ENABLE_LOGGING = True                          # Enable logging to output_jupiter.log

# Slice parameters
POSITION = [500, 0, 0]                         # Position of the slice plane [x, y, z]
NORMAL = [0, 0, 1]                             # Normal vector of the slice plane [x, y, z]
SUBSTRATE_INDEX = 11                           # Index of substrate to visualize
CONTOUR_VALUE = 0.02                           # Value to draw contour at (or None for no contour)

# Visualization parameters
COLORMAP = "jet"                               # Colormap: "default", "jet", "rainbow", "viridis"
RANGE = None                                   # Value range [min, max] or None for auto-range
SHOW_GRID = True                               # Show grid lines on the slice
SHOW_AXES = True                               # Show orientation axes
SHOW_SCALAR_BAR = True                         # Show scalar bar (colormap legend)

# Output parameters
IMAGE_WIDTH = 800                              # Width of the output image in pixels
IMAGE_HEIGHT = 600                             # Height of the output image in pixels
BACKGROUND_COLOR = [1, 1, 1]                   # Background color [r, g, b], white=[1,1,1]

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_enabled):
    """Configure logging to file and console."""
    logger = logging.getLogger('jupiter_physicell')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    if logger.handlers:
        logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    if log_enabled:
        # Create file handler
        file_handler = logging.FileHandler('output_jupiter.log', mode='w')
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
        logger.info(f"Jupiter PhysiCell Slicer started at {datetime.datetime.now()}")
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
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_colormap(min_val, max_val, colormap_name="default"):
    """Create a color transfer function for the slice coloring."""
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(256)
    
    if colormap_name == "jet":
        # Jet colormap (blue -> cyan -> green -> yellow -> red)
        lut.SetHueRange(0.667, 0.0)
        lut.SetSaturationRange(1.0, 1.0)
        lut.SetValueRange(1.0, 1.0)
    elif colormap_name == "rainbow":
        # Rainbow colormap
        lut.SetHueRange(0.0, 0.667)
        lut.SetSaturationRange(1.0, 1.0)
        lut.SetValueRange(1.0, 1.0)
    elif colormap_name == "viridis":
        # Viridis-like colormap
        for i in range(256):
            t = i / 255.0
            if t < 0.25:
                r, g, b = 0.267, 0.004, 0.329
            elif t < 0.5:
                r, g, b = 0.270, 0.313, 0.612
            elif t < 0.75:
                r, g, b = 0.204, 0.553, 0.573
            else:
                r, g, b = 0.992, 0.906, 0.144
            lut.SetTableValue(i, r, g, b, 1.0)
    else:  # Default
        # Default blue to red colormap
        lut.SetHueRange(0.667, 0.0)
        lut.SetSaturationRange(1.0, 1.0)
        lut.SetValueRange(1.0, 1.0)
    
    lut.SetRange(min_val, max_val)
    lut.Build()
    return lut

def create_scalar_bar(lut, title="Substrate Concentration"):
    """Create a scalar bar (legend) for the colormap."""
    scalar_bar = vtk.vtkScalarBarActor()
    scalar_bar.SetLookupTable(lut)
    scalar_bar.SetTitle(title)
    scalar_bar.SetNumberOfLabels(5)
    scalar_bar.SetOrientationToVertical()
    scalar_bar.SetPosition(0.85, 0.1)
    scalar_bar.SetWidth(0.1)
    scalar_bar.SetHeight(0.8)
    
    # Set text properties
    prop = scalar_bar.GetTitleTextProperty()
    prop.SetColor(0, 0, 0)
    prop.SetFontFamilyToArial()
    prop.SetFontSize(12)
    prop.SetBold(1)
    
    label_prop = scalar_bar.GetLabelTextProperty()
    label_prop.SetColor(0, 0, 0)
    label_prop.SetFontFamilyToArial()
    label_prop.SetFontSize(10)
    
    return scalar_bar

def create_slice(microenv_data, position, normal, substrate_idx, logger=None):
    """Create a slice of the microenvironment data."""
    if microenv_data is None:
        if logger:
            logger.error("No microenvironment data to slice")
        return None, None
    
    if logger:
        logger.info(f"Creating slice at position {position} with normal {normal}")
    
    # Extract positions (first 3 rows) and substrate data
    positions = microenv_data[0:3, :]
    substrate = microenv_data[substrate_idx + 4, :]  # +4 to skip x,y,z and time
    
    # Get bounds of the data
    x_min, x_max = np.min(positions[0, :]), np.max(positions[0, :])
    y_min, y_max = np.min(positions[1, :]), np.max(positions[1, :])
    z_min, z_max = np.min(positions[2, :]), np.max(positions[2, :])
    
    # Get min and max values for the substrate
    data_min, data_max = np.min(substrate), np.max(substrate)
    
    if logger:
        logger.info(f"Grid bounds: x: [{x_min}, {x_max}], y: [{y_min}, {y_max}], z: [{z_min}, {z_max}]")
        logger.info(f"Substrate range: {data_min} to {data_max}")
    
    # Create a structured grid for the microenvironment
    # First, determine the dimensions of the grid
    # Assuming the positions are ordered in a structured way
    
    # Find unique x, y, z coordinates
    unique_x = np.unique(positions[0, :])
    unique_y = np.unique(positions[1, :])
    unique_z = np.unique(positions[2, :])
    
    # Determine dimensions
    nx, ny, nz = len(unique_x), len(unique_y), len(unique_z)
    
    if logger:
        logger.info(f"Grid dimensions: {nx} x {ny} x {nz}")
    
    # Check if it's a properly structured grid
    if nx * ny * nz != positions.shape[1]:
        if logger:
            logger.warning(f"Data points ({positions.shape[1]}) don't match expected grid size ({nx}x{ny}x{nz}={nx*ny*nz})")
            logger.warning("Will try to create an unstructured grid instead")
        
        # Create a vtkPoints object for the positions
        points = vtk.vtkPoints()
        for i in range(positions.shape[1]):
            points.InsertNextPoint(positions[0, i], positions[1, i], positions[2, i])
        
        # Create scalar array for the substrate
        scalars = vtk.vtkFloatArray()
        scalars.SetName("substrate")
        for i in range(substrate.shape[0]):
            scalars.InsertNextValue(substrate[i])
        
        # Create an unstructured grid
        grid = vtk.vtkUnstructuredGrid()
        grid.SetPoints(points)
        grid.GetPointData().SetScalars(scalars)
        
        # Create a cell for each point (could also create a voxel grid if needed)
        for i in range(positions.shape[1]):
            cell = vtk.vtkVertex()
            cell.GetPointIds().SetId(0, i)
            grid.InsertNextCell(cell.GetCellType(), cell.GetPointIds())
        
        # Create a probe filter with an implicit plane function
        plane = vtk.vtkPlane()
        plane.SetOrigin(position)
        plane.SetNormal(normal)
        
        # Create cutter
        cutter = vtk.vtkCutter()
        cutter.SetInputData(grid)
        cutter.SetCutFunction(plane)
        cutter.Update()
        
        # Create a delaunay filter to create a surface from the cut points
        delaunay = vtk.vtkDelaunay2D()
        delaunay.SetInputConnection(cutter.GetOutputPort())
        delaunay.Update()
        
        return delaunay.GetOutput(), (data_min, data_max)
    
    else:
        # It's a structured grid, which is more efficient
        # Create structured grid
        grid = vtk.vtkStructuredGrid()
        grid.SetDimensions(nx, ny, nz)
        
        # Create points
        points = vtk.vtkPoints()
        points.SetNumberOfPoints(positions.shape[1])
        
        # Create a mapping from (x,y,z) indices to flattened index
        index_map = {}
        for i in range(len(unique_x)):
            for j in range(len(unique_y)):
                for k in range(len(unique_z)):
                    # Compute flattened index (assuming C-order: z changes fastest, then y, then x)
                    flat_idx = i * (ny * nz) + j * nz + k
                    index_map[(i, j, k)] = flat_idx
        
        # Fill the points and scalar array
        scalars = vtk.vtkFloatArray()
        scalars.SetName("substrate")
        scalars.SetNumberOfValues(positions.shape[1])
        
        # Map each position to its grid index
        for i in range(positions.shape[1]):
            x, y, z = positions[0, i], positions[1, i], positions[2, i]
            
            # Find indices in the unique arrays
            idx_x = np.searchsorted(unique_x, x)
            idx_y = np.searchsorted(unique_y, y)
            idx_z = np.searchsorted(unique_z, z)
            
            # Adjust if needed
            if idx_x == len(unique_x) or unique_x[idx_x] != x:
                idx_x -= 1
            if idx_y == len(unique_y) or unique_y[idx_y] != y:
                idx_y -= 1
            if idx_z == len(unique_z) or unique_z[idx_z] != z:
                idx_z -= 1
            
            # Get flattened index
            flat_idx = index_map.get((idx_x, idx_y, idx_z), i)  # Fallback to i if not found
            
            # Set point and scalar value
            points.SetPoint(flat_idx, x, y, z)
            scalars.SetValue(flat_idx, substrate[i])
        
        grid.SetPoints(points)
        grid.GetPointData().SetScalars(scalars)
        
        # Create cutter
        plane = vtk.vtkPlane()
        plane.SetOrigin(position)
        plane.SetNormal(normal)
        
        cutter = vtk.vtkCutter()
        cutter.SetInputData(grid)
        cutter.SetCutFunction(plane)
        cutter.Update()
        
        return cutter.GetOutput(), (data_min, data_max)

def create_contour(slice_output, contour_value, logger=None):
    """Create a contour on the slice at the specified value."""
    if slice_output is None:
        if logger:
            logger.error("No slice data to create contour")
        return None
    
    if contour_value is None:
        if logger:
            logger.info("No contour value specified, skipping contour creation")
        return None
    
    if logger:
        logger.info(f"Creating contour at value {contour_value}")
        logger.info(f"Slice output has {slice_output.GetNumberOfPoints()} points and {slice_output.GetNumberOfCells()} cells")
    
    # Create banded contour filter
    contour_filter = vtk.vtkBandedPolyDataContourFilter()
    contour_filter.SetInputData(slice_output)
    contour_filter.SetNumberOfContours(1)
    contour_filter.SetValue(0, contour_value)
    contour_filter.SetScalarModeToValue()
    contour_filter.GenerateContourEdgesOn()
    contour_filter.Update()
    
    # Get the contour edges output
    contour_edges = contour_filter.GetContourEdgesOutput()
    
    if logger:
        logger.info(f"Contour has {contour_edges.GetNumberOfPoints()} points and {contour_edges.GetNumberOfCells()} cells")
    
    # Create mapper and actor
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(contour_edges)
    
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0, 0, 0)  # Black contour
    actor.GetProperty().SetLineWidth(2)
    
    return actor

# =============================================================================
# MAIN VISUALIZATION FUNCTION
# =============================================================================

def create_slice_visualization(filename=None, position=None, normal=None, substrate_index=None, 
                              contour_value=None, colormap=None, value_range=None, logger=None):
    """Create a visualization of a slice through the microenvironment data."""
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
        return None, None
    
    # Parse substrate names from XML file
    substrates = parse_substrates_from_xml(files['xml_file'], logger)
    
    # Load microenvironment data
    microenv_data = load_microenvironment_data(files['microenv_file'], logger)
    
    if microenv_data is None:
        logger.error("Failed to load microenvironment data")
        return None, None
    
    # Get substrate name
    substrate_name = get_substrate_name(substrate_index, substrates, logger)
    
    # Create a slice through the data
    slice_output, scalar_range = create_slice(microenv_data, position, normal, substrate_index, logger)
    
    if slice_output is None:
        logger.error("Failed to create slice")
        return None, None
    
    # Use custom range if provided, otherwise use auto-range
    if value_range:
        min_val, max_val = value_range
    else:
        min_val, max_val = scalar_range
        logger.info(f"Using auto range: {min_val} to {max_val}")
    
    # Create a renderer
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(*BACKGROUND_COLOR)  # White background by default
    
    # Create mapper and actor for the slice
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(slice_output)
    mapper.SetScalarRange(min_val, max_val)
    
    # Create colormap
    lut = create_colormap(min_val, max_val, colormap)
    mapper.SetLookupTable(lut)
    
    # Create actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    renderer.AddActor(actor)
    
    # Create contour if requested
    if contour_value is not None:
        contour_actor = create_contour(slice_output, contour_value, logger)
        if contour_actor:
            renderer.AddActor(contour_actor)
    
    # Add scalar bar if requested
    if SHOW_SCALAR_BAR:
        scalar_bar = create_scalar_bar(lut, substrate_name)
        renderer.AddActor(scalar_bar)
    
    # Add axes if requested
    if SHOW_AXES:
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(50, 50, 50)  # Adjust length as needed
        axes.GetXAxisCaptionActor2D().GetTextActor().SetTextScaleMode(vtk.vtkTextActor.TEXT_SCALE_MODE_NONE)
        axes.GetYAxisCaptionActor2D().GetTextActor().SetTextScaleMode(vtk.vtkTextActor.TEXT_SCALE_MODE_NONE)
        axes.GetZAxisCaptionActor2D().GetTextActor().SetTextScaleMode(vtk.vtkTextActor.TEXT_SCALE_MODE_NONE)
        
        axes_widget = vtk.vtkOrientationMarkerWidget()
        axes_widget.SetOrientationMarker(axes)
        axes_widget.SetViewport(0, 0, 0.2, 0.2)
        
        renderer.AddActor(axes)
    
    # Create render window
    render_window = vtk.vtkRenderWindow()
    render_window.SetSize(IMAGE_WIDTH, IMAGE_HEIGHT)
    render_window.SetWindowName(f"PhysiCell Slice - {substrate_name} - {os.path.basename(filename)}")
    render_window.AddRenderer(renderer)
    
    # Add text with information
    text_actor = vtk.vtkTextActor()
    text = f"File: {os.path.basename(filename)}\n"
    text += f"Position: [{position[0]}, {position[1]}, {position[2]}]\n"
    text += f"Normal: [{normal[0]}, {normal[1]}, {normal[2]}]\n"
    text += f"Substrate: {substrate_name}\n"
    text += f"Range: {min_val:.2f} to {max_val:.2f}"
    if contour_value is not None:
        text += f"\nContour at: {contour_value:.3f}"
    
    text_actor.SetInput(text)
    text_actor.GetTextProperty().SetColor(0, 0, 0)  # Black text
    text_actor.GetTextProperty().SetFontSize(12)
    text_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    text_actor.SetPosition(0.02, 0.95)
    renderer.AddActor2D(text_actor)
    
    # Log summary before returning
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
    
    return renderer, render_window

# =============================================================================
# JUPYTER NOTEBOOK FUNCTIONS
# =============================================================================

def capture_window_image(render_window, width=None, height=None):
    """Capture an image of the render window for display in a notebook."""
    width = width or IMAGE_WIDTH
    height = height or IMAGE_HEIGHT
    
    # Set up rendering for image capture
    render_window.SetOffScreenRendering(1)
    render_window.SetSize(width, height)
    render_window.Render()
    
    # Set up the window to image filter
    window_to_image = vtk.vtkWindowToImageFilter()
    window_to_image.SetInput(render_window)
    window_to_image.SetScale(1)  # Default is 1, can increase for higher resolution
    window_to_image.SetInputBufferTypeToRGB()
    window_to_image.ReadFrontBufferOff()
    window_to_image.Update()
    
    # Write image to a temporary file
    writer = vtk.vtkPNGWriter()
    writer.SetFileName("temp_slice_image.png")
    writer.SetInputConnection(window_to_image.GetOutputPort())
    writer.Write()
    
    # Clean up
    render_window.SetOffScreenRendering(0)
    
    # Return image for display in notebook
    return Image("temp_slice_image.png")

def interactive_visualization(filename=None):
    """Create interactive widgets for the visualization in a notebook."""
    filename = filename or FILENAME
    
    # Find files and get substrate names
    files = find_output_files(filename)
    substrates = parse_substrates_from_xml(files['xml_file']) or []
    
    # If no substrates found, use numeric indices
    if not substrates:
        substrates = [f"substrate_{i}" for i in range(10)]  # Default to 10 substrates
    
    # Create a dropdown for substrate selection
    substrate_dropdown = widgets.Dropdown(
        options=[(s, i) for i, s in enumerate(substrates)],
        value=SUBSTRATE_INDEX,
        description='Substrate:'
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
    update_button = widgets.Button(description='Update Visualization')
    output = widgets.Output()
    
    # Define update function
    def update_viz(b):
        with output:
            output.clear_output()
            position = [pos_x.value, pos_y.value, pos_z.value]
            contour_value = contour_slider.value if contour_checkbox.value else None
            renderer, window = create_slice_visualization(
                filename=filename,
                position=position,
                normal=normal_dropdown.value,
                substrate_index=substrate_dropdown.value,
                contour_value=contour_value
            )
            if renderer and window:
                display(capture_window_image(window))
            else:
                print("Error creating visualization. Check the log file for details.")
    
    update_button.on_click(update_viz)
    
    # Create the UI layout
    position_box = widgets.HBox([pos_x, pos_y, pos_z])
    contour_box = widgets.HBox([contour_checkbox, contour_slider])
    controls = widgets.VBox([substrate_dropdown, position_box, normal_dropdown, contour_box, update_button])
    
    return widgets.VBox([controls, output])

# =============================================================================
# MAIN FUNCTION - WILL RUN IF SCRIPT IS EXECUTED DIRECTLY
# =============================================================================

def main():
    """Main function to run the Jupiter PhysiCell slicer."""
    logger = setup_logging(ENABLE_LOGGING)
    
    logger.info("Starting Jupiter PhysiCell Slicer")
    logger.info(f"Parameters:")
    logger.info(f"  Filename: {FILENAME}")
    logger.info(f"  Position: {POSITION}")
    logger.info(f"  Normal: {NORMAL}")
    logger.info(f"  Substrate index: {SUBSTRATE_INDEX}")
    logger.info(f"  Contour value: {CONTOUR_VALUE}")
    
    # Create the visualization
    renderer, window = create_slice_visualization(logger=logger)
    
    if renderer and window:
        # If running in a Jupyter environment, return an image
        try:
            from IPython import get_ipython
            if get_ipython() is not None:
                return capture_window_image(window)
            else:
                # If running from command line, create interactor
                interactor = vtk.vtkRenderWindowInteractor()
                interactor.SetRenderWindow(window)
                interactor.Initialize()
                window.Render()
                interactor.Start()
        except ImportError:
            # Not in an IPython environment, create interactor
            interactor = vtk.vtkRenderWindowInteractor()
            interactor.SetRenderWindow(window)
            interactor.Initialize()
            window.Render()
            interactor.Start()
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
            
            # Run initial visualization with default parameters
            logger = setup_logging(ENABLE_LOGGING)
            renderer, window = create_slice_visualization(logger=logger)
            if renderer and window:
                image = capture_window_image(window)
                from IPython.display import display
                display(vis_widget)  # Show the interactive controls
                display(image)       # Show the initial visualization
                print("Use the controls above to adjust the visualization parameters.")
                print("Click 'Update Visualization' after making changes.")
            else:
                print("Error creating visualization. Check the log file for details.")
except ImportError:
    # Not running in IPython/Jupyter
    pass 