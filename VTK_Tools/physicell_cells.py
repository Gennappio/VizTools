#!/usr/bin/env python3
"""
PhysiCell Cells Visualizer: A utility to display cells from PhysiCell simulations using VTK.

Usage:
    python physicell_cells.py --filename <filename> [--log] [--scalar cell_attribute] [--range min,max] [--colormap name]

Parameters:
    --filename:     Root name of PhysiCell output files (without extension)
                    (e.g. 'output0000060' will load output0000060.xml, output0000060_cells.mat, etc.)
    --log:          Enable logging to understand what is going on (writes to output.log)
    --scalar:       Cell attribute to use for coloring (e.g. "cycle_model", "volume", "pressure")
                    Default: "cycle_model"
    --range:        User-defined range of values (min,max) to display in the colormap
                    If not provided, auto-range will be used
    --colormap:     Colormap to use (default, jet, rainbow, viridis)
                    Default: "default"
"""

import os
import sys
import argparse
import numpy as np
import scipy.io as sio
import vtk
import xml.etree.ElementTree as ET
import logging
import datetime
import math


# Set up logging
def setup_logging(log_enabled):
    """Configure logging to file and console."""
    logger = logging.getLogger('physicell_cells')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    if logger.handlers:
        logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    if log_enabled:
        # Create file handler
        file_handler = logging.FileHandler('output_cells.log', mode='w')
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
        logger.info(f"PhysiCell Cells Visualizer started at {datetime.datetime.now()}")
        logger.info('-' * 70)
    
    return logger


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


def load_cells_data(cells_file, logger=None):
    """Load cell data from a .mat file."""
    if logger:
        logger.info(f"Loading cells data from {cells_file}")
    
    try:
        mat_contents = sio.loadmat(cells_file)
        mat_contents = {k:v for k, v in mat_contents.items() 
                       if not k.startswith('__')}
        
        # Get the cells data
        if 'cells' in mat_contents:
            cells_data = mat_contents['cells']
            
            if logger:
                logger.info(f"Cells data shape: {cells_data.shape}")
                logger.info(f"Number of cells: {cells_data.shape[1]}")
                logger.info(f"Number of cell attributes: {cells_data.shape[0]}")
                
                # Log info about first few cells
                num_sample = min(5, cells_data.shape[1])
                logger.info(f"Sample of first {num_sample} cells:")
                for i in range(num_sample):
                    logger.info(f"Cell {i} raw data: {[cells_data[j, i] for j in range(min(10, cells_data.shape[0]))]}")
            
            return cells_data
        
        return None
        
    except Exception as e:
        if logger:
            logger.error(f"Error loading cells data: {e}")
            import traceback
            logger.error(traceback.format_exc())
        return None


def extract_cell_attribute_indices(cells_data, logger=None):
    """Extract indices of different cell attributes in the cells array."""
    # Try to dynamically detect cell attributes from matrices.xml file
    # However, also provide fallback mappings for common attributes
    
    # Common default indices based on PhysiCell cell data format
    # These are standard in PhysiCell outputs but may vary in different versions
    attribute_indices = {
        "ID": 0,            # Cell ID
        "position_x": 1,    # x position
        "position_y": 2,    # y position
        "position_z": 3,    # z position
        "total_volume": 4,  # Cell volume
        "cell_type": 5,     # Cell type
        "cycle_model": 6,   # Cell cycle model
        "current_phase": 7, # Current phase
        "elapsed_time": 8,  # Elapsed time in phase
        "nuclear_volume": 9, # Nuclear volume
        "cytoplasmic_volume": 10, # Cytoplasmic volume
        "fluid_fraction": 11,   # Fluid fraction
        "calcified_fraction": 12, # Calcified fraction
        "orientation_x": 13,    # Orientation x
        "orientation_y": 14,    # Orientation y
        "orientation_z": 15,    # Orientation z
        "polarity": 16,         # Polarity
        "migration_speed": 17,  # Migration speed
        "motility_vector_x": 18, # Motility vector x
        "motility_vector_y": 19, # Motility vector y
        "motility_vector_z": 20, # Motility vector z
        "migration_bias": 21,   # Migration bias direction
        "motility_reserved": 22, # Reserved for future motility
        "adhesive_affinities": 23, # Adhesive affinities
        "dead_phagocytosis": 24, # Dead phagocytosis
        "cell_velocity_x": 25,  # Cell velocity x
        "cell_velocity_y": 26,  # Cell velocity y
        "cell_velocity_z": 27,  # Cell velocity z
        "pressure": 28,         # Pressure
        
        # Aliases for convenience
        "volume": 4,            # Alias for total_volume
    }
    
    # Add dynamic indices for custom attributes
    # We'll check the first cell for non-zero values beyond standard indices
    if cells_data.shape[0] > 29:  # If we have more than standard attributes
        for idx in range(29, cells_data.shape[0]):
            # Try to identify custom attributes by checking values
            # This is a heuristic approach since we don't have exact mappings
            if logger:
                logger.info(f"Custom attribute at index {idx}, sample values: {cells_data[idx, 0:min(5, cells_data.shape[1])]}")
            
            # Generate a generic name for unmapped attributes
            attribute_name = f"custom_attribute_{idx}"
            attribute_indices[attribute_name] = idx
            
            # Check if it might be ATP-related by looking at the value pattern
            # This is just a guess - would need more info from PhysiCell to be certain
            if "ATP" not in attribute_indices and idx >= 29:
                attribute_indices["cell_ATP_source"] = idx
    
    # For debugging - print actual values for first cell
    if logger:
        logger.info("Debugging first cell values:")
        for attr, idx in sorted(attribute_indices.items(), key=lambda x: x[1]):
            if idx < cells_data.shape[0]:
                logger.info(f"  {attr} (index {idx}): {cells_data[idx, 0]}")
    
    # Log available attributes
    if logger:
        logger.info("Available cell attributes:")
        for attr, idx in sorted(attribute_indices.items(), key=lambda x: x[1]):
            if idx < cells_data.shape[0]:  # Check if this attribute exists in the data
                # Get values range for better understanding
                values = cells_data[idx, :]
                min_val = np.min(values)
                max_val = np.max(values)
                mean_val = np.mean(values)
                
                logger.info(f"  {attr}: index {idx}, range {min_val:.3f} to {max_val:.3f}, mean {mean_val:.3f}")
    
    return attribute_indices


def create_colormap(min_val, max_val, colormap_name="default"):
    """Create a color transfer function for the cells coloring."""
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


def create_scalar_bar(lut, title="Cell Property"):
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


def calculate_cell_radius(volume):
    """Calculate the radius of a cell based on its volume, assuming spherical shape."""
    # V = (4/3) * π * r³
    # r = (3V / 4π)^(1/3)
    return (3.0 * volume / (4.0 * math.pi)) ** (1.0/3.0)


def create_cell_visualization(cells_data, scalar_attribute, attribute_indices, logger=None):
    """Create a VTK visualization of cells with scalar coloring."""
    if cells_data is None or cells_data.shape[1] == 0:
        if logger:
            logger.error("No cell data to visualize")
        return None, None, None
    
    if logger:
        logger.info(f"Creating visualization for {cells_data.shape[1]} cells")
        logger.info(f"Using {scalar_attribute} for coloring")
    
    # First check if scalar attribute is a direct index
    try:
        scalar_idx = int(scalar_attribute)
        if scalar_idx >= 0 and scalar_idx < cells_data.shape[0]:
            if logger:
                logger.info(f"Using numeric index {scalar_idx} for scalar coloring")
        else:
            logger.warning(f"Invalid scalar index {scalar_idx}. Using cell_type as default.")
            scalar_idx = attribute_indices.get("cell_type", 5)
    except ValueError:
        # Not a numeric index, check named attributes
        if scalar_attribute in attribute_indices and attribute_indices[scalar_attribute] < cells_data.shape[0]:
            scalar_idx = attribute_indices[scalar_attribute]
            if logger:
                logger.info(f"Using attribute '{scalar_attribute}' (index {scalar_idx}) for coloring")
        else:
            if logger:
                logger.warning(f"Scalar attribute '{scalar_attribute}' not found. Using 'cell_type' as default.")
            scalar_idx = attribute_indices.get("cell_type", 5)
    
    # Extract scalar values for coloring
    scalar_values = cells_data[scalar_idx, :]
    
    # Calculate min and max values for the colormap
    min_val = np.min(scalar_values)
    max_val = np.max(scalar_values)
    
    if logger:
        logger.info(f"Scalar range: {min_val} to {max_val}")
    
    # Debug position data
    if logger:
        pos_x_idx = attribute_indices.get("position_x", 1)
        pos_y_idx = attribute_indices.get("position_y", 2)
        pos_z_idx = attribute_indices.get("position_z", 3)
        
        # Check if all positions are zero
        all_x_zero = np.all(cells_data[pos_x_idx, :] == 0)
        all_y_zero = np.all(cells_data[pos_y_idx, :] == 0)
        all_z_zero = np.all(cells_data[pos_z_idx, :] == 0)
        
        if all_x_zero and all_y_zero and all_z_zero:
            logger.error("WARNING: All cell positions are zero! This suggests a data issue.")
            # Print the first few rows of data to debug
            for i in range(min(10, cells_data.shape[0])):
                logger.info(f"Row {i}: {cells_data[i, 0:5]}")
        
        # Log position stats
        logger.info(f"Position X range: {np.min(cells_data[pos_x_idx, :])} to {np.max(cells_data[pos_x_idx, :])}")
        logger.info(f"Position Y range: {np.min(cells_data[pos_y_idx, :])} to {np.max(cells_data[pos_y_idx, :])}")
        logger.info(f"Position Z range: {np.min(cells_data[pos_z_idx, :])} to {np.max(cells_data[pos_z_idx, :])}")
    
    # Create points for the cells
    points = vtk.vtkPoints()
    points.SetNumberOfPoints(cells_data.shape[1])
    
    # Create cell array for the vertices
    vertices = vtk.vtkCellArray()
    
    # Create scalar array for coloring
    scalars = vtk.vtkFloatArray()
    scalars.SetName(str(scalar_attribute))
    scalars.SetNumberOfValues(cells_data.shape[1])
    
    # Create array for cell sizes
    sizes = vtk.vtkFloatArray()
    sizes.SetName("Sizes")
    sizes.SetNumberOfValues(cells_data.shape[1])
    
    # Get indices for position and volume
    pos_x_idx = attribute_indices.get("position_x", 1)
    pos_y_idx = attribute_indices.get("position_y", 2)
    pos_z_idx = attribute_indices.get("position_z", 3)
    volume_idx = attribute_indices.get("volume", 4)
    
    # Process each cell
    for i in range(cells_data.shape[1]):
        # Get cell position
        x = cells_data[pos_x_idx, i]
        y = cells_data[pos_y_idx, i]
        z = cells_data[pos_z_idx, i]
        
        # Add point
        points.SetPoint(i, x, y, z)
        
        # Create vertex
        vertex = vtk.vtkVertex()
        vertex.GetPointIds().SetId(0, i)
        vertices.InsertNextCell(vertex)
        
        # Set scalar value
        scalars.SetValue(i, scalar_values[i])
        
        # Calculate cell radius from volume
        volume = cells_data[volume_idx, i]
        radius = calculate_cell_radius(volume)
        sizes.SetValue(i, radius)
    
    # Create polydata for the cells
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetVerts(vertices)
    polydata.GetPointData().SetScalars(scalars)
    polydata.GetPointData().AddArray(sizes)
    
    # Setup glyph for cell visualization
    sphere_source = vtk.vtkSphereSource()
    sphere_source.SetRadius(1.0)  # Base radius of 1.0, will be scaled
    sphere_source.SetPhiResolution(16)
    sphere_source.SetThetaResolution(16)
    
    # Create glyph3D to represent cells as spheres
    glyph3D = vtk.vtkGlyph3D()
    
    # Make sizes the active scalars for scaling
    polydata.GetPointData().SetActiveScalars("Sizes")
    
    # Use cells as input for the glyph filter
    glyph3D.SetInputData(polydata)
    glyph3D.SetSourceConnection(sphere_source.GetOutputPort())
    
    # Configure scaling - this is compatible with older VTK versions
    glyph3D.SetScaleModeToScaleByScalar()
    
    # For VTK versions that don't have SetScaleArray, use InputArrayToProcess instead
    glyph3D.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "Sizes")
    
    # Ensure scalar coloring is preserved
    polydata.GetPointData().SetActiveScalars(str(scalar_attribute))
    
    glyph3D.Update()
    
    # Create mapper for the cells
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(glyph3D.GetOutputPort())
    mapper.SetScalarRange(min_val, max_val)
    mapper.ScalarVisibilityOn()
    mapper.SetScalarModeToUsePointFieldData()
    mapper.SelectColorArray(str(scalar_attribute))
    
    # Create actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    
    return actor, mapper, (min_val, max_val)


def create_axes():
    """Create axes actor for orientation."""
    axes = vtk.vtkAxesActor()
    
    # Set labels
    axes.SetXAxisLabelText("X")
    axes.SetYAxisLabelText("Y")
    axes.SetZAxisLabelText("Z")
    
    # Set label font size
    axes.GetXAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
    axes.GetYAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
    axes.GetZAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
    
    axes.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(12)
    axes.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(12)
    axes.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(12)
    
    return axes


def main():
    """Main function to run the PhysiCell cells visualizer."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='PhysiCell Cells Visualizer')
    parser.add_argument('--filename', required=True, help='Root name of PhysiCell output files')
    parser.add_argument('--log', action='store_true', help='Enable logging to output_cells.log')
    parser.add_argument('--scalar', type=str, default='cycle_model', help='Cell attribute to use for coloring (name or numeric index)')
    parser.add_argument('--range', type=str, help='Value range min,max (e.g. 0,1)')
    parser.add_argument('--colormap', type=str, default='default', help='Colormap to use (default, jet, rainbow, viridis)')
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(args.log)
    
    # Set VTK window backend
    # Force VTK to use the native window mode (not Cocoa)
    os.environ["VTK_RENDERER"] = "OpenGL"
    
    # Process arguments
    if args.log:
        logger.info(f"Processing PhysiCell output with root: {args.filename}")
        logger.info(f"Command line arguments: {args}")
    
    # Parse range if provided
    custom_range = None
    if args.range:
        try:
            custom_range = [float(x) for x in args.range.split(',')]
            if len(custom_range) != 2:
                logger.error("Range must be specified as min,max")
                return
            logger.info(f"Using custom range: {custom_range[0]} to {custom_range[1]}")
        except ValueError:
            logger.error("Invalid range format. Must be min,max (e.g. 0,1)")
            return
    
    # Find output files
    files = find_output_files(args.filename, logger)
    
    # Check if cells file exists
    if not files['cells_file']:
        logger.error(f"No cells file found for {args.filename}")
        return
    
    # Load cells data
    cells_data = load_cells_data(files['cells_file'], logger)
    
    if cells_data is None:
        logger.error("Failed to load cells data")
        return
    
    # Extract cell attribute indices
    attribute_indices = extract_cell_attribute_indices(cells_data, logger)
    
    # Create a renderer
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1, 1, 1)  # White background
    
    # Create the cell visualization
    cell_actor, cell_mapper, scalar_range = create_cell_visualization(
        cells_data, args.scalar, attribute_indices, logger)
    
    if cell_actor is None:
        logger.error("Failed to create cell visualization")
        return
    
    # Use custom range if provided, otherwise use auto-range
    if custom_range:
        min_val, max_val = custom_range
    else:
        min_val, max_val = scalar_range
        logger.info(f"Using auto range: {min_val} to {max_val}")
    
    # Create colormap
    lut = create_colormap(min_val, max_val, args.colormap)
    cell_mapper.SetLookupTable(lut)
    cell_mapper.SetScalarRange(min_val, max_val)
    renderer.AddActor(cell_actor)
    
    # Create scalar bar (legend)
    scalar_bar = create_scalar_bar(lut, args.scalar)
    renderer.AddActor(scalar_bar)
    
    # Add axes
    axes = create_axes()
    renderer.AddActor(axes)
    
    # Create render window
    render_window = vtk.vtkRenderWindow()
    render_window.SetSize(800, 600)
    render_window.SetWindowName(f"PhysiCell Cells - {args.scalar} - {os.path.basename(args.filename)}")
    render_window.AddRenderer(renderer)
    
    # Create interactor and style
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    
    style = vtk.vtkInteractorStyleTrackballCamera()
    interactor.SetInteractorStyle(style)
    
    # Initialize and start the interactor
    interactor.Initialize()
    render_window.Render()
    
    # Add orientation axes widget
    axes_widget = vtk.vtkOrientationMarkerWidget()
    axes_widget.SetOrientationMarker(create_axes())
    axes_widget.SetInteractor(interactor)
    axes_widget.SetViewport(0, 0, 0.2, 0.2)
    axes_widget.EnabledOn()
    axes_widget.InteractiveOff()
    
    # Add text with information
    text_actor = vtk.vtkTextActor()
    text = f"File: {os.path.basename(args.filename)}\n"
    text += f"Cells: {cells_data.shape[1]}\n"
    text += f"Scalar: {args.scalar}\n"
    text += f"Range: {min_val:.2f} to {max_val:.2f}\n"
    text += f"Colormap: {args.colormap}"
    
    text_actor.SetInput(text)
    text_actor.GetTextProperty().SetColor(0, 0, 0)
    text_actor.GetTextProperty().SetFontSize(12)
    text_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    text_actor.SetPosition(0.02, 0.95)
    renderer.AddActor2D(text_actor)
    
    # Log summary before starting visualization
    if args.log:
        logger.info("\nVisualization Summary:")
        logger.info("-" * 40)
        logger.info(f"File: {os.path.basename(args.filename)}")
        logger.info(f"Number of cells: {cells_data.shape[1]}")
        logger.info(f"Scalar attribute: {args.scalar}")
        logger.info(f"Scalar range: {min_val} to {max_val}")
        logger.info(f"Colormap: {args.colormap}")
        logger.info("-" * 40)
    
    # Start interaction
    logger.info("Starting visualization. Close the window to exit.")
    interactor.Start()


if __name__ == "__main__":
    main() 