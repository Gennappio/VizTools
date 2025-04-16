#!/usr/bin/env python3
"""
PhysiCell Slicer: A utility to display slices of PhysiCell microenvironment data using VTK.

Usage:
    python physicell_slicer.py --filename <filename> [--log] [--range min,max] [--position x,y,z] [--normal i,j,k] [--substrate name_or_index] [--contour value]

Parameters:
    --filename:     Root name of PhysiCell output files (without extension)
                    (e.g. 'output0000060' will load output0000060.xml, output0000060_cells.mat, etc.)
    --log:          Enable logging to understand what is going on (writes to output.log)
    --range:        User-defined range of values (min,max) to display in the colormap
                    If not provided, auto-range will be used
    --position:     Position of the slice (x,y,z)
                    Default: center of the domain
    --normal:       Normal direction of the slice (i,j,k)
                    Default: (0,0,1) - XY plane
    --substrate:    Name or index of the substrate to display (e.g. "oxygen" or "4")
                    Default: first substrate found
    --contour:      Draw a contour polyline at the specified value (e.g. 0.02)
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


# Set up logging
def setup_logging(log_enabled):
    """Configure logging to file and console."""
    logger = logging.getLogger('physicell_slicer')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    if logger.handlers:
        logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    if log_enabled:
        # Create file handler
        file_handler = logging.FileHandler('output.log', mode='w')
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
        logger.info(f"PhysiCell Slicer started at {datetime.datetime.now()}")
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


def extract_substrate_names(xml_file, logger=None):
    """Extract substrate names from the XML configuration file."""
    if not xml_file or not os.path.exists(xml_file):
        if logger:
            logger.warning(f"XML file not found or not specified: {xml_file}")
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
                        substrate_names.append(name_node.text)
                        if logger:
                            logger.info(f"Found substrate: {name_node.text}")
                
                # Log the substrates with indices if logging is enabled
                if logger and substrate_names:
                    _log_substrate_list(logger, substrate_names)
                
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
        
        for container in variable_containers:
            if container is not None:
                variable_nodes = container.findall(".//variable")
                if not variable_nodes:
                    # Try direct children
                    variable_nodes = container.findall("variable")
                
                if variable_nodes:
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
                substrate_names.append(name_node.text)
                if logger:
                    logger.info(f"Found substrate: {name_node.text}")
        
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
        
        # Log the substrates with indices if logging is enabled
        if logger and substrate_names:
            _log_substrate_list(logger, substrate_names)
        
        return substrate_names
    
    except Exception as e:
        if logger:
            logger.error(f"Error extracting substrate names from XML: {e}")
            import traceback
            logger.error(traceback.format_exc())
        return []


def _log_substrate_list(logger, substrate_names):
    """Helper function to log substrate list with indices."""
    logger.info("\nAvailable substrates from XML file:")
    logger.info("-" * 40)
    logger.info(f"{'Index':<6} {'Substrate Name':<30}")
    logger.info("-" * 40)
    for i, name in enumerate(substrate_names):
        logger.info(f"{i:<6} {name:<30}")
    logger.info("-" * 40 + "\n")


def load_microenv_data(microenv_file, logger=None):
    """Load microenvironment data from a .mat file."""
    if logger:
        logger.info(f"Loading microenvironment data from {microenv_file}")
    
    try:
        mat_contents = sio.loadmat(microenv_file)
        mat_contents = {k:v for k, v in mat_contents.items() 
                       if not k.startswith('__')}
        
        # Get the microenvironment data
        if 'multiscale_microenvironment' in mat_contents:
            microenv_data = mat_contents['multiscale_microenvironment']
            
            if logger:
                logger.info(f"Microenvironment data shape: {microenv_data.shape}")
                
                # Extract coordinate info
                x = np.unique(microenv_data[0, :])
                y = np.unique(microenv_data[1, :])
                z = np.unique(microenv_data[2, :])
                
                logger.info(f"X range: {x.min():.6f} to {x.max():.6f}, values: {len(x)}")
                logger.info(f"Y range: {y.min():.6f} to {y.max():.6f}, values: {len(y)}")
                logger.info(f"Z range: {z.min():.6f} to {z.max():.6f}, values: {len(z)}")
                
                # Number of substrates (chemical species)
                substrate_count = microenv_data.shape[0] - 4
                logger.info(f"Number of substrates: {substrate_count}")
                
                # Print substrate ranges
                for substrate_idx in range(substrate_count):
                    substrate_data = microenv_data[4 + substrate_idx, :]
                    logger.info(f"Substrate {substrate_idx} range: {substrate_data.min():.6f} to {substrate_data.max():.6f}")
            
            return microenv_data
        
        return None
        
    except Exception as e:
        if logger:
            logger.error(f"Error loading microenvironment data: {e}")
        return None


def find_substrate_index(substrate_name_or_index, substrate_names, substrate_count, logger=None):
    """Find the index of a substrate by name or index."""
    if not substrate_name_or_index:
        return 0  # Default to first substrate
    
    # First try to interpret as an integer (index)
    try:
        idx = int(substrate_name_or_index)
        # Check if it's a valid index
        if 0 <= idx < substrate_count:
            if logger:
                logger.info(f"Using substrate index: {idx}")
            return idx
        else:
            if logger:
                logger.warning(f"Substrate index {idx} out of range (0-{substrate_count-1}). Using default (0).")
            return 0
    except ValueError:
        # Not an integer, treat as a name
        # Convert to lowercase for case-insensitive comparison
        substrate_name_lower = substrate_name_or_index.lower()
        
        # Try exact match
        for i, name in enumerate(substrate_names):
            if name.lower() == substrate_name_lower:
                if logger:
                    logger.info(f"Found exact match for substrate '{substrate_name_or_index}' at index {i}")
                return i
        
        # Try partial match
        for i, name in enumerate(substrate_names):
            if substrate_name_lower in name.lower():
                if logger:
                    logger.info(f"Found partial match for substrate '{substrate_name_or_index}' in '{name}' at index {i}")
                return i
        
        # No match found
        if logger:
            logger.warning(f"Substrate '{substrate_name_or_index}' not found. Using default (index 0)")
        return 0


def process_microenv_data(microenv_data, substrate_idx=0, logger=None):
    """Process microenvironment data for visualization."""
    if microenv_data is None:
        if logger:
            logger.error("No microenvironment data to process")
        return None, None, None, None
    
    # Extract spatial coordinates
    x_coords = microenv_data[0, :]
    y_coords = microenv_data[1, :]
    z_coords = microenv_data[2, :]
    
    # Extract unique coordinates to determine mesh dimensions
    x_unique = np.unique(x_coords)
    y_unique = np.unique(y_coords)
    z_unique = np.unique(z_coords)
    
    # Grid dimensions
    nx = len(x_unique)
    ny = len(y_unique)
    nz = len(z_unique)
    
    if logger:
        logger.info(f"Grid dimensions: {nx} x {ny} x {nz}")
    
    # Create a structured grid
    grid = vtk.vtkStructuredGrid()
    grid.SetDimensions(nx, ny, nz)
    
    # Create points
    points = vtk.vtkPoints()
    points.SetNumberOfPoints(nx * ny * nz)
    
    # Extract substrate data (adjust index based on microenvironment structure)
    # First 4 rows are coordinates and time, so substrate starts at index 4
    substrate_data = microenv_data[4 + substrate_idx, :]
    
    # Scalars for the grid
    scalars = vtk.vtkFloatArray()
    scalars.SetNumberOfValues(nx * ny * nz)
    scalars.SetName(f"Substrate_{substrate_idx}")
    
    # Populate the grid points and scalars
    # The data is typically stored in a flattened format, so we need to map it to the grid
    grid_indices = {}
    point_idx = 0
    
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                # 3D grid indices to 1D index
                idx = i + j * nx + k * nx * ny
                # Set point coordinates
                points.SetPoint(idx, x_unique[i], y_unique[j], z_unique[k])
                # Store mapping from 3D space to mesh index
                grid_indices[(x_unique[i], y_unique[j], z_unique[k])] = idx
    
    # Map data to grid
    for i in range(len(x_coords)):
        x = x_coords[i]
        y = y_coords[i]
        z = z_coords[i]
        
        # Find closest grid point
        x_idx = np.argmin(np.abs(x_unique - x))
        y_idx = np.argmin(np.abs(y_unique - y))
        z_idx = np.argmin(np.abs(z_unique - z))
        
        # Get the grid index
        idx = x_idx + y_idx * nx + z_idx * nx * ny
        
        # Set the scalar value
        scalars.SetValue(idx, substrate_data[i])
    
    # Add points and scalars to the grid
    grid.SetPoints(points)
    grid.GetPointData().SetScalars(scalars)
    
    # Calculate min/max values of the substrate
    min_val = substrate_data.min()
    max_val = substrate_data.max()
    
    if logger:
        logger.info(f"Substrate {substrate_idx} range: {min_val:.6f} to {max_val:.6f}")
    
    return grid, min_val, max_val, scalars


def create_slice(grid, position=None, normal=(0, 0, 1), logger=None):
    """Create a slice through the microenvironment grid."""
    if grid is None:
        if logger:
            logger.error("No grid data to slice")
        return None, None, None
    
    # Get grid bounds
    bounds = grid.GetBounds()  # (xmin, xmax, ymin, ymax, zmin, zmax)
    
    # If position is not specified, use center of grid
    if position is None:
        position = [
            (bounds[0] + bounds[1]) / 2,  # x center
            (bounds[2] + bounds[3]) / 2,  # y center
            (bounds[4] + bounds[5]) / 2   # z center
        ]
    
    if logger:
        logger.info(f"Creating slice at position {position} with normal {normal}")
        logger.info(f"Grid bounds: {bounds}")
    
    # Create the slice plane
    plane = vtk.vtkPlane()
    plane.SetOrigin(position)
    plane.SetNormal(normal)
    
    # Create a cutter
    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(plane)
    cutter.SetInputData(grid)
    cutter.Update()
    
    # Create a mapper and actor for the slice
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(cutter.GetOutputPort())
    
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    
    return actor, mapper, cutter


def create_colormap(min_val, max_val):
    """Create a color transfer function for the slice coloring."""
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(256)
    lut.SetHueRange(0.667, 0.0)  # Blue to Red
    lut.SetRange(min_val, max_val)
    lut.Build()
    return lut


def create_scalar_bar(lut, title="Substrate"):
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


def create_outline(grid):
    """Create a wireframe outline of the domain."""
    outline = vtk.vtkOutlineFilter()
    outline.SetInputData(grid)
    
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(outline.GetOutputPort())
    
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0, 0, 0)  # Black wireframe
    actor.GetProperty().SetLineWidth(2.0)
    
    return actor


def create_contour(cutter, contour_value, logger=None):
    """Create a contour polyline at the specified value."""
    if logger:
        logger.info(f"Creating contour at value: {contour_value}")
    
    # Get the output from the cutter
    slice_output = cutter.GetOutput()
    
    if logger:
        logger.info(f"Slice output has {slice_output.GetNumberOfPoints()} points and {slice_output.GetNumberOfCells()} cells")
        if slice_output.GetPointData().GetScalars():
            scalar_range = slice_output.GetPointData().GetScalars().GetRange()
            logger.info(f"Slice scalar range: {scalar_range[0]} to {scalar_range[1]}")
        else:
            logger.warning("Slice has no scalar data")
    
    # Try a different approach - use vtkBandedPolyDataContourFilter
    contour_filter = vtk.vtkBandedPolyDataContourFilter()
    contour_filter.SetInputData(slice_output)
    contour_filter.SetNumberOfContours(1)
    contour_filter.SetValue(0, contour_value)
    contour_filter.SetClipping(False)
    contour_filter.SetScalarModeToValue()
    contour_filter.GenerateContourEdgesOn()
    contour_filter.Update()
    
    if logger:
        logger.info(f"Contour filter type: {type(contour_filter)}")
        contour_output = contour_filter.GetOutput()
        logger.info(f"Contour filter output has {contour_output.GetNumberOfPoints()} points and {contour_output.GetNumberOfCells()} cells")
        
        # Get contour edges
        contour_edges = contour_filter.GetContourEdgesOutput()
        logger.info(f"Contour edges has {contour_edges.GetNumberOfPoints()} points and {contour_edges.GetNumberOfCells()} cells")
    
    # Create mapper and actor for the contour edges
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(contour_filter.GetContourEdgesOutput())
    
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0, 0, 0)  # Black contour
    actor.GetProperty().SetLineWidth(3.0)  # Make it a bit thicker for visibility
    
    return actor


def main():
    """Main function to run the PhysiCell slicer."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='PhysiCell Microenvironment Slicer')
    parser.add_argument('--filename', required=True, help='Root name of PhysiCell output files')
    parser.add_argument('--log', action='store_true', help='Enable logging to output.log')
    parser.add_argument('--range', type=str, help='Value range min,max (e.g. 0,1)')
    parser.add_argument('--position', type=str, help='Slice position x,y,z (e.g. 0,0,0)')
    parser.add_argument('--normal', type=str, help='Slice normal i,j,k (e.g. 0,0,1)')
    parser.add_argument('--substrate', type=str, help='Name or index of the substrate to display')
    parser.add_argument('--contour', type=float, help='Draw a contour polyline at the specified value')
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(args.log)
    
    # Set VTK window backend
    # Force VTK to use the native window mode (not Cocoa)
    os.environ["VTK_RENDERER"] = "OpenGL"
    
    # Process arguments
    if args.log:
        logger.info(f"Processing PhysiCell output with root: {args.filename}")
    
    # Parse position if provided
    position = None
    if args.position:
        try:
            position = [float(x) for x in args.position.split(',')]
            if len(position) != 3:
                logger.error("Position must be specified as x,y,z")
                return
        except ValueError:
            logger.error("Invalid position format. Must be x,y,z (e.g. 0,0,0)")
            return
    
    # Parse normal if provided
    normal = (0, 0, 1)  # Default: XY plane
    if args.normal:
        try:
            normal = [float(x) for x in args.normal.split(',')]
            if len(normal) != 3:
                logger.error("Normal must be specified as i,j,k")
                return
        except ValueError:
            logger.error("Invalid normal format. Must be i,j,k (e.g. 0,0,1)")
            return
    
    # Parse range if provided
    custom_range = None
    if args.range:
        try:
            custom_range = [float(x) for x in args.range.split(',')]
            if len(custom_range) != 2:
                logger.error("Range must be specified as min,max")
                return
        except ValueError:
            logger.error("Invalid range format. Must be min,max (e.g. 0,1)")
            return
    
    # Find output files
    files = find_output_files(args.filename, logger)
    
    # Check if microenvironment file exists
    if not files['microenv_file']:
        logger.error(f"No microenvironment file found for {args.filename}")
        return
    
    # Load microenvironment data
    microenv_data = load_microenv_data(files['microenv_file'], logger)
    
    if microenv_data is None:
        logger.error("Failed to load microenvironment data")
        return
    
    # Calculate number of substrates
    substrate_count = microenv_data.shape[0] - 4
    logger.info(f"Total number of substrates in microenvironment data: {substrate_count}")
    
    # Extract substrate names from XML if available
    substrate_names = []
    if files['xml_file']:
        substrate_names = extract_substrate_names(files['xml_file'], logger)
    
    # Check if we found substrate names that match the count in the data
    if not substrate_names:
        logger.warning("No substrate names found in XML file")
        substrate_names = [f"substrate_{i}" for i in range(substrate_count)]
        logger.info(f"Using default substrate names: {substrate_names}")
    elif len(substrate_names) != substrate_count:
        logger.warning(f"Number of substrate names from XML ({len(substrate_names)}) doesn't match number of substrates in data ({substrate_count})")
        # Expand or truncate the list as needed
        if len(substrate_names) < substrate_count:
            # Add generic names for missing substrates
            for i in range(len(substrate_names), substrate_count):
                substrate_names.append(f"substrate_{i}")
            logger.info(f"Added default names for missing substrates: {substrate_names}")
        else:
            # Truncate extra names
            substrate_names = substrate_names[:substrate_count]
            logger.info(f"Truncated extra substrate names: {substrate_names}")
    
    # Determine substrate index (process as numeric index if possible)
    substrate_idx = 0  # Default to first substrate
    requested_substrate = args.substrate
    
    if requested_substrate:
        try:
            # First try to interpret as a number
            idx = int(requested_substrate)
            if 0 <= idx < substrate_count:
                substrate_idx = idx
                logger.info(f"Using substrate index {idx} ({substrate_names[idx] if idx < len(substrate_names) else 'unknown'})")
            else:
                logger.warning(f"Substrate index {idx} out of range (0-{substrate_count-1}). Using default substrate (0).")
        except ValueError:
            # If not a number, try to find by name
            name_match_found = False
            for i, name in enumerate(substrate_names):
                if name.lower() == requested_substrate.lower():
                    substrate_idx = i
                    name_match_found = True
                    logger.info(f"Found exact match for substrate '{requested_substrate}' at index {i}")
                    break
            
            # If no exact match, try partial match
            if not name_match_found:
                for i, name in enumerate(substrate_names):
                    if requested_substrate.lower() in name.lower():
                        substrate_idx = i
                        name_match_found = True
                        logger.info(f"Found partial match for substrate '{requested_substrate}' in '{name}' at index {i}")
                        break
            
            if not name_match_found:
                logger.warning(f"Substrate '{requested_substrate}' not found. Using default substrate (0).")
    
    # Validate substrate index is in range
    if substrate_idx >= substrate_count:
        logger.warning(f"Substrate index {substrate_idx} out of range (0-{substrate_count-1}). Using default (0).")
        substrate_idx = 0
    
    # Get the substrate name for display
    substrate_name = substrate_names[substrate_idx] if substrate_idx < len(substrate_names) else f"Substrate_{substrate_idx}"
    logger.info(f"Selected substrate: {substrate_idx} - {substrate_name}")
    
    # Process microenvironment data for visualization
    grid, min_val, max_val, scalars = process_microenv_data(microenv_data, substrate_idx, logger)
    
    if grid is None:
        logger.error("Failed to process microenvironment data")
        return
    
    # Use custom range if provided, otherwise use auto-range
    if custom_range:
        min_val, max_val = custom_range
        logger.info(f"Using custom range: {min_val} to {max_val}")
    else:
        logger.info(f"Using auto range: {min_val} to {max_val}")
    
    # Create a renderer
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1, 1, 1)  # White background
    
    # Create the slice
    slice_actor, slice_mapper, cutter = create_slice(grid, position, normal, logger)
    
    if slice_actor is None:
        logger.error("Failed to create slice")
        return
    
    # Create contour if requested - MOVED HERE, right after slice creation
    contour_actor = None
    if args.contour is not None:
        if min_val <= args.contour <= max_val:
            try:
                logger.info(f"Attempting to create contour at value: {args.contour}")
                contour_actor = create_contour(cutter, args.contour, logger)
                renderer.AddActor(contour_actor)
                logger.info(f"Added contour at value: {args.contour}")
            except Exception as e:
                logger.error(f"Error creating contour: {e}")
                import traceback
                logger.error(traceback.format_exc())
        else:
            logger.warning(f"Contour value {args.contour} is outside the data range ({min_val} - {max_val})")
    
    # Create colormap
    lut = create_colormap(min_val, max_val)
    slice_mapper.SetLookupTable(lut)
    slice_mapper.SetScalarRange(min_val, max_val)
    renderer.AddActor(slice_actor)
    
    # Create scalar bar (legend)
    scalar_bar = create_scalar_bar(lut, substrate_name)
    renderer.AddActor(scalar_bar)
    
    # Create domain outline
    outline_actor = create_outline(grid)
    renderer.AddActor(outline_actor)
    
    # Create render window
    render_window = vtk.vtkRenderWindow()
    render_window.SetSize(800, 600)
    render_window.SetWindowName(f"PhysiCell Slicer - {substrate_name} - {os.path.basename(args.filename)}")
    render_window.AddRenderer(renderer)
    
    # Create interactor and style
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    
    style = vtk.vtkInteractorStyleTrackballCamera()
    interactor.SetInteractorStyle(style)
    
    # Initialize and start the interactor
    interactor.Initialize()
    render_window.Render()
    
    # Add orientation axes
    axes = vtk.vtkAxesActor()
    axes_widget = vtk.vtkOrientationMarkerWidget()
    axes_widget.SetOrientationMarker(axes)
    axes_widget.SetInteractor(interactor)
    axes_widget.SetViewport(0, 0, 0.2, 0.2)
    axes_widget.EnabledOn()
    axes_widget.InteractiveOff()
    
    # Add text with slice information
    text_actor = vtk.vtkTextActor()
    if position:
        text = f"Slice position: ({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f})\n"
    else:
        text = "Slice position: Center\n"
    text += f"Slice normal: ({normal[0]:.1f}, {normal[1]:.1f}, {normal[2]:.1f})\n"
    text += f"Substrate: {substrate_name} (Index: {substrate_idx})\n"
    text += f"Range: {min_val:.6f} to {max_val:.6f}"
    if args.contour is not None:
        text += f"\nContour: {args.contour}"
    
    text_actor.SetInput(text)
    text_actor.GetTextProperty().SetColor(0, 0, 0)
    text_actor.GetTextProperty().SetFontSize(12)
    text_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    text_actor.SetPosition(0.02, 0.95)
    renderer.AddActor2D(text_actor)
    
    # Print available substrates with more details if logging is enabled
    if args.log and substrate_names:
        # Print detailed substrate information with the microenvironment data
        logger.info("\nSubstrate Information Summary:")
        logger.info("-" * 60)
        logger.info(f"{'Index':<6} {'Name':<30} {'Min Value':<12} {'Max Value':<12}")
        logger.info("-" * 60)
        
        for i, name in enumerate(substrate_names):
            if i < substrate_count:  # Only print substrates that exist in data
                substrate_data = microenv_data[4 + i, :]
                min_val = substrate_data.min()
                max_val = substrate_data.max()
                logger.info(f"{i:<6} {name:<30} {min_val:<12.6f} {max_val:<12.6f}")
        
        logger.info("-" * 60)
        logger.info(f"Selected substrate: {substrate_idx} - {substrate_name}")
        logger.info("-" * 60 + "\n")
    
    # Start interaction
    logger.info("Starting visualization. Close the window to exit.")
    interactor.Start()


if __name__ == "__main__":
    main() 