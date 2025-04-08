"""
VTK utility functions for creating and managing VTK objects
"""

import vtk
import numpy as np

def create_color_transfer_function(min_val, max_val):
    """Create a color transfer function for volume rendering"""
    ctf = vtk.vtkColorTransferFunction()
    range_val = max_val - min_val
    
    if range_val > 0:
        # Create color points that work well for visualizing concentrations
        ctf.AddRGBPoint(min_val, 0.0, 0.0, 1.0)  # Blue for min
        ctf.AddRGBPoint(min_val + 0.25 * range_val, 0.0, 1.0, 1.0)  # Cyan
        ctf.AddRGBPoint(min_val + 0.5 * range_val, 0.0, 1.0, 0.0)   # Green
        ctf.AddRGBPoint(min_val + 0.75 * range_val, 1.0, 1.0, 0.0)  # Yellow
        ctf.AddRGBPoint(max_val, 1.0, 0.0, 0.0)  # Red for max
    else:
        # Handle case where all values are the same
        ctf.AddRGBPoint(min_val, 0.0, 0.0, 1.0)
    
    return ctf

def create_opacity_transfer_function(min_val, max_val, opacity_factor):
    """Create an opacity transfer function for volume rendering"""
    otf = vtk.vtkPiecewiseFunction()
    range_val = max_val - min_val
    
    # For values close to minimum, set lower opacity
    otf.AddPoint(min_val, 0.0)
    
    # Higher opacity for higher values, scaled by the user slider
    if range_val > 0:
        otf.AddPoint(min_val + 0.25 * range_val, 0.1 * opacity_factor)
        otf.AddPoint(min_val + 0.5 * range_val, 0.3 * opacity_factor)
        otf.AddPoint(min_val + 0.75 * range_val, 0.6 * opacity_factor)
        otf.AddPoint(max_val, 0.8 * opacity_factor)
    else:
        otf.AddPoint(min_val, 0.5 * opacity_factor)
    
    return otf

def create_cell_color(cell_type):
    """Create a color for a cell based on its type"""
    colors = {
        0: (0.7, 0.7, 0.7),  # Grey
        1: (1.0, 0.0, 0.0),  # Red
        2: (0.0, 1.0, 0.0),  # Green
        3: (0.0, 0.0, 1.0),  # Blue
        4: (1.0, 1.0, 0.0),  # Yellow
        5: (1.0, 0.0, 1.0),  # Magenta
        6: (0.0, 1.0, 1.0),  # Cyan
    }
    return colors.get(cell_type, (0.7, 0.7, 0.7))  # Default to grey

def create_cell_rgb_color(cell_type):
    """Create an RGB color tuple (0-255) for a cell based on its type"""
    cell_type_colors = {
        0: (180, 180, 180),  # Grey
        1: (255, 0, 0),      # Red
        2: (0, 255, 0),      # Green
        3: (0, 0, 255),      # Blue
        4: (255, 255, 0),    # Yellow
        5: (255, 0, 255),    # Magenta
        6: (0, 255, 255),    # Cyan
    }
    return cell_type_colors.get(cell_type, (180, 180, 180))  # Default to grey

def add_orientation_axes(renderer, size=0.1):
    """Add orientation axes to the renderer"""
    axes = vtk.vtkAxesActor()
    
    # Set axis labels
    axes.SetXAxisLabelText("X")
    axes.SetYAxisLabelText("Y")
    axes.SetZAxisLabelText("Z")
    
    # Set up widget
    axes_widget = vtk.vtkOrientationMarkerWidget()
    axes_widget.SetOrientationMarker(axes)
    axes_widget.SetViewport(0.0, 0.0, size, size)
    
    return axes_widget

def calculate_cell_radius(volume):
    """Calculate cell radius from volume"""
    return (3.0 * volume / (4.0 * np.pi)) ** (1.0/3.0)
