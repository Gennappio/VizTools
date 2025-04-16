#!/usr/bin/env python3
"""
Test script for contour functionality in PhysiCell Slicer
"""

import vtk
import numpy as np

def create_test_grid():
    """Create a test grid with artificial data."""
    # Create a 20x20x20 structured grid
    grid = vtk.vtkStructuredGrid()
    grid.SetDimensions(20, 20, 20)
    
    # Create points
    points = vtk.vtkPoints()
    points.SetNumberOfPoints(20 * 20 * 20)
    
    # Create data
    scalars = vtk.vtkFloatArray()
    scalars.SetNumberOfValues(20 * 20 * 20)
    scalars.SetName("TestData")
    
    # Fill the grid with points and data
    idx = 0
    for z in range(20):
        for y in range(20):
            for x in range(20):
                # Calculate point index
                idx = x + y * 20 + z * 20 * 20
                
                # Set point coordinates
                points.SetPoint(idx, x * 10, y * 10, z * 10)
                
                # Create a radial gradient from the center
                dx = x - 10
                dy = y - 10
                dz = z - 10
                distance = np.sqrt(dx*dx + dy*dy + dz*dz)
                
                # Value decreases with distance from center
                value = 2.0 * np.exp(-distance/5)
                
                # Set the scalar value
                scalars.SetValue(idx, value)
    
    # Add points and scalars to the grid
    grid.SetPoints(points)
    grid.GetPointData().SetScalars(scalars)
    
    return grid

def create_slice(grid, position=None, normal=(0, 0, 1)):
    """Create a slice through the grid."""
    # Get grid bounds
    bounds = grid.GetBounds()  # (xmin, xmax, ymin, ymax, zmin, zmax)
    
    # If position is not specified, use center of grid
    if position is None:
        position = [
            (bounds[0] + bounds[1]) / 2,  # x center
            (bounds[2] + bounds[3]) / 2,  # y center
            (bounds[4] + bounds[5]) / 2   # z center
        ]
    
    print(f"Creating slice at position {position} with normal {normal}")
    print(f"Grid bounds: {bounds}")
    
    # Create the slice plane
    plane = vtk.vtkPlane()
    plane.SetOrigin(position)
    plane.SetNormal(normal)
    
    # Create a cutter
    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(plane)
    cutter.SetInputData(grid)
    cutter.Update()
    
    slice_output = cutter.GetOutput()
    print(f"Slice has {slice_output.GetNumberOfPoints()} points and {slice_output.GetNumberOfCells()} cells")
    
    scalar_range = slice_output.GetPointData().GetScalars().GetRange()
    print(f"Slice scalar range: {scalar_range[0]} to {scalar_range[1]}")
    
    # Create a mapper and actor for the slice
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(cutter.GetOutputPort())
    
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    
    return actor, mapper, cutter

def create_contour(cutter, contour_value):
    """Create a contour polyline at the specified value."""
    print(f"Creating contour at value: {contour_value}")
    
    # Create a banded contour filter
    contour_filter = vtk.vtkBandedPolyDataContourFilter()
    contour_filter.SetInputConnection(cutter.GetOutputPort())
    contour_filter.SetNumberOfContours(1)
    contour_filter.SetValue(0, contour_value)
    contour_filter.SetClipping(False)
    contour_filter.SetScalarModeToValue()
    contour_filter.GenerateContourEdgesOn()
    contour_filter.Update()
    
    # Log info about the output
    contour_output = contour_filter.GetOutput()
    print(f"Contour filter output has {contour_output.GetNumberOfPoints()} points and {contour_output.GetNumberOfCells()} cells")
    
    contour_edges = contour_filter.GetContourEdgesOutput()
    print(f"Contour edges has {contour_edges.GetNumberOfPoints()} points and {contour_edges.GetNumberOfCells()} cells")
    
    # Create a mapper and actor for the contour
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(contour_filter.GetContourEdgesOutput())
    
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0, 0, 0)  # Black contour
    actor.GetProperty().SetLineWidth(3.0)  # Thicker for visibility
    
    return actor

def main():
    """Main function to test contour functionality."""
    # Create the test grid
    grid = create_test_grid()
    
    # Create a renderer
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1, 1, 1)  # White background
    
    # Create a slice
    slice_position = [100, 100, 100]  # Center of the grid
    slice_normal = [0, 0, 1]  # XY plane
    slice_actor, slice_mapper, cutter = create_slice(grid, slice_position, slice_normal)
    
    # Create a colormap
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(256)
    lut.SetHueRange(0.667, 0.0)  # Blue to Red
    lut.SetRange(0, 2)
    lut.Build()
    
    # Apply the colormap to the slice
    slice_mapper.SetLookupTable(lut)
    slice_mapper.SetScalarRange(0, 2)
    renderer.AddActor(slice_actor)
    
    # Create multiple contours at different values
    contour_values = [0.2, 0.5, 1.0, 1.5]
    for value in contour_values:
        contour_actor = create_contour(cutter, value)
        renderer.AddActor(contour_actor)
        print(f"Added contour at value: {value}")
    
    # Create a scalar bar for the colormap
    scalar_bar = vtk.vtkScalarBarActor()
    scalar_bar.SetLookupTable(lut)
    scalar_bar.SetTitle("Test Data")
    scalar_bar.SetNumberOfLabels(5)
    scalar_bar.SetOrientationToVertical()
    scalar_bar.SetPosition(0.85, 0.1)
    scalar_bar.SetWidth(0.1)
    scalar_bar.SetHeight(0.8)
    renderer.AddActor(scalar_bar)
    
    # Create render window
    render_window = vtk.vtkRenderWindow()
    render_window.SetSize(800, 600)
    render_window.SetWindowName("Contour Test")
    render_window.AddRenderer(renderer)
    
    # Create interactor
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    
    style = vtk.vtkInteractorStyleTrackballCamera()
    interactor.SetInteractorStyle(style)
    
    # Initialize and start
    interactor.Initialize()
    render_window.Render()
    
    # Add text to show contour values
    text_actor = vtk.vtkTextActor()
    text = "Contour values: "
    text += ", ".join([str(v) for v in contour_values])
    text_actor.SetInput(text)
    text_actor.GetTextProperty().SetColor(0, 0, 0)
    text_actor.GetTextProperty().SetFontSize(12)
    text_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    text_actor.SetPosition(0.02, 0.95)
    renderer.AddActor2D(text_actor)
    
    # Start interaction
    print("Starting visualization. Close the window to exit.")
    interactor.Start()

if __name__ == "__main__":
    main() 