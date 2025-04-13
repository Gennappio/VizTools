"""
Slicing functionality for visualizing planar slices through volumetric data
"""

import vtk
from physi_cell_vtk_viewer.utils.vtk_utils import create_color_transfer_function

class SliceVisualizer:
    """Class for creating and managing slice visualizations"""
    
    def __init__(self, renderer):
        """Initialize the slicer with a VTK renderer"""
        self.renderer = renderer
        self.slice_actor = None
        self.slice_mapper = None
        self.slice_plane = None
        self.slice_contour_actor = None
        self.slice_contour_mapper = None
        self.scalar_bar_actor = None
        self.contour_actor = None  # For specific contour value
        
        # Store last valid microenvironment data and properties
        self.last_microenv_data = None
        self.last_microenv_vol_prop = None
        self.last_min_val = 0.0
        self.last_max_val = 1.0
    
    def clear_slice(self):
        """Remove any existing slice visualization"""
        if self.slice_actor:
            self.renderer.RemoveActor(self.slice_actor)
            self.slice_actor = None
        
        if self.slice_contour_actor:
            self.renderer.RemoveActor(self.slice_contour_actor)
            self.slice_contour_actor = None
            
        if self.scalar_bar_actor:
            self.renderer.RemoveActor2D(self.scalar_bar_actor)
            self.scalar_bar_actor = None
            
        if self.contour_actor:
            self.renderer.RemoveActor(self.contour_actor)
            self.contour_actor = None
    
    def update_slice(self, microenv_data, microenv_vol_prop, origin, normal, auto_range, min_val=0, max_val=1, show_contour=False, contour_value=0.5):
        """Update the slice visualization based on current settings"""
        # Clean up existing slice actors
        self.clear_slice()
        
        # Store valid data for future use
        if microenv_data is not None:
            self.last_microenv_data = microenv_data
            self.last_microenv_vol_prop = microenv_vol_prop
            self.last_min_val = min_val
            self.last_max_val = max_val
        else:
            # Use last stored data if available
            microenv_data = self.last_microenv_data
            microenv_vol_prop = self.last_microenv_vol_prop
            min_val = self.last_min_val
            max_val = self.last_max_val
            
        # If no data is available, don't create a slice
        if not microenv_data:
            return
            
        # Create or update the slice plane
        if not self.slice_plane:
            self.slice_plane = vtk.vtkPlane()
            
        # Set plane origin and normal
        self.slice_plane.SetOrigin(*origin)
        
        # Normalize the normal vector
        i, j, k = normal
        length = (i*i + j*j + k*k)**0.5
        if length > 0:
            i /= length
            j /= length
            k /= length
        self.slice_plane.SetNormal(i, j, k)
        
        # Create the cutter to slice the microenvironment data
        cutter = vtk.vtkCutter()
        cutter.SetCutFunction(self.slice_plane)
        cutter.SetInputData(microenv_data)  # Use stored microenv data
        cutter.Update()
        
        # Get the slice output
        slice_output = cutter.GetOutput()
        
        # Create mapper for the slice
        self.slice_mapper = vtk.vtkPolyDataMapper()
        self.slice_mapper.SetInputConnection(cutter.GetOutputPort())
        self.slice_mapper.ScalarVisibilityOn()  # Show the scalars
        
        # Determine the scalar range to use
        scalar_range = [min_val, max_val]  # Default to specified values
        if auto_range:
            if microenv_data.GetPointData() and microenv_data.GetPointData().GetScalars():
                scalar_range = microenv_data.GetPointData().GetScalars().GetRange()
        
        # Get or create an appropriate color transfer function
        ctf = None
        if microenv_vol_prop and microenv_vol_prop.GetRGBTransferFunction():
            # Use the volume property's color transfer function
            ctf = microenv_vol_prop.GetRGBTransferFunction()
            print(f"Using volume property CTF with range: {ctf.GetRange()}")
        else:
            # Create a new color transfer function with standard rainbow colors
            ctf = vtk.vtkColorTransferFunction()
            ctf.AddRGBPoint(scalar_range[0], 0.0, 0.0, 1.0)  # Blue for min
            ctf.AddRGBPoint(scalar_range[0] + (scalar_range[1] - scalar_range[0]) * 0.25, 0.0, 1.0, 1.0)  # Cyan 
            ctf.AddRGBPoint(scalar_range[0] + (scalar_range[1] - scalar_range[0]) * 0.5, 0.0, 1.0, 0.0)   # Green
            ctf.AddRGBPoint(scalar_range[0] + (scalar_range[1] - scalar_range[0]) * 0.75, 1.0, 1.0, 0.0)  # Yellow
            ctf.AddRGBPoint(scalar_range[1], 1.0, 0.0, 0.0)  # Red for max
            print(f"Created new CTF with range: {scalar_range}")
            
        # Apply the color transfer function to the mapper
        self.slice_mapper.SetLookupTable(ctf)
        self.slice_mapper.SetScalarRange(scalar_range)
        print(f"Set slice mapper scalar range to: {scalar_range}")
        
        # Create actor for the slice
        self.slice_actor = vtk.vtkActor()
        self.slice_actor.SetMapper(self.slice_mapper)
        
        # Make the slice more visible
        self.slice_actor.GetProperty().SetLineWidth(1)
        self.slice_actor.GetProperty().SetPointSize(4)
        self.slice_actor.GetProperty().SetOpacity(1.0)  # Fully opaque
        
        # Add the slice actor to the renderer
        self.renderer.AddActor(self.slice_actor)
        
        # Create a triangulated surface for better visualization
        warp = vtk.vtkWarpScalar()
        warp.SetInputConnection(cutter.GetOutputPort())
        warp.SetScaleFactor(0)  # No actual warping, just for triangulation
        
        triangulate = vtk.vtkTriangleFilter()
        triangulate.SetInputConnection(warp.GetOutputPort())
        
        # Create mapper for the triangulated slice
        self.slice_contour_mapper = vtk.vtkPolyDataMapper()
        self.slice_contour_mapper.SetInputConnection(triangulate.GetOutputPort())
        self.slice_contour_mapper.ScalarVisibilityOn()
        
        # Apply the same color transfer function to the contour mapper
        self.slice_contour_mapper.SetLookupTable(ctf)
        self.slice_contour_mapper.SetScalarRange(scalar_range)
        
        # Create an actor for the triangulated slice
        self.slice_contour_actor = vtk.vtkActor()
        self.slice_contour_actor.SetMapper(self.slice_contour_mapper)
        self.slice_contour_actor.GetProperty().SetOpacity(0.7)  # Slightly transparent
        
        # Add a scalar bar (color legend) for the slice
        scalar_bar = vtk.vtkScalarBarActor()
        scalar_bar.SetLookupTable(ctf)
        scalar_bar.SetTitle("Slice Values")
        scalar_bar.SetNumberOfLabels(5)
        scalar_bar.SetPosition(0.05, 0.85)  # Position in the top-left corner
        scalar_bar.SetWidth(0.15)
        scalar_bar.SetHeight(0.5)
        scalar_bar.SetLabelFormat("%.3g")
        self.renderer.AddActor2D(scalar_bar)
        self.scalar_bar_actor = scalar_bar  # Store reference to the scalar bar actor
        
        # Add the triangulated slice actor to the renderer
        self.renderer.AddActor(self.slice_contour_actor)
        
        # Add contour line if requested
        if show_contour:
            self.add_contour_line(slice_output, contour_value, scalar_range)
    
    def add_contour_line(self, slice_data, contour_value, scalar_range):
        """Add a contour line at the specified value to the slice visualization"""
        # Create a contour filter to extract the contour line
        contour_filter = vtk.vtkContourFilter()
        contour_filter.SetInputData(slice_data)
        contour_filter.SetValue(0, contour_value)  # Set the contour value
        contour_filter.Update()
        
        # Check if the contour contains any points
        if contour_filter.GetOutput().GetNumberOfPoints() == 0:
            print(f"No contour generated at value {contour_value} (outside of scalar range {scalar_range})")
            return
            
        # Create a tube filter to make the contour line more visible
        tube_filter = vtk.vtkTubeFilter()
        tube_filter.SetInputConnection(contour_filter.GetOutputPort())
        tube_filter.SetRadius(0.2)  # Adjust size of the tube/line
        tube_filter.SetNumberOfSides(12)
        tube_filter.Update()
        
        # Create a mapper for the contour
        contour_mapper = vtk.vtkPolyDataMapper()
        contour_mapper.SetInputConnection(tube_filter.GetOutputPort())
        contour_mapper.ScalarVisibilityOff()  # Don't use scalar coloring
        
        # Create an actor for the contour line
        self.contour_actor = vtk.vtkActor()
        self.contour_actor.SetMapper(contour_mapper)
        
        # Set contour line color (using black for high contrast)
        self.contour_actor.GetProperty().SetColor(0, 0, 0)  # Black
        self.contour_actor.GetProperty().SetLineWidth(2)
        
        # Add the contour actor to the renderer
        self.renderer.AddActor(self.contour_actor)
        
        print(f"Added contour line at value {contour_value}")
    
    def is_visible(self):
        """Check if the slice is currently visible"""
        return self.slice_actor is not None and self.slice_actor.GetVisibility()
    
    def set_visibility(self, visible):
        """Set the visibility of the slice"""
        if self.slice_actor:
            self.slice_actor.SetVisibility(visible)
        if self.slice_contour_actor:
            self.slice_contour_actor.SetVisibility(visible)
        if self.scalar_bar_actor:
            self.scalar_bar_actor.SetVisibility(visible)
        if self.contour_actor:
            self.contour_actor.SetVisibility(visible) 