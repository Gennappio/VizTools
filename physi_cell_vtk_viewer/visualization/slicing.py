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
    
    def update_slice(self, microenv_data, microenv_vol_prop, origin, normal, auto_range, min_val=0, max_val=1):
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
        
        # Use the color transfer function from the microenvironment if available
        if microenv_vol_prop and microenv_vol_prop.GetRGBTransferFunction():
            ctf = microenv_vol_prop.GetRGBTransferFunction()
            self.slice_mapper.SetLookupTable(ctf)
            
            # Set scalar range - use the current microenvironment range
            if auto_range:
                if microenv_data.GetPointData() and microenv_data.GetPointData().GetScalars():
                    scalar_range = microenv_data.GetPointData().GetScalars().GetRange()
                    self.slice_mapper.SetScalarRange(scalar_range)
            else:
                self.slice_mapper.SetScalarRange(min_val, max_val)
        
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
        
        # Use the same color mapping as the original slice
        if microenv_vol_prop and microenv_vol_prop.GetRGBTransferFunction():
            self.slice_contour_mapper.SetLookupTable(microenv_vol_prop.GetRGBTransferFunction())
            if auto_range:
                if microenv_data.GetPointData() and microenv_data.GetPointData().GetScalars():
                    self.slice_contour_mapper.SetScalarRange(microenv_data.GetPointData().GetScalars().GetRange())
            else:
                self.slice_contour_mapper.SetScalarRange(min_val, max_val)
        
        # Create an actor for the triangulated slice
        self.slice_contour_actor = vtk.vtkActor()
        self.slice_contour_actor.SetMapper(self.slice_contour_mapper)
        self.slice_contour_actor.GetProperty().SetOpacity(0.7)  # Slightly transparent
        
        # Add the triangulated slice actor to the renderer
        self.renderer.AddActor(self.slice_contour_actor)
    
    def is_visible(self):
        """Check if the slice is currently visible"""
        return self.slice_actor is not None and self.slice_actor.GetVisibility()
    
    def set_visibility(self, visible):
        """Set the visibility of the slice"""
        if self.slice_actor:
            self.slice_actor.SetVisibility(visible)
        if self.slice_contour_actor:
            self.slice_contour_actor.SetVisibility(visible) 