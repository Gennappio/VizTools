"""
Microenvironment visualization module for PhysiCell data
"""

import vtk
import numpy as np


class MicroenvVisualizer:
    """
    Visualizes PhysiCell microenvironment data in VTK
    """
    
    def __init__(self, renderer):
        """Initialize with a VTK renderer"""
        self.renderer = renderer
        self.volume_actor = None
        self.outline_actor = None
        self.scalar_bar = None
        self.visible = True
        self.opacity = 50  # Default to 50% opacity
        self.wireframe_mode = False
        
        # Set up color transfer function
        self.setup_color_functions()
        
    def setup_color_functions(self):
        """Set up color and opacity transfer functions"""
        # Color transfer function - cool to warm (blue to red)
        self.color_function = vtk.vtkColorTransferFunction()
        self.color_function.AddRGBPoint(0.0, 0.0, 0.0, 1.0)    # Blue for low values
        self.color_function.AddRGBPoint(0.5, 0.5, 0.5, 0.5)    # Gray for mid values
        self.color_function.AddRGBPoint(1.0, 1.0, 0.0, 0.0)    # Red for high values
        
        # Opacity transfer function - more transparent for low values
        self.opacity_function = vtk.vtkPiecewiseFunction()
        self.opacity_function.AddPoint(0.0, 0.0)               # Transparent for low values
        self.opacity_function.AddPoint(0.2, 0.3 * self.opacity / 100.0)  # Semi-transparent for low-mid
        self.opacity_function.AddPoint(0.5, 0.5 * self.opacity / 100.0)  # Mid opacity for mid values
        self.opacity_function.AddPoint(1.0, 0.8 * self.opacity / 100.0)  # Nearly opaque for high values
        
    def update_opacity_function(self):
        """Update opacity function based on current opacity setting"""
        if self.opacity_function:
            self.opacity_function.RemoveAllPoints()
            self.opacity_function.AddPoint(0.0, 0.0)
            self.opacity_function.AddPoint(0.2, 0.3 * self.opacity / 100.0)
            self.opacity_function.AddPoint(0.5, 0.5 * self.opacity / 100.0)
            self.opacity_function.AddPoint(1.0, 0.8 * self.opacity / 100.0)
            
            # Update volume properties if exists
            if hasattr(self, 'volume_property') and self.volume_property:
                self.volume_property.SetScalarOpacity(self.opacity_function)
                
            # Trigger render if volume actor exists
            if self.volume_actor and self.renderer:
                self.renderer.GetRenderWindow().Render()
    
    def clear(self):
        """Remove all actors from the renderer"""
        if self.volume_actor:
            self.renderer.RemoveViewProp(self.volume_actor)
            self.volume_actor = None
            
        if self.outline_actor:
            self.renderer.RemoveActor(self.outline_actor)
            self.outline_actor = None
            
        if self.scalar_bar:
            self.renderer.RemoveActor2D(self.scalar_bar)
            self.scalar_bar = None
            
    def set_visibility(self, visible):
        """Set visibility for all actors"""
        self.visible = visible
        
        if self.volume_actor:
            self.volume_actor.SetVisibility(visible)
            
        if self.outline_actor:
            self.outline_actor.SetVisibility(visible and self.wireframe_mode)
            
        if self.scalar_bar:
            self.scalar_bar.SetVisibility(visible)
            
    def set_opacity(self, opacity_percent):
        """Set opacity for microenvironment volume"""
        self.opacity = max(0, min(100, opacity_percent))
        self.update_opacity_function()
            
    def set_wireframe_mode(self, enabled):
        """Toggle wireframe mode for microenvironment visualization"""
        self.wireframe_mode = enabled
        
        if self.outline_actor:
            self.outline_actor.SetVisibility(enabled and self.visible)
            
        if self.volume_actor:
            # If wireframe is enabled, make volume less visible
            if enabled:
                self.volume_actor.SetVisibility(False)
            else:
                self.volume_actor.SetVisibility(self.visible)
        
    def visualize_microenv(self, data, variable_name="Substrate", variable_idx=0):
        """Visualize microenvironment data as volume rendering"""
        # Clear previous visualization
        self.clear()
        
        if data is None or not hasattr(data, 'shape'):
            return
            
        # Handle data based on its dimensions
        if len(data.shape) == 3:
            # Direct 3D grid data (X, Y, Z) with values
            grid_data = data
            dims = grid_data.shape
            
            # Create bounds based on grid dimensions
            # Assuming unit spacing for now
            bounds = [0, dims[0]-1, 0, dims[1]-1, 0, dims[2]-1]
            
            # Find min/max for normalization
            data_min = np.min(grid_data)
            data_max = np.max(grid_data)
            data_range = data_max - data_min if data_max > data_min else 1.0
            
        elif len(data.shape) == 4:
            # Multiple substrates (X, Y, Z, substrate_idx)
            if variable_idx >= data.shape[3]:
                variable_idx = 0  # Default to first substrate if index out of range
                
            # Extract the 3D grid for the selected substrate
            grid_data = data[:, :, :, variable_idx]
            dims = grid_data.shape
            
            # Create bounds based on grid dimensions
            bounds = [0, dims[0]-1, 0, dims[1]-1, 0, dims[2]-1]
            
            # Find min/max for normalization
            data_min = np.min(grid_data)
            data_max = np.max(grid_data)
            data_range = data_max - data_min if data_max > data_min else 1.0
            
        else:
            # Unsupported data format
            print(f"Unsupported data shape for microenvironment: {data.shape}")
            return
            
        # Create a VTK image data with the microenvironment values
        vtk_data = vtk.vtkImageData()
        vtk_data.SetDimensions(dims[0], dims[1], dims[2])
        vtk_data.SetOrigin(bounds[0], bounds[2], bounds[4])
        vtk_data.SetSpacing(
            (bounds[1] - bounds[0]) / (dims[0] - 1),
            (bounds[3] - bounds[2]) / (dims[1] - 1),
            (bounds[5] - bounds[4]) / (dims[2] - 1)
        )
        
        # Create point data array for the selected substrate
        vtk_array = vtk.vtkFloatArray()
        vtk_array.SetName(variable_name)
        
        # Normalize and populate the array
        for z in range(dims[2]):
            for y in range(dims[1]):
                for x in range(dims[0]):
                    normalized_value = (grid_data[x, y, z] - data_min) / data_range
                    vtk_array.InsertNextValue(normalized_value)
                    
        # Add array to image data
        vtk_data.GetPointData().SetScalars(vtk_array)
        
        # Create a rectilinear grid for wireframe representation
        rectilinear_grid = vtk.vtkRectilinearGrid()
        rectilinear_grid.SetDimensions(dims[0], dims[1], dims[2])
        
        # Create coordinate arrays
        x_coords = vtk.vtkFloatArray()
        y_coords = vtk.vtkFloatArray()
        z_coords = vtk.vtkFloatArray()
        
        for i in range(dims[0]):
            x_coords.InsertNextValue(bounds[0] + i * (bounds[1] - bounds[0]) / (dims[0] - 1))
            
        for i in range(dims[1]):
            y_coords.InsertNextValue(bounds[2] + i * (bounds[3] - bounds[2]) / (dims[1] - 1))
            
        for i in range(dims[2]):
            z_coords.InsertNextValue(bounds[4] + i * (bounds[5] - bounds[4]) / (dims[2] - 1))
            
        rectilinear_grid.SetXCoordinates(x_coords)
        rectilinear_grid.SetYCoordinates(y_coords)
        rectilinear_grid.SetZCoordinates(z_coords)
        
        # Set up wireframe outline
        outline = vtk.vtkRectilinearGridOutlineFilter()
        outline.SetInputData(rectilinear_grid)
        
        outline_mapper = vtk.vtkPolyDataMapper()
        outline_mapper.SetInputConnection(outline.GetOutputPort())
        
        self.outline_actor = vtk.vtkActor()
        self.outline_actor.SetMapper(outline_mapper)
        self.outline_actor.GetProperty().SetColor(0.0, 0.0, 0.0)  # Black outline
        self.outline_actor.GetProperty().SetLineWidth(2.0)
        self.outline_actor.SetVisibility(self.wireframe_mode and self.visible)
        
        self.renderer.AddActor(self.outline_actor)
        
        # Set up the volume rendering
        self.volume_property = vtk.vtkVolumeProperty()
        self.volume_property.SetColor(self.color_function)
        self.volume_property.SetScalarOpacity(self.opacity_function)
        self.volume_property.ShadeOn()
        self.volume_property.SetInterpolationTypeToLinear()
        self.volume_property.SetAmbient(0.4)
        self.volume_property.SetDiffuse(0.6)
        self.volume_property.SetSpecular(0.2)
        
        # Set up GPU ray casting mapper for better performance
        volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
        volume_mapper.SetInputData(vtk_data)
        
        # Create the volume actor
        self.volume_actor = vtk.vtkVolume()
        self.volume_actor.SetMapper(volume_mapper)
        self.volume_actor.SetProperty(self.volume_property)
        self.volume_actor.SetVisibility(self.visible and not self.wireframe_mode)
        
        self.renderer.AddViewProp(self.volume_actor)
        
        # Add scalar bar
        self.scalar_bar = vtk.vtkScalarBarActor()
        self.scalar_bar.SetLookupTable(self.color_function)
        self.scalar_bar.SetTitle(variable_name)
        self.scalar_bar.SetNumberOfLabels(5)
        self.scalar_bar.SetWidth(0.08)
        self.scalar_bar.SetHeight(0.5)
        self.scalar_bar.SetPosition(0.9, 0.25)
        self.scalar_bar.GetLabelTextProperty().SetColor(0, 0, 0)
        self.scalar_bar.GetTitleTextProperty().SetColor(0, 0, 0)
        
        # Create custom labels with actual (non-normalized) values
        labels = vtk.vtkStringArray()
        labels.SetNumberOfValues(5)
        for i in range(5):
            normalized_value = i / 4.0
            actual_value = data_min + normalized_value * data_range
            labels.SetValue(i, f"{actual_value:.2f}")
            
        self.scalar_bar.SetLabels(labels)
        self.scalar_bar.SetVisibility(self.visible)
        
        self.renderer.AddActor2D(self.scalar_bar)
        
        # Refresh the renderer
        self.renderer.GetRenderWindow().Render() 