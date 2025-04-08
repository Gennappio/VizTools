"""
Visualization components for PhysiCell microenvironment data
"""

import vtk
import numpy as np
from scipy.spatial import cKDTree
from physi_cell_vtk_viewer.utils.vtk_utils import create_color_transfer_function, create_opacity_transfer_function

class MicroenvironmentVisualizer:
    """Class for visualizing PhysiCell microenvironment data"""
    
    def __init__(self, renderer):
        """Initialize with a VTK renderer"""
        self.renderer = renderer
        self.volume_actors = []
        self.wireframe_actors = []
        self.info_actors = []
        
        # Store microenvironment data for slicing
        self.microenv_data = None
        self.microenv_vol_prop = None
        
        # Scalar range values
        self.min_val = 0.0
        self.max_val = 1.0
    
    def clear(self):
        """Remove all microenvironment actors from the renderer"""
        for actor in self.volume_actors:
            self.renderer.RemoveActor(actor)
        self.volume_actors = []
        
        for actor in self.wireframe_actors:
            self.renderer.RemoveActor(actor)
        self.wireframe_actors = []
        
        for actor in self.info_actors:
            self.renderer.RemoveActor(actor)
        self.info_actors = []
        
        # Clear stored data
        self.microenv_data = None
        self.microenv_vol_prop = None
    
    def visualize_microenvironment(self, microenv_data, wireframe_mode=False, 
                                   opacity=0.5, auto_range=True, min_val=0.0, max_val=1.0, debug=False):
        """Visualize microenvironment data from a multiscale_microenvironment array"""
        # Store the min/max values
        if auto_range:
            # Will be set from the actual data
            self.min_val = 0.0
            self.max_val = 1.0
        else:
            self.min_val = min_val
            self.max_val = max_val
        
        # Extract coordinate and substrate data
        x = microenv_data[0, :].flatten()  # x coordinates
        y = microenv_data[1, :].flatten()  # y coordinates
        z = microenv_data[2, :].flatten()  # z coordinates
        
        # Print coordinate information for debugging
        if debug:
            print(f"X range: {x.min():.6f} to {x.max():.6f}, shape: {x.shape}")
            print(f"Y range: {y.min():.6f} to {y.max():.6f}, shape: {y.shape}")
            print(f"Z range: {z.min():.6f} to {z.max():.6f}, shape: {z.shape}")
        
        # Number of substrates (chemical species)
        substrate_count = microenv_data.shape[0] - 4
        position_indices = microenv_data.shape[1]
        
        # Determine the grid dimensions
        # In PhysiCell, the grid is typically structured with equally spaced points
        unique_x = np.unique(x)
        unique_y = np.unique(y)
        unique_z = np.unique(z)
        
        nx = len(unique_x)
        ny = len(unique_y)
        nz = len(unique_z)
        
        if debug:
            print(f"Grid dimensions: {nx} x {ny} x {nz}")
        
        # Create a structured grid for the visualization
        if nz <= 1:  # 2D case - add a small z-dimension for visualization
            nz = 2
            unique_z = np.array([z[0]-0.5, z[0]+0.5]) if len(unique_z) > 0 else np.array([-0.5, 0.5])
        
        # Create VTK arrays for coordinates
        x_vtk = vtk.vtkDoubleArray()
        for val in unique_x:
            x_vtk.InsertNextValue(val)
            
        y_vtk = vtk.vtkDoubleArray()
        for val in unique_y:
            y_vtk.InsertNextValue(val)
            
        z_vtk = vtk.vtkDoubleArray()
        for val in unique_z:
            z_vtk.InsertNextValue(val)
        
        # Create a rectilinear grid (which works well for regularly spaced data)
        grid = vtk.vtkRectilinearGrid()
        grid.SetDimensions(nx, ny, nz)
        grid.SetXCoordinates(x_vtk)
        grid.SetYCoordinates(y_vtk)
        grid.SetZCoordinates(z_vtk)
        
        # Store the grid for slicing, independent of visualization
        self.microenv_data = vtk.vtkRectilinearGrid()
        self.microenv_data.DeepCopy(grid)
        
        # Process each substrate
        for substrate_idx in range(substrate_count):
            # Get the substrate data (row 4+idx in the microenvironment matrix)
            substrate_data = microenv_data[4 + substrate_idx, :]
            
            # Print substrate range for debugging
            if debug:
                print(f"Substrate {substrate_idx} range: {substrate_data.min():.6f} to {substrate_data.max():.6f}")
            
            # Create the scalar array for this substrate
            substrate_vtk = vtk.vtkDoubleArray()
            substrate_vtk.SetName(f"Substrate_{substrate_idx}")
            
            # For a rectilinear grid, we need to map the unstructured data to the structured grid
            # This requires reshaping/interpolating the values to fit the grid
            
            # Initialize a 3D array to hold the interpolated values
            grid_values = np.zeros((nx, ny, nz))
            
            # Simple case: the number of points matches the grid size
            if nx * ny == len(substrate_data) and nz <= 2:
                # Reshape for 2D data mapped to a 3D visualization
                grid_2d = substrate_data.reshape((ny, nx)).transpose()
                
                # Duplicate the 2D layer if we needed to create a fake z-dimension
                for k in range(nz):
                    grid_values[:, :, k] = grid_2d
                    
            else:
                # More complex case: we need to map points to the 3D grid
                # Use nearest neighbor mapping
                
                # Create KD-tree for fast nearest-neighbor lookup
                points = np.vstack((x, y, z)).T
                tree = cKDTree(points)
                
                # Query points on the structured grid
                all_points = []
                for i, xi in enumerate(unique_x):
                    for j, yi in enumerate(unique_y):
                        for k, zi in enumerate(unique_z):
                            all_points.append([xi, yi, zi])
                
                all_points = np.array(all_points)
                
                # Find nearest neighbors
                distances, indices = tree.query(all_points, k=1)
                
                # Map values to grid
                counter = 0
                for i in range(nx):
                    for j in range(ny):
                        for k in range(nz):
                            if counter < len(indices):
                                idx = indices[counter]
                                if idx < len(substrate_data):
                                    grid_values[i, j, k] = substrate_data[idx]
                            counter += 1
            
            # Add the scalar values to the grid in VTK order (k, j, i)
            for k in range(nz):
                for j in range(ny):
                    for i in range(nx):
                        substrate_vtk.InsertNextValue(grid_values[i, j, k])
            
            grid.GetPointData().SetScalars(substrate_vtk)
            
            # Also add to the stored microenv_data grid for slicing
            self.microenv_data.GetPointData().SetScalars(substrate_vtk.NewInstance())
            self.microenv_data.GetPointData().GetScalars().DeepCopy(substrate_vtk)
            
            # Create a volume mapper
            mapper = vtk.vtkSmartVolumeMapper()
            mapper.SetInputData(grid)
            
            # Determine color range - either auto or user-defined
            if auto_range:
                min_val = substrate_data.min()
                max_val = substrate_data.max()
                
                # Update instance variables for reference
                self.min_val = min_val
                self.max_val = max_val
            
            range_val = max_val - min_val
            
            # Create a color transfer function
            ctf = create_color_transfer_function(min_val, max_val)
            
            # Create an opacity transfer function
            otf = create_opacity_transfer_function(min_val, max_val, opacity)
            
            # Create volume properties
            volume_property = vtk.vtkVolumeProperty()
            volume_property.SetColor(ctf)
            volume_property.SetScalarOpacity(otf)
            volume_property.ShadeOn()
            volume_property.SetInterpolationTypeToLinear()
            
            # Store the volume property for slice visualization consistency
            self.microenv_vol_prop = vtk.vtkVolumeProperty()
            self.microenv_vol_prop.DeepCopy(volume_property)
            
            # Set wireframe mode based on checkbox
            if wireframe_mode:
                # Use edges to create wireframe effect
                outline_filter = vtk.vtkOutlineFilter()
                outline_filter.SetInputData(grid)
                outline_mapper = vtk.vtkPolyDataMapper()
                outline_mapper.SetInputConnection(outline_filter.GetOutputPort())
                outline_actor = vtk.vtkActor()
                outline_actor.SetMapper(outline_mapper)
                outline_actor.GetProperty().SetColor(1, 1, 1)  # White wireframe
                self.renderer.AddActor(outline_actor)
                self.wireframe_actors.append(outline_actor)
                
                # Also add visible grid lines
                grid_filter = vtk.vtkRectilinearGridOutlineFilter()
                grid_filter.SetInputData(grid)
                grid_mapper = vtk.vtkPolyDataMapper()
                grid_mapper.SetInputConnection(grid_filter.GetOutputPort())
                grid_actor = vtk.vtkActor()
                grid_actor.SetMapper(grid_mapper)
                grid_actor.GetProperty().SetColor(0.7, 0.7, 0.7)  # Light grey grid
                self.renderer.AddActor(grid_actor)
                self.wireframe_actors.append(grid_actor)
                
                # Make volume more transparent in wireframe mode
                for i in range(otf.GetSize()):
                    val = otf.GetDataPointer()[i*2]
                    opacity = otf.GetValue(val) * 0.3  # Reduce opacity
                    otf.AddPoint(val, opacity)
            
            # Create the volume
            volume = vtk.vtkVolume()
            volume.SetMapper(mapper)
            volume.SetProperty(volume_property)
            
            # Add the volume to the renderer
            self.renderer.AddVolume(volume)
            self.volume_actors.append(volume)
            
            # Add color bar for this substrate
            scalar_bar = vtk.vtkScalarBarActor()
            scalar_bar.SetLookupTable(ctf)
            scalar_bar.SetTitle(f"Substrate {substrate_idx}")
            scalar_bar.SetNumberOfLabels(5)
            
            # Position the scalar bar based on the substrate index
            x_pos = 0.05 + (substrate_idx * 0.15)
            scalar_bar.SetPosition(x_pos, 0.05)
            scalar_bar.SetWidth(0.1)
            scalar_bar.SetHeight(0.3)
            
            self.renderer.AddActor2D(scalar_bar)
            self.info_actors.append(scalar_bar)
        
        # Add info text about the microenvironment
        text_actor = vtk.vtkTextActor()
        text_actor.SetInput(f"Microenvironment: {substrate_count} substrates")
        text_actor.GetTextProperty().SetFontSize(16)
        text_actor.GetTextProperty().SetColor(1, 1, 0)
        text_actor.SetPosition(20, 90)
        self.renderer.AddActor2D(text_actor)
        self.info_actors.append(text_actor)
        
        return True
    
    def get_microenv_data(self):
        """Get the stored microenvironment data for slicing"""
        return self.microenv_data
    
    def get_microenv_vol_prop(self):
        """Get the stored microenvironment volume property for coloring slices"""
        return self.microenv_vol_prop
    
    def get_data_range(self):
        """Get the microenvironment data range"""
        return self.min_val, self.max_val
    
    def set_visibility(self, visible):
        """Set the visibility of the microenvironment visualization"""
        for actor in self.volume_actors:
            actor.SetVisibility(visible)
        for actor in self.wireframe_actors:
            actor.SetVisibility(visible)
        for actor in self.info_actors:
            actor.SetVisibility(visible)
    
    def set_opacity(self, opacity):
        """Set the opacity of the microenvironment visualization"""
        opacity_factor = opacity / 100.0  # Convert from slider value (0-100)
        
        for actor in self.volume_actors:
            volume_property = actor.GetProperty()
            otf = volume_property.GetScalarOpacity()
            
            try:
                # Create a new opacity transfer function
                new_otf = vtk.vtkPiecewiseFunction()
                
                # Get min and max values from the current opacity function
                data_min = self.min_val if hasattr(self, 'min_val') else 0.0
                data_max = self.max_val if hasattr(self, 'max_val') else 1.0
                
                # Create a new opacity function with scaled values
                if data_min != data_max:
                    new_otf.AddPoint(data_min, 0.0)
                    new_otf.AddPoint(data_min + (data_max - data_min) * 0.25, 0.1 * opacity_factor)
                    new_otf.AddPoint(data_min + (data_max - data_min) * 0.5, 0.3 * opacity_factor)
                    new_otf.AddPoint(data_min + (data_max - data_min) * 0.75, 0.6 * opacity_factor)
                    new_otf.AddPoint(data_max, 0.8 * opacity_factor)
                else:
                    new_otf.AddPoint(data_min, 0.5 * opacity_factor)
                
                # Set the new opacity function
                volume_property.SetScalarOpacity(new_otf)
            except Exception as e:
                print(f"Error setting opacity: {e}")
                # Fallback to a simple opacity function
                new_otf = vtk.vtkPiecewiseFunction()
                new_otf.AddPoint(0.0, 0.0)
                new_otf.AddPoint(1.0, opacity_factor)
                volume_property.SetScalarOpacity(new_otf) 