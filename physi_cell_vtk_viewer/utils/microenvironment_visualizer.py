"""
MicroenvironmentVisualizer module for visualizing microenvironment data from PhysiCell.
"""

import vtk
import numpy as np
import os
import scipy.io as sio


class MicroenvironmentVisualizer:
    """Class for visualizing microenvironment data from PhysiCell simulations."""
    
    def __init__(self, renderer):
        """
        Initialize the microenvironment visualizer.
        
        Args:
            renderer: VTK renderer to use for visualization
        """
        self.renderer = renderer
        self.substrate_data = None
        self.substrate_names = []
        self.current_substrate_index = 0
        self.min_value = None
        self.max_value = None
        self.slice_min_value = None
        self.slice_max_value = None
        self.volume_actor = None
        self.outline_actor = None
        self.grid_actor = None
        self.info_actors = []
        self.wireframe_mode = False
        self.sample_distance = 0.5  # Default sample distance for volume rendering
        self.grid_spacing = [2, 2, 2]  # Default grid line spacing
        self.mesh_data = None
        self.xyz_mesh = None
        self.vtk_grid = None
        
        # Initialize colormaps
        self.colormaps = {
            "BlueToRed": [
                (0.0, 0.0, 0.0, 1.0),  # Blue
                (0.5, 0.0, 1.0, 0.0),  # Green
                (1.0, 1.0, 0.0, 0.0)   # Red
            ],
            "Rainbow": [
                (0.0, 0.0, 0.0, 1.0),   # Blue
                (0.25, 0.0, 1.0, 1.0),  # Cyan
                (0.5, 0.0, 1.0, 0.0),   # Green
                (0.75, 1.0, 1.0, 0.0),  # Yellow
                (1.0, 1.0, 0.0, 0.0)    # Red
            ],
            "Grayscale": [
                (0.0, 0.0, 0.0, 0.0),  # Black
                (1.0, 1.0, 1.0, 1.0)   # White
            ],
            "Viridis": [
                (0.0, 0.267, 0.004, 0.329),
                (0.25, 0.270, 0.313, 0.612),
                (0.5, 0.204, 0.553, 0.573),
                (0.75, 0.455, 0.768, 0.357),
                (1.0, 0.992, 0.906, 0.144)
            ]
        }
        self.current_colormap_name = "BlueToRed"  # Default colormap
        
    def clear(self):
        """Remove all microenvironment actors from the renderer."""
        if self.volume_actor:
            self.renderer.RemoveActor(self.volume_actor)
            self.volume_actor = None
            
        if self.outline_actor:
            self.renderer.RemoveActor(self.outline_actor)
            self.outline_actor = None
            
        if self.grid_actor:
            self.renderer.RemoveActor(self.grid_actor)
            self.grid_actor = None
            
        self.substrate_data = None
        self.mesh_data = None
        self.xyz_mesh = None
        
    def _create_default_colormap(self):
        """
        Create a default color transfer function for volume rendering.
        
        Returns:
            VTK color transfer function
        """
        ctf = vtk.vtkColorTransferFunction()
        ctf.AddRGBPoint(0.0, 0.0, 0.0, 0.0)  # Black for lowest values
        ctf.AddRGBPoint(0.25, 0.0, 0.0, 0.8)  # Blue for low values
        ctf.AddRGBPoint(0.5, 0.0, 0.8, 0.8)   # Cyan for mid values
        ctf.AddRGBPoint(0.75, 0.8, 0.8, 0.0)   # Yellow for high values
        ctf.AddRGBPoint(1.0, 0.8, 0.0, 0.0)    # Red for highest values
        return ctf
        
    def _create_default_opacity_function(self):
        """
        Create a default opacity transfer function for volume rendering.
        
        Returns:
            VTK opacity transfer function
        """
        otf = vtk.vtkPiecewiseFunction()
        otf.AddPoint(0.0, 0.0)   # Fully transparent for lowest values
        otf.AddPoint(0.1, 0.1)   # Mostly transparent for low values
        otf.AddPoint(0.5, 0.2)   # Semi-transparent for mid values
        otf.AddPoint(0.9, 0.3)   # More opaque for high values
        otf.AddPoint(1.0, 0.4)   # Most opaque for highest values
        return otf
        
    def set_colormap(self, colormap_type="default"):
        """
        Set the colormap type for volume rendering.
        
        Args:
            colormap_type: String identifying the colormap (default, rainbow, jet, etc.)
        """
        if colormap_type == "rainbow":
            self.lut = vtk.vtkColorTransferFunction()
            self.lut.AddRGBPoint(0.0, 0.0, 0.0, 1.0)    # Blue
            self.lut.AddRGBPoint(0.25, 0.0, 0.5, 1.0)   # Blue-Purple
            self.lut.AddRGBPoint(0.5, 0.0, 1.0, 0.0)    # Green
            self.lut.AddRGBPoint(0.75, 1.0, 1.0, 0.0)   # Yellow
            self.lut.AddRGBPoint(1.0, 1.0, 0.0, 0.0)    # Red
        elif colormap_type == "jet":
            self.lut = vtk.vtkColorTransferFunction()
            self.lut.AddRGBPoint(0.0, 0.0, 0.0, 0.5)    # Dark Blue
            self.lut.AddRGBPoint(0.2, 0.0, 0.0, 1.0)    # Blue
            self.lut.AddRGBPoint(0.4, 0.0, 1.0, 1.0)    # Cyan
            self.lut.AddRGBPoint(0.6, 1.0, 1.0, 0.0)    # Yellow
            self.lut.AddRGBPoint(0.8, 1.0, 0.0, 0.0)    # Red
            self.lut.AddRGBPoint(1.0, 0.5, 0.0, 0.0)    # Dark Red
        elif colormap_type == "grayscale":
            self.lut = vtk.vtkColorTransferFunction()
            self.lut.AddRGBPoint(0.0, 0.0, 0.0, 0.0)    # Black
            self.lut.AddRGBPoint(1.0, 1.0, 1.0, 1.0)    # White
        else:  # Default
            self.lut = self._create_default_colormap()
            
        # Update volume properties if volume exists
        if self.volume_actor:
            volume_property = self.volume_actor.GetProperty()
            volume_property.SetColor(self.lut)
            self.renderer.Render()
            
    def set_opacity(self, opacity_percent):
        """
        Set the opacity of the microenvironment visualization
        
        Args:
            opacity_percent: Opacity percentage (0-100)
        """
        opacity = opacity_percent / 100.0 if isinstance(opacity_percent, (int, float)) else 0.5
        
        if self.volume_actor:
            # Get current volume property
            volume_property = self.volume_actor.GetProperty()
            
            # Create a new opacity transfer function
            new_otf = vtk.vtkPiecewiseFunction()
            new_otf.AddPoint(0.0, 0.0)   # Fully transparent for lowest values
            new_otf.AddPoint(0.1, 0.1 * opacity / 0.4)   # Mostly transparent for low values
            new_otf.AddPoint(0.5, 0.2 * opacity / 0.4)   # Semi-transparent for mid values
            new_otf.AddPoint(0.9, 0.3 * opacity / 0.4)   # More opaque for high values
            new_otf.AddPoint(1.0, 0.4 * opacity / 0.4)   # Most opaque for highest values
            
            # Set the new opacity function
            volume_property.SetScalarOpacity(new_otf)
            
            self.renderer.Render()
            
    def set_value_range(self, min_value, max_value):
        """
        Set the min and max values for the colormap.
        
        Args:
            min_value: Minimum value for colormap
            max_value: Maximum value for colormap
        """
        if min_value >= max_value:
            print("Error: min_value must be less than max_value")
            return
            
        self.min_value = min_value
        self.max_value = max_value
        
        # Update the color map with the new range
        if self.substrate_data is not None and self.current_substrate_index < len(self.substrate_data):
            self._update_visualization()
    
    def set_wireframe_mode(self, enable_wireframe):
        """
        Toggle wireframe visualization mode for the microenvironment.
        
        Args:
            enable_wireframe: Boolean indicating whether to enable wireframe mode
        """
        self.wireframe_mode = enable_wireframe
        
        if self.substrate_data is not None:
            self._update_visualization()
    
    def set_sample_distance(self, distance):
        """
        Set the sample distance for volume rendering.
        
        Args:
            distance: Float value for sample distance (smaller = higher quality, slower)
        """
        self.sample_distance = max(0.1, min(2.0, distance))
        
        # Update volume mapper if it exists
        if self.volume_actor:
            mapper = self.volume_actor.GetMapper()
            if isinstance(mapper, vtk.vtkGPUVolumeRayCastMapper):
                mapper.SetSampleDistance(self.sample_distance)
                self.renderer.Render()
    
    def set_grid_spacing(self, x_spacing, y_spacing, z_spacing):
        """
        Set the grid spacing for wireframe mode.
        
        Args:
            x_spacing: Grid spacing in x direction
            y_spacing: Grid spacing in y direction
            z_spacing: Grid spacing in z direction
        """
        self.grid_spacing = [x_spacing, y_spacing, z_spacing]
        
        # Update wireframe if in wireframe mode
        if self.wireframe_mode and self.mesh_data is not None:
            self._update_wireframe()
    
    def visualize_microenvironment_from_mat(self, mat_file):
        """
        Visualize microenvironment data from a MAT file.
        
        Args:
            mat_file: Path to MAT file containing microenvironment data
            
        Returns:
            True if visualization was successful, False otherwise
        """
        if not os.path.exists(mat_file):
            print(f"MAT file not found: {mat_file}")
            return False
            
        try:
            # Load MAT file
            mat_data = sio.loadmat(mat_file)
            
            # Check for microenvironment data
            if 'multiscale_microenvironment' not in mat_data:
                print("No microenvironment data found in MAT file")
                return False
                
            # Extract data and metadata
            microenv = mat_data['multiscale_microenvironment']
            
            # Store substrate data (should be a 2D array with substrates as rows)
            if len(microenv.shape) != 2:
                print(f"Unexpected microenvironment data shape: {microenv.shape}")
                return False
                
            # PhysiCell format: first row contains x,y,z coordinates, remaining rows are substrates
            # We need to reshape this into a structured 3D grid
            self.substrate_data = []
            self.substrate_names = []
            
            # Extract unique x, y, z coordinates to determine grid dimensions
            x_coords = np.unique(microenv[0, :])
            y_coords = np.unique(microenv[1, :])
            z_coords = np.unique(microenv[2, :])
            
            x_coords.sort()
            y_coords.sort()
            z_coords.sort()
            
            # Store grid dimensions
            self.xyz_mesh = (x_coords, y_coords, z_coords)
            
            nx = len(x_coords)
            ny = len(y_coords)
            nz = len(z_coords)
            
            print(f"Detected grid dimensions: {nx} x {ny} x {nz}")
            
            # Number of substrates (skip first 3 rows which are x,y,z coordinates)
            num_substrates = microenv.shape[0] - 3
            
            # Try to read substrate names if present
            if 'multiscale_microenvironment_substrate_metadata' in mat_data:
                metadata = mat_data['multiscale_microenvironment_substrate_metadata']
                self.substrate_names = [name[0] for name in metadata]
            else:
                # Generate default substrate names
                self.substrate_names = [f"Substrate {i}" for i in range(num_substrates)]
                
            print(f"Found {num_substrates} substrates: {', '.join(self.substrate_names)}")
            
            # Reshape each substrate into a 3D array
            for i in range(num_substrates):
                # Extract substrate data (offset by 3 to skip x,y,z coordinates)
                substrate_data = microenv[i+3, :]
                
                # Create an empty 3D array
                grid_data = np.zeros((nx, ny, nz))
                
                # Map flat data into structured grid
                for idx in range(microenv.shape[1]):
                    x = microenv[0, idx]
                    y = microenv[1, idx]
                    z = microenv[2, idx]
                    
                    # Find indices in our structured grid
                    x_idx = np.where(x_coords == x)[0][0]
                    y_idx = np.where(y_coords == y)[0][0]
                    z_idx = np.where(z_coords == z)[0][0]
                    
                    # Set value
                    grid_data[x_idx, y_idx, z_idx] = substrate_data[idx]
                
                self.substrate_data.append(grid_data)
            
            # Store mesh_data for later use
            self.mesh_data = {
                'dimensions': (nx, ny, nz),
                'bounds': (
                    x_coords.min(), x_coords.max(),
                    y_coords.min(), y_coords.max(),
                    z_coords.min(), z_coords.max()
                )
            }
            
            # Visualize the first substrate by default
            self.current_substrate_index = 0
            self._update_visualization()
            
            return True
            
        except Exception as e:
            print(f"Error visualizing microenvironment: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def select_substrate(self, substrate_index):
        """
        Select a substrate to visualize.
        
        Args:
            substrate_index: Index of the substrate to visualize
            
        Returns:
            True if successful, False otherwise
        """
        if self.substrate_data is None or substrate_index >= len(self.substrate_data):
            print(f"Invalid substrate index: {substrate_index}")
            return False
            
        self.current_substrate_index = substrate_index
        self._update_visualization()
        return True
    
    def get_substrate_list(self):
        """
        Get a list of available substrates.
        
        Returns:
            List of substrate names
        """
        return self.substrate_names
    
    def _update_visualization(self):
        """Update the visualization based on current settings"""
        # Clear existing visualization
        self.clear_visualization()
        
        if self.substrate_data is None or len(self.substrate_data) == 0:
            return
        
        # Get current substrate data
        data = self.substrate_data[self.current_substrate_index]
        
        # Get mesh dimensions and coordinates
        nx, ny, nz = self.mesh_data['dimensions']
        x_coords, y_coords, z_coords = self.xyz_mesh
        
        # Create VTK arrays for coordinates
        x_array = vtk.vtkDoubleArray()
        for x in x_coords:
            x_array.InsertNextValue(x)
        
        y_array = vtk.vtkDoubleArray()
        for y in y_coords:
            y_array.InsertNextValue(y)
        
        z_array = vtk.vtkDoubleArray()
        for z in z_coords:
            z_array.InsertNextValue(z)
        
        # Create a rectilinear grid
        grid = vtk.vtkRectilinearGrid()
        grid.SetDimensions(nx, ny, nz)
        grid.SetXCoordinates(x_array)
        grid.SetYCoordinates(y_array)
        grid.SetZCoordinates(z_array)
        
        # Create a scalar array for the grid
        scalars = vtk.vtkDoubleArray()
        scalars.SetName(self.substrate_names[self.current_substrate_index])
        
        # Fill the scalar array with substrate data
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    scalars.InsertNextValue(data[i, j, k])
        
        # Set the scalars on the grid
        grid.GetPointData().SetScalars(scalars)
        
        # Check if the data range is very narrow (near-constant values)
        data_range = np.max(data) - np.min(data)
        is_near_constant = data_range < 1e-6
        
        if is_near_constant:
            print(f"Near-constant data detected: {np.min(data)} to {np.max(data)}")
            print("Using special visualization mode for constant data")
            # For near-constant data, create a colormap centered on the value
            mean_val = np.mean(data)
            # Create a small range around the mean value for better visualization
            min_val = mean_val * 0.99
            max_val = mean_val * 1.01
            
            # If the value is very close to zero, use a small absolute range
            if np.abs(mean_val) < 1e-6:
                min_val = -0.001
                max_val = 0.001
            
            print(f"Using value range: {min_val} to {max_val}")
            
            # Create color and opacity transfer functions for near-constant data
            ctf = vtk.vtkColorTransferFunction()
            ctf.AddRGBPoint(min_val, 0.0, 0.0, 1.0)  # Blue for lowest
            ctf.AddRGBPoint(mean_val, 0.0, 1.0, 0.0)  # Green for the constant value
            ctf.AddRGBPoint(max_val, 1.0, 0.0, 0.0)  # Red for highest
        else:
            # Create color transfer function
            ctf = vtk.vtkColorTransferFunction()
            
            # Use pre-defined color map if available
            if self.current_colormap_name in self.colormaps and not is_near_constant:
                # Get colormap data
                colormap = self.colormaps[self.current_colormap_name]
                
                # Add color points based on the colormap
                for point in colormap:
                    val, r, g, b = point
                    normalized_val = val * (self.max_value - self.min_value) + self.min_value
                    ctf.AddRGBPoint(normalized_val, r, g, b)
            else:
                # Default blue-to-red color map
                ctf.AddRGBPoint(self.min_value, 0.0, 0.0, 1.0)  # Blue for lowest
                ctf.AddRGBPoint((self.min_value + self.max_value) / 2, 0.0, 1.0, 0.0)  # Green for middle
                ctf.AddRGBPoint(self.max_value, 1.0, 0.0, 0.0)  # Red for highest
        
        # Create opacity transfer function
        otf = vtk.vtkPiecewiseFunction()
        
        # Use custom opacity function if available
        if hasattr(self, 'opacity_function') and self.opacity_function is not None:
            otf = self.opacity_function
        else:
            otf.AddPoint(0.0, 0.0)   # Fully transparent for lowest values
            otf.AddPoint(0.1, 0.1)   # Mostly transparent for low values
            otf.AddPoint(0.5, 0.2)   # Semi-transparent for mid values
            otf.AddPoint(0.9, 0.3)   # More opaque for high values
            otf.AddPoint(1.0, 0.4)   # Most opaque for highest values
        
        # If in wireframe mode, add outline and grid
        if self.wireframe_mode:
            # Add white outline
            outline_filter = vtk.vtkRectilinearGridOutlineFilter()
            outline_filter.SetInputData(grid)
            
            outline_mapper = vtk.vtkPolyDataMapper()
            outline_mapper.SetInputConnection(outline_filter.GetOutputPort())
            
            self.outline_actor = vtk.vtkActor()
            self.outline_actor.SetMapper(outline_mapper)
            self.outline_actor.GetProperty().SetColor(1, 1, 1)  # White outline
            self.outline_actor.GetProperty().SetLineWidth(2)
            
            self.renderer.AddActor(self.outline_actor)
            
            # Add grid planes
            self._add_grid_planes(grid)
        
        # Create volume mapper
        volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
        volume_mapper.SetInputData(grid)
        volume_mapper.SetSampleDistance(self.sample_distance)
        
        # Set up volume properties
        volume_property = vtk.vtkVolumeProperty()
        volume_property.SetColor(ctf)
        volume_property.SetScalarOpacity(otf)
        volume_property.ShadeOn()
        volume_property.SetInterpolationTypeToLinear()
        
        # Create volume actor
        self.volume_actor = vtk.vtkVolume()
        self.volume_actor.SetMapper(volume_mapper)
        self.volume_actor.SetProperty(volume_property)
        
        # Add scalar bar (color legend)
        scalar_bar = vtk.vtkScalarBarActor()
        scalar_bar.SetLookupTable(ctf)
        scalar_bar.SetTitle(f"{self.substrate_names[self.current_substrate_index]}")
        scalar_bar.SetNumberOfLabels(5)
        scalar_bar.SetPosition(0.05, 0.01)
        scalar_bar.SetWidth(0.9)
        scalar_bar.SetHeight(0.1)
        scalar_bar.SetOrientationToHorizontal()
        scalar_bar.SetLabelFormat("%.3g")
        self.renderer.AddActor2D(scalar_bar)
        self.info_actors.append(scalar_bar)
        
        # Add to renderer
        self.renderer.AddActor(self.volume_actor)
        self.renderer.ResetCamera()
    
    def _add_grid_planes(self, grid):
        """
        Add grid planes to the visualization for wireframe mode.
        
        Args:
            grid: VTK rectilinear grid containing the microenvironment data
        """
        # Get mesh dimensions and bounds
        nx, ny, nz = self.mesh_data['dimensions']
        x_min, x_max, y_min, y_max, z_min, z_max = self.mesh_data['bounds']
        
        # Get the grid spacing
        x_spacing, y_spacing, z_spacing = self.grid_spacing
        
        # Create a light grey grid
        grid_color = [0.7, 0.7, 0.7]  # Light grey
        
        # Create grid lines using vtkPlaneSource
        grid_actors = vtk.vtkActorCollection()
        
        # X grid planes
        for i in range(0, nx, x_spacing):
            if i >= nx:
                continue
                
            x = self.xyz_mesh[0][i]
            
            # Create a plane at x position
            plane = vtk.vtkPlaneSource()
            plane.SetOrigin(x, y_min, z_min)
            plane.SetPoint1(x, y_max, z_min)
            plane.SetPoint2(x, y_min, z_max)
            plane.SetResolution(1, 1)
            
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(plane.GetOutputPort())
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetRepresentationToWireframe()
            actor.GetProperty().SetColor(grid_color)
            actor.GetProperty().SetAmbient(1.0)
            actor.GetProperty().SetDiffuse(0.0)
            actor.GetProperty().SetOpacity(0.3)
            
            grid_actors.AddItem(actor)
            
        # Y grid planes
        for j in range(0, ny, y_spacing):
            if j >= ny:
                continue
                
            y = self.xyz_mesh[1][j]
            
            # Create a plane at y position
            plane = vtk.vtkPlaneSource()
            plane.SetOrigin(x_min, y, z_min)
            plane.SetPoint1(x_max, y, z_min)
            plane.SetPoint2(x_min, y, z_max)
            plane.SetResolution(1, 1)
            
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(plane.GetOutputPort())
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetRepresentationToWireframe()
            actor.GetProperty().SetColor(grid_color)
            actor.GetProperty().SetAmbient(1.0)
            actor.GetProperty().SetDiffuse(0.0)
            actor.GetProperty().SetOpacity(0.3)
            
            grid_actors.AddItem(actor)
            
        # Z grid planes
        for k in range(0, nz, z_spacing):
            if k >= nz:
                continue
                
            z = self.xyz_mesh[2][k]
            
            # Create a plane at z position
            plane = vtk.vtkPlaneSource()
            plane.SetOrigin(x_min, y_min, z)
            plane.SetPoint1(x_max, y_min, z)
            plane.SetPoint2(x_min, y_max, z)
            plane.SetResolution(1, 1)
            
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(plane.GetOutputPort())
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetRepresentationToWireframe()
            actor.GetProperty().SetColor(grid_color)
            actor.GetProperty().SetAmbient(1.0)
            actor.GetProperty().SetDiffuse(0.0)
            actor.GetProperty().SetOpacity(0.3)
            
            grid_actors.AddItem(actor)
            
        # Create a composite grid actor
        self.grid_actor = vtk.vtkAssembly()
        grid_actors.InitTraversal()
        
        for i in range(grid_actors.GetNumberOfItems()):
            self.grid_actor.AddPart(grid_actors.GetNextActor())
            
        self.renderer.AddActor(self.grid_actor)
        
    def _update_wireframe(self):
        """Update the wireframe visualization."""
        if not self.wireframe_mode or self.mesh_data is None:
            return
            
        # Remove existing wireframe actors
        if self.outline_actor:
            self.renderer.RemoveActor(self.outline_actor)
            self.outline_actor = None
            
        if self.grid_actor:
            self.renderer.RemoveActor(self.grid_actor)
            self.grid_actor = None
            
        # Get mesh dimensions and coordinates
        nx, ny, nz = self.mesh_data['dimensions']
        x_coords, y_coords, z_coords = self.xyz_mesh
        
        # Create a rectilinear grid for the wireframe
        grid = vtk.vtkRectilinearGrid()
        grid.SetDimensions(nx, ny, nz)
        
        # Set the coordinates
        x_array = vtk.vtkDoubleArray()
        for x in x_coords:
            x_array.InsertNextValue(x)
        
        y_array = vtk.vtkDoubleArray()
        for y in y_coords:
            y_array.InsertNextValue(y)
        
        z_array = vtk.vtkDoubleArray()
        for z in z_coords:
            z_array.InsertNextValue(z)
        
        grid.SetXCoordinates(x_array)
        grid.SetYCoordinates(y_array)
        grid.SetZCoordinates(z_array)
        
        # Add white outline
        outline_filter = vtk.vtkRectilinearGridOutlineFilter()
        outline_filter.SetInputData(grid)
        
        outline_mapper = vtk.vtkPolyDataMapper()
        outline_mapper.SetInputConnection(outline_filter.GetOutputPort())
        
        self.outline_actor = vtk.vtkActor()
        self.outline_actor.SetMapper(outline_mapper)
        self.outline_actor.GetProperty().SetColor(1, 1, 1)  # White outline
        self.outline_actor.GetProperty().SetLineWidth(2)
        
        self.renderer.AddActor(self.outline_actor)
        
        # Add grid planes
        self._add_grid_planes(grid)
        
        self.renderer.Render()
    
    def visualize_microenvironment(self, microenv_data, substrate_id=0, wireframe_mode=False, 
                                  opacity=0.5, auto_range=True, min_val=0.0, max_val=1.0,
                                  slice_auto_range=True, slice_min_val=0.0, slice_max_val=1.0,
                                  debug=False):
        """
        Visualize the microenvironment data in 3D
        
        Args:
            microenv_data: Dictionary containing microenvironment data
            substrate_id: ID of the substrate to visualize
            wireframe_mode: Whether to show the volume as a wireframe
            opacity: Opacity of the volume rendering (0-1)
            auto_range: Whether to automatically determine the color range
            min_val: Minimum value for color range (if auto_range is False)
            max_val: Maximum value for color range (if auto_range is False)
            slice_auto_range: Whether to automatically determine the color range for slices
            slice_min_val: Minimum value for slice color range (if slice_auto_range is False)
            slice_max_val: Maximum value for slice color range (if slice_auto_range is False)
            debug: Whether to print debug information
        """
        # Process and store microenvironment data
        self._process_microenv_data(microenv_data, substrate_id)
        
        # Get data range if auto-ranging is enabled
        if auto_range and self.substrate_data is not None and self.substrate_data:
            data_min = np.min(self.substrate_data[self.current_substrate_index])
            data_max = np.max(self.substrate_data[self.current_substrate_index])
            
            # Update min/max values
            self.min_value = data_min
            self.max_value = data_max
            
            if debug:
                print(f"Auto-determined data range: {data_min} to {data_max}")
        else:
            # Use specified min/max values
            self.min_value = min_val
            self.max_value = max_val
            
            if debug:
                print(f"Using specified data range: {min_val} to {max_val}")
                
        # Set slice data range
        if slice_auto_range and self.substrate_data is not None and self.substrate_data:
            data_min = np.min(self.substrate_data[self.current_substrate_index])
            data_max = np.max(self.substrate_data[self.current_substrate_index])
            
            # Update slice min/max values
            self.slice_min_value = data_min
            self.slice_max_value = data_max
            
            if debug:
                print(f"Auto-determined slice data range: {data_min} to {data_max}")
        else:
            # Use specified slice min/max values
            self.slice_min_value = slice_min_val
            self.slice_max_value = slice_max_val
            
            if debug:
                print(f"Using specified slice data range: {slice_min_val} to {slice_max_val}")
        
        # Set wireframe mode
        self.wireframe_mode = wireframe_mode
        
        # Update the visualization
        self._update_visualization()
    
    def _create_vtk_grid(self):
        """Create a VTK rectilinear grid for the microenvironment data for slicing"""
        if self.substrate_data is None or len(self.substrate_data) == 0:
            return None
        
        # Get current substrate data
        data = self.substrate_data[self.current_substrate_index]
        
        # Get mesh dimensions and coordinates
        nx, ny, nz = self.mesh_data['dimensions']
        x_coords, y_coords, z_coords = self.xyz_mesh
        
        # Create VTK arrays for coordinates
        x_vtk = vtk.vtkDoubleArray()
        for x in x_coords:
            x_vtk.InsertNextValue(x)
        
        y_vtk = vtk.vtkDoubleArray()
        for y in y_coords:
            y_vtk.InsertNextValue(y)
        
        z_vtk = vtk.vtkDoubleArray()
        for z in z_coords:
            z_vtk.InsertNextValue(z)
        
        # Create a rectilinear grid
        self.vtk_grid = vtk.vtkRectilinearGrid()
        self.vtk_grid.SetDimensions(nx, ny, nz)
        self.vtk_grid.SetXCoordinates(x_vtk)
        self.vtk_grid.SetYCoordinates(y_vtk)
        self.vtk_grid.SetZCoordinates(z_vtk)
        
        # Create scalar array for the current substrate
        scalars = vtk.vtkDoubleArray()
        scalars.SetName(self.substrate_names[self.current_substrate_index])
        
        # Flatten the 3D array to match VTK's expectations
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    scalars.InsertNextValue(data[i, j, k])
        
        # Set the scalars on the grid
        self.vtk_grid.GetPointData().SetScalars(scalars)
        
        return self.vtk_grid
    
    def get_microenv_data(self):
        """Get the microenvironment VTK data for slicing"""
        return self.vtk_grid if hasattr(self, 'vtk_grid') else None
    
    def get_microenv_vol_prop(self):
        """Get the volume property for consistent slice appearance"""
        if hasattr(self, 'volume_actor') and self.volume_actor:
            return self.volume_actor.GetProperty()
        return None
    
    def get_data_range(self):
        """Get the current data range for volume visualization"""
        if hasattr(self, 'min_value') and hasattr(self, 'max_value'):
            return self.min_value, self.max_value
        elif self.substrate_data is not None and len(self.substrate_data) > 0:
            data = self.substrate_data[self.current_substrate_index]
            return np.min(data), np.max(data)
        else:
            return 0.0, 1.0
    
    def get_slice_data_range(self):
        """Get the current data range for slice visualization"""
        if hasattr(self, 'slice_min_value') and hasattr(self, 'slice_max_value'):
            return self.slice_min_value, self.slice_max_value
        elif self.substrate_data is not None and len(self.substrate_data) > 0:
            data = self.substrate_data[self.current_substrate_index]
            return np.min(data), np.max(data)
        else:
            return 0.0, 1.0
    
    def set_visibility(self, visible):
        """
        Set the visibility of all microenvironment actors
        
        Args:
            visible: Boolean indicating whether actors should be visible
        """
        if self.volume_actor:
            self.volume_actor.SetVisibility(visible)
            
        if self.outline_actor:
            self.outline_actor.SetVisibility(visible)
            
        if self.grid_actor:
            self.grid_actor.SetVisibility(visible)
            
        self.renderer.Render()
    
    def set_opacity(self, opacity_percent):
        """
        Set the opacity of the microenvironment visualization
        
        Args:
            opacity_percent: Opacity percentage (0-100)
        """
        opacity = opacity_percent / 100.0 if isinstance(opacity_percent, (int, float)) else 0.5
        
        if self.volume_actor:
            # Get current volume property
            volume_property = self.volume_actor.GetProperty()
            
            # Create a new opacity transfer function
            new_otf = vtk.vtkPiecewiseFunction()
            new_otf.AddPoint(0.0, 0.0)   # Fully transparent for lowest values
            new_otf.AddPoint(0.1, 0.1 * opacity / 0.4)   # Mostly transparent for low values
            new_otf.AddPoint(0.5, 0.2 * opacity / 0.4)   # Semi-transparent for mid values
            new_otf.AddPoint(0.9, 0.3 * opacity / 0.4)   # More opaque for high values
            new_otf.AddPoint(1.0, 0.4 * opacity / 0.4)   # Most opaque for highest values
            
            # Set the new opacity function
            volume_property.SetScalarOpacity(new_otf)
            
            self.renderer.Render()
    
    def _process_microenv_data(self, microenv_data, substrate_id=0):
        """
        Process and store microenvironment data
        
        Args:
            microenv_data: NumPy array containing microenvironment data
            substrate_id: ID of the substrate to visualize
        """
        # Store data for retrieval later
        if isinstance(microenv_data, np.ndarray):
            # Convert PhysiCell MAT format to our internal format
            print(f"Microenvironment data shape: {microenv_data.shape}")
            print(f"Requested substrate ID: {substrate_id}")
            
            # Check if this is a PhysiCell multiscale_microenvironment matrix
            if microenv_data.shape[0] >= 4:  # At least x,y,z coords + 1 substrate
                # Extract coordinates and substrates
                x_coords = np.unique(microenv_data[0, :])
                y_coords = np.unique(microenv_data[1, :])
                z_coords = np.unique(microenv_data[2, :])
                
                x_coords.sort()
                y_coords.sort()
                z_coords.sort()
                
                # Print diagnostic information about grid dimensions
                print(f"Grid dimensions: {len(x_coords)} x {len(y_coords)} x {len(z_coords)}")
                print(f"Total voxels: {len(x_coords) * len(y_coords) * len(z_coords)}")
                print(f"Data points: {microenv_data.shape[1]}")
                
                # Verify grid size matches expected data points
                expected_points = len(x_coords) * len(y_coords) * len(z_coords)
                if expected_points != microenv_data.shape[1]:
                    print(f"WARNING: Grid dimensions ({expected_points}) don't match data points ({microenv_data.shape[1]})")
                    print("Adjusting x,y,z coordinates to ensure proper grid dimensions")
                    
                    # Try to infer correct dimensions from data shape
                    # Assuming cubic domain if unclear
                    total_points = microenv_data.shape[1]
                    cube_dimension = int(np.cbrt(total_points))
                    
                    # If the data looks close to a cube
                    if abs(cube_dimension**3 - total_points) < 0.01 * total_points:
                        print(f"Adjusting to cubic grid: {cube_dimension}x{cube_dimension}x{cube_dimension}")
                        # Generate evenly spaced coordinates
                        x_min, x_max = microenv_data[0, :].min(), microenv_data[0, :].max()
                        y_min, y_max = microenv_data[1, :].min(), microenv_data[1, :].max()
                        z_min, z_max = microenv_data[2, :].min(), microenv_data[2, :].max()
                        
                        x_coords = np.linspace(x_min, x_max, cube_dimension)
                        y_coords = np.linspace(y_min, y_max, cube_dimension)
                        z_coords = np.linspace(z_min, z_max, cube_dimension)
                
                # Store mesh data
                self.xyz_mesh = (x_coords, y_coords, z_coords)
                nx, ny, nz = len(x_coords), len(y_coords), len(z_coords)
                
                # Store mesh dimensions and bounds
                self.mesh_data = {
                    'dimensions': (nx, ny, nz),
                    'bounds': (
                        x_coords.min(), x_coords.max(),
                        y_coords.min(), y_coords.max(),
                        z_coords.min(), z_coords.max()
                    )
                }
                
                # Extract substrates (skip first 4 rows which are x,y,z,time coordinates)
                num_substrates = microenv_data.shape[0] - 4
                
                # Create substrate names
                self.substrate_names = [f"Substrate {i}" for i in range(num_substrates)]
                
                # Extract each substrate and reshape to 3D grid
                self.substrate_data = []
                for i in range(num_substrates):
                    # Extract substrate data (offset by 4 to skip x,y,z,time coordinates)
                    substrate_data = microenv_data[i+4, :]
                    substrate_min = substrate_data.min()
                    substrate_max = substrate_data.max()
                    print(f"Substrate {i} range: {substrate_min} to {substrate_max}")
                    
                    # Check if this is a near-constant substrate
                    is_constant = np.isclose(substrate_min, substrate_max, rtol=1e-10, atol=1e-10)
                    
                    # Create an empty 3D array
                    grid_data = np.zeros((nx, ny, nz))
                    
                    if is_constant:
                        # For constant values, just fill the grid with that value
                        print(f"Substrate {i} has constant value {substrate_min}, filling grid directly")
                        grid_data.fill(substrate_min)
                    else:
                        # Map flat data into structured grid
                        coord_map = {}
                        # Pre-compute coordinate indices for faster lookup
                        for x_idx, x in enumerate(x_coords):
                            for y_idx, y in enumerate(y_coords):
                                for z_idx, z in enumerate(z_coords):
                                    coord_map[(x, y, z)] = (x_idx, y_idx, z_idx)
                        
                        # Process each data point
                        print(f"Mapping data points for Substrate {i}...")
                        mapped_points = 0
                        for idx in range(microenv_data.shape[1]):
                            x = microenv_data[0, idx]
                            y = microenv_data[1, idx]
                            z = microenv_data[2, idx]
                            
                            # Find closest coordinates if exact match not found
                            x_idx = np.abs(x_coords - x).argmin()
                            y_idx = np.abs(y_coords - y).argmin()
                            z_idx = np.abs(z_coords - z).argmin()
                            
                            # Set value
                            grid_data[x_idx, y_idx, z_idx] = substrate_data[idx]
                            mapped_points += 1
                        
                        print(f"Mapped {mapped_points} points for Substrate {i}")
                    
                    self.substrate_data.append(grid_data)
                
                # Use specified substrate ID but make sure it's valid
                print(f"Setting current substrate index to {substrate_id}")
                self.current_substrate_index = min(substrate_id, num_substrates - 1)
                
                # Print debug info about the selected substrate
                print(f"Using substrate index: {self.current_substrate_index}")
                if self.current_substrate_index < len(self.substrate_data):
                    selected_data = self.substrate_data[self.current_substrate_index]
                    print(f"Selected substrate data range: {np.min(selected_data)} to {np.max(selected_data)}")
                
                # Set opacity
                self.opacity_function = self._create_default_opacity_function()
                
                # Store for slice visualization
                self._create_vtk_grid()

    def clear_visualization(self):
        """Remove all visualization actors from the renderer"""
        # Clear existing actors
        if hasattr(self, 'volume_actor') and self.volume_actor:
            self.renderer.RemoveActor(self.volume_actor)
            self.volume_actor = None
            
        if hasattr(self, 'outline_actor') and self.outline_actor:
            self.renderer.RemoveActor(self.outline_actor)
            self.outline_actor = None
            
        if hasattr(self, 'grid_actor') and self.grid_actor:
            self.renderer.RemoveActor(self.grid_actor)
            self.grid_actor = None
            
        # Clear info actors (like text and scalar bars)
        if hasattr(self, 'info_actors'):
            for actor in self.info_actors:
                self.renderer.RemoveActor(actor)
            self.info_actors = []
        else:
            self.info_actors = [] 