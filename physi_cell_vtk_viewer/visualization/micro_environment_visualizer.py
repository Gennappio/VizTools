"""
Microenvironment visualization module for PhysiCell data
"""

import numpy as np
import vtk


class MicroEnvironmentVisualizer:
    """
    Class for visualizing microenvironment data from PhysiCell
    """
    
    def __init__(self, renderer):
        """
        Initialize the microenvironment visualizer
        
        Parameters:
        -----------
        renderer : vtkRenderer
            The VTK renderer to use for visualization
        """
        self.renderer = renderer
        self.actors = []
        self.outline_actor = None
        self.grid_actors = []
        self.volume_actor = None
        self.scalar_bar = None
        self.data = None
        
        # Visualization properties
        self.opacity = 0.7
        self.wireframe_visible = False
        self.show_scalar_bar = True
        self.current_substrate = 0
        self.substrate_names = []
        self.color_map = "rainbow"  # Default color map
        
        # Available color maps
        self.color_maps = {
            "rainbow": self._create_rainbow_lut,
            "jet": self._create_jet_lut,
            "viridis": self._create_viridis_lut,
            "plasma": self._create_plasma_lut,
            "red_blue": self._create_red_blue_lut,
            "cool_warm": self._create_cool_warm_lut,
            "custom": self._create_custom_lut,
        }
        
        # Initialize lookup table
        self.lut = self._create_rainbow_lut()
    
    def clear(self):
        """Clear all visualization elements from the renderer"""
        # Remove all existing actors
        for actor in self.actors:
            self.renderer.RemoveActor(actor)
        
        # Remove outline actor
        if self.outline_actor:
            self.renderer.RemoveActor(self.outline_actor)
            
        # Remove grid actors
        for actor in self.grid_actors:
            self.renderer.RemoveActor(actor)
            
        # Remove volume actor
        if self.volume_actor:
            self.renderer.RemoveActor(self.volume_actor)
            
        # Remove scalar bar
        if self.scalar_bar:
            self.renderer.RemoveActor(self.scalar_bar)
        
        # Clear actor lists and references
        self.actors = []
        self.grid_actors = []
        self.outline_actor = None
        self.volume_actor = None
        self.scalar_bar = None
        
        # Force render update
        self.renderer.GetRenderWindow().Render()
    
    def set_colormap(self, colormap_name):
        """
        Set the colormap to use for the visualization
        
        Parameters:
        -----------
        colormap_name : str
            The name of the colormap to use
        """
        if colormap_name in self.color_maps:
            self.color_map = colormap_name
            self.lut = self.color_maps[colormap_name]()
            
            # Update visualization if data is available
            if self.data is not None:
                self.visualize_substrate(self.current_substrate)
    
    def set_opacity(self, opacity):
        """
        Set the opacity for the volume rendering
        
        Parameters:
        -----------
        opacity : float
            Opacity value between 0.0 (transparent) and 1.0 (opaque)
        """
        self.opacity = opacity
        
        # Update volume actor if it exists
        if self.volume_actor:
            volume_property = self.volume_actor.GetProperty()
            # Update all opacity gradient points
            for i in range(volume_property.GetScalarOpacityFunction().GetSize()):
                x, y = volume_property.GetScalarOpacityFunction().GetNodeValue(i)
                # Scale y by our opacity factor
                volume_property.GetScalarOpacityFunction().AddPoint(x, y * self.opacity)
            
            # Force update
            self.renderer.GetRenderWindow().Render()
    
    def toggle_wireframe(self, show):
        """
        Toggle the wireframe visualization
        
        Parameters:
        -----------
        show : bool
            Whether to show the wireframe
        """
        self.wireframe_visible = show
        
        # Update outline and grid if they exist
        if self.outline_actor:
            self.outline_actor.SetVisibility(show)
            
        for actor in self.grid_actors:
            actor.SetVisibility(show)
            
        # Reduce volume opacity when wireframe is visible
        if self.volume_actor:
            if show:
                # Reduce opacity when wireframe is visible
                volume_property = self.volume_actor.GetProperty()
                for i in range(volume_property.GetScalarOpacityFunction().GetSize()):
                    x, y = volume_property.GetScalarOpacityFunction().GetNodeValue(i)
                    # Scale y by a reduced factor for wireframe visibility
                    volume_property.GetScalarOpacityFunction().AddPoint(x, y * 0.5)
            else:
                # Restore original opacity
                volume_property = self.volume_actor.GetProperty()
                for i in range(volume_property.GetScalarOpacityFunction().GetSize()):
                    x, y = volume_property.GetScalarOpacityFunction().GetNodeValue(i)
                    # Restore to original opacity
                    volume_property.GetScalarOpacityFunction().AddPoint(x, y * (self.opacity / 0.5))
        
        # Force update
        self.renderer.GetRenderWindow().Render()
    
    def toggle_scalar_bar(self, show):
        """
        Toggle the scalar bar visualization
        
        Parameters:
        -----------
        show : bool
            Whether to show the scalar bar
        """
        self.show_scalar_bar = show
        
        if self.scalar_bar:
            self.scalar_bar.SetVisibility(show)
            
            # Force update
            self.renderer.GetRenderWindow().Render()
    
    def _create_rainbow_lut(self):
        """Create a rainbow lookup table"""
        lut = vtk.vtkLookupTable()
        lut.SetHueRange(0.667, 0.0)  # Blue to red
        lut.SetSaturationRange(1.0, 1.0)
        lut.SetValueRange(1.0, 1.0)
        lut.SetNumberOfColors(256)
        lut.Build()
        return lut
    
    def _create_jet_lut(self):
        """Create a jet-like lookup table"""
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(256)
        
        # Jet colormap approximation (blue-cyan-green-yellow-red)
        for i in range(256):
            if i < 64:
                r, g, b = 0, 0, 0.5 + i/64
            elif i < 128:
                r, g, b = 0, (i-64)/64, 1
            elif i < 192:
                r, g, b = (i-128)/64, 1, 1 - (i-128)/64
            else:
                r, g, b = 1, 1 - (i-192)/64, 0
            
            lut.SetTableValue(i, r, g, b, 1.0)
            
        lut.Build()
        return lut
    
    def _create_viridis_lut(self):
        """Create a viridis-like lookup table"""
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(256)
        
        # Approximation of viridis colormap
        colors = [
            [68, 1, 84],      # Dark purple
            [70, 50, 126],    # Purple
            [54, 92, 141],    # Blue
            [39, 127, 142],   # Teal
            [31, 161, 135],   # Green
            [74, 194, 109],   # Light green
            [159, 218, 58],   # Yellow-green
            [253, 231, 37]    # Yellow
        ]
        
        # Normalize colors to 0-1 range
        colors = [[r/255.0, g/255.0, b/255.0] for r, g, b in colors]
        
        # Interpolate between colors
        for i in range(256):
            t = i / 255.0
            idx = int(t * (len(colors) - 1))
            frac = t * (len(colors) - 1) - idx
            
            if idx < len(colors) - 1:
                r = colors[idx][0] + frac * (colors[idx+1][0] - colors[idx][0])
                g = colors[idx][1] + frac * (colors[idx+1][1] - colors[idx][1])
                b = colors[idx][2] + frac * (colors[idx+1][2] - colors[idx][2])
            else:
                r, g, b = colors[-1]
                
            lut.SetTableValue(i, r, g, b, 1.0)
            
        lut.Build()
        return lut
    
    def _create_plasma_lut(self):
        """Create a plasma-like lookup table"""
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(256)
        
        # Approximation of plasma colormap
        colors = [
            [13, 8, 135],      # Dark blue
            [84, 2, 163],      # Purple
            [139, 10, 165],    # Magenta
            [185, 50, 137],    # Pink
            [219, 92, 104],    # Light pink
            [244, 136, 73],    # Orange
            [254, 188, 43],    # Light orange
            [240, 249, 33]     # Yellow
        ]
        
        # Normalize colors to 0-1 range
        colors = [[r/255.0, g/255.0, b/255.0] for r, g, b in colors]
        
        # Interpolate between colors
        for i in range(256):
            t = i / 255.0
            idx = int(t * (len(colors) - 1))
            frac = t * (len(colors) - 1) - idx
            
            if idx < len(colors) - 1:
                r = colors[idx][0] + frac * (colors[idx+1][0] - colors[idx][0])
                g = colors[idx][1] + frac * (colors[idx+1][1] - colors[idx][1])
                b = colors[idx][2] + frac * (colors[idx+1][2] - colors[idx][2])
            else:
                r, g, b = colors[-1]
                
            lut.SetTableValue(i, r, g, b, 1.0)
            
        lut.Build()
        return lut
    
    def _create_red_blue_lut(self):
        """Create a simple blue-to-red lookup table"""
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(256)
        
        for i in range(256):
            t = i / 255.0
            # Blue to red
            r = t
            g = 0
            b = 1 - t
            lut.SetTableValue(i, r, g, b, 1.0)
            
        lut.Build()
        return lut
    
    def _create_cool_warm_lut(self):
        """Create a cool-to-warm diverging lookup table"""
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(256)
        
        for i in range(256):
            t = i / 255.0
            
            if t < 0.5:
                # Cool colors (blue to white)
                s = t * 2
                r = s
                g = s
                b = 1.0
            else:
                # Warm colors (white to red)
                s = (t - 0.5) * 2
                r = 1.0
                g = 1.0 - s
                b = 1.0 - s
                
            lut.SetTableValue(i, r, g, b, 1.0)
            
        lut.Build()
        return lut
    
    def _create_custom_lut(self):
        """Create a custom lookup table (can be modified as needed)"""
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(256)
        
        # Example: Green to yellow to red
        for i in range(256):
            t = i / 255.0
            
            if t < 0.5:
                # Green to yellow
                r = 2 * t
                g = 1.0
                b = 0.0
            else:
                # Yellow to red
                r = 1.0
                g = 1.0 - 2 * (t - 0.5)
                b = 0.0
                
            lut.SetTableValue(i, r, g, b, 1.0)
            
        lut.Build()
        return lut
    
    def _create_scalar_bar(self, title):
        """Create a scalar bar actor for the visualization"""
        scalar_bar = vtk.vtkScalarBarActor()
        scalar_bar.SetLookupTable(self.lut)
        scalar_bar.SetTitle(title)
        scalar_bar.SetNumberOfLabels(5)
        scalar_bar.SetWidth(0.08)
        scalar_bar.SetHeight(0.5)
        scalar_bar.SetPosition(0.9, 0.25)
        scalar_bar.GetLabelTextProperty().SetColor(0, 0, 0)
        scalar_bar.GetTitleTextProperty().SetColor(0, 0, 0)
        
        return scalar_bar
    
    def _create_wireframe(self, grid, bounds):
        """
        Create wireframe visualization for the grid
        
        Parameters:
        -----------
        grid : vtkRectilinearGrid
            The grid to create wireframe for
        bounds : list
            The bounds of the grid [xmin, xmax, ymin, ymax, zmin, zmax]
        
        Returns:
        --------
        tuple
            Tuple containing the outline actor and list of grid actors
        """
        # Create outline for the domain
        outline_filter = vtk.vtkRectilinearGridOutlineFilter()
        outline_filter.SetInputData(grid)
        
        outline_mapper = vtk.vtkPolyDataMapper()
        outline_mapper.SetInputConnection(outline_filter.GetOutputPort())
        
        outline_actor = vtk.vtkActor()
        outline_actor.SetMapper(outline_mapper)
        outline_actor.GetProperty().SetColor(1, 1, 1)  # White outline
        outline_actor.GetProperty().SetLineWidth(2.0)
        
        # Create grid lines
        grid_actors = []
        
        # X grid planes
        x_planes = vtk.vtkPlanes()
        x_planes_pts = vtk.vtkPoints()
        
        x_extent = grid.GetExtent()
        x_coords = [grid.GetXCoordinate(i) for i in range(x_extent[1] + 1)]
        
        # Only show a reasonable number of grid lines
        stride = max(1, len(x_coords) // 10)
        for i in range(0, len(x_coords), stride):
            x = x_coords[i]
            # Create a plane at this X coordinate
            plane = vtk.vtkPlane()
            plane.SetOrigin(x, bounds[2], bounds[4])
            plane.SetNormal(1, 0, 0)
            
            # Cut the grid with this plane
            cutter = vtk.vtkCutter()
            cutter.SetInputData(grid)
            cutter.SetCutFunction(plane)
            
            # Create mapper and actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(cutter.GetOutputPort())
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.7, 0.7, 0.7)  # Light gray
            actor.GetProperty().SetOpacity(0.3)
            
            grid_actors.append(actor)
        
        # Y grid planes
        y_extent = grid.GetExtent()
        y_coords = [grid.GetYCoordinate(i) for i in range(y_extent[3] + 1)]
        
        stride = max(1, len(y_coords) // 10)
        for i in range(0, len(y_coords), stride):
            y = y_coords[i]
            # Create a plane at this Y coordinate
            plane = vtk.vtkPlane()
            plane.SetOrigin(bounds[0], y, bounds[4])
            plane.SetNormal(0, 1, 0)
            
            # Cut the grid with this plane
            cutter = vtk.vtkCutter()
            cutter.SetInputData(grid)
            cutter.SetCutFunction(plane)
            
            # Create mapper and actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(cutter.GetOutputPort())
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.7, 0.7, 0.7)  # Light gray
            actor.GetProperty().SetOpacity(0.3)
            
            grid_actors.append(actor)
        
        # Z grid planes
        z_extent = grid.GetExtent()
        z_coords = [grid.GetZCoordinate(i) for i in range(z_extent[5] + 1)]
        
        stride = max(1, len(z_coords) // 10)
        for i in range(0, len(z_coords), stride):
            z = z_coords[i]
            # Create a plane at this Z coordinate
            plane = vtk.vtkPlane()
            plane.SetOrigin(bounds[0], bounds[2], z)
            plane.SetNormal(0, 0, 1)
            
            # Cut the grid with this plane
            cutter = vtk.vtkCutter()
            cutter.SetInputData(grid)
            cutter.SetCutFunction(plane)
            
            # Create mapper and actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(cutter.GetOutputPort())
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.7, 0.7, 0.7)  # Light gray
            actor.GetProperty().SetOpacity(0.3)
            
            grid_actors.append(actor)
        
        return outline_actor, grid_actors
    
    def visualize_microenvironment(self, micro_env_data):
        """
        Visualize microenvironment data
        
        Parameters:
        -----------
        micro_env_data : dict
            Dictionary containing microenvironment data with:
            - 'shape': (x, y, z) shape of the grid
            - 'origin': (x, y, z) origin coordinates
            - 'spacing': (dx, dy, dz) spacing between grid points
            - 'substrates': list of substrate data arrays
            - 'substrate_names': list of substrate names
        """
        # Clear previous visualization
        self.clear()
        
        # Store data for later reference
        self.data = micro_env_data
        
        # Check if we have valid data
        if not micro_env_data or 'substrates' not in micro_env_data or len(micro_env_data['substrates']) == 0:
            print("No microenvironment data to visualize")
            return
        
        # Extract data from the dictionary
        shape = micro_env_data.get('shape', (0, 0, 0))
        origin = micro_env_data.get('origin', (0, 0, 0))
        spacing = micro_env_data.get('spacing', (1, 1, 1))
        substrates = micro_env_data.get('substrates', [])
        self.substrate_names = micro_env_data.get('substrate_names', [])
        
        if len(shape) != 3 or shape[0] == 0 or shape[1] == 0 or shape[2] == 0:
            print("Invalid grid shape:", shape)
            return
        
        if len(substrates) == 0:
            print("No substrate data to visualize")
            return
        
        # Default to first substrate if names list is empty
        if not self.substrate_names:
            self.substrate_names = [f"Substrate {i}" for i in range(len(substrates))]
        
        # Create a rectilinear grid
        grid = vtk.vtkRectilinearGrid()
        
        # Set grid dimensions
        grid.SetDimensions(shape)
        
        # Create coordinate arrays
        x_coords = vtk.vtkDoubleArray()
        y_coords = vtk.vtkDoubleArray()
        z_coords = vtk.vtkDoubleArray()
        
        # Fill coordinate arrays
        for i in range(shape[0]):
            x_coords.InsertNextValue(origin[0] + i * spacing[0])
        
        for j in range(shape[1]):
            y_coords.InsertNextValue(origin[1] + j * spacing[1])
        
        for k in range(shape[2]):
            z_coords.InsertNextValue(origin[2] + k * spacing[2])
        
        # Set coordinates
        grid.SetXCoordinates(x_coords)
        grid.SetYCoordinates(y_coords)
        grid.SetZCoordinates(z_coords)
        
        # Calculate bounds
        bounds = [
            origin[0], origin[0] + (shape[0] - 1) * spacing[0],
            origin[1], origin[1] + (shape[1] - 1) * spacing[1],
            origin[2], origin[2] + (shape[2] - 1) * spacing[2]
        ]
        
        # Create wireframe visualization (outline and grid)
        self.outline_actor, self.grid_actors = self._create_wireframe(grid, bounds)
        
        # Set wireframe visibility based on current state
        self.outline_actor.SetVisibility(self.wireframe_visible)
        for actor in self.grid_actors:
            actor.SetVisibility(self.wireframe_visible)
        
        # Add actors to renderer
        self.renderer.AddActor(self.outline_actor)
        for actor in self.grid_actors:
            self.renderer.AddActor(actor)
        
        # Visualize the first substrate by default
        if len(substrates) > 0:
            self.visualize_substrate(0)
        
        # Force renderer update
        self.renderer.GetRenderWindow().Render()
    
    def visualize_substrate(self, substrate_idx):
        """
        Visualize a specific substrate from the microenvironment
        
        Parameters:
        -----------
        substrate_idx : int
            Index of the substrate to visualize
        """
        # Store current substrate index
        self.current_substrate = substrate_idx
        
        # Check if we have valid data
        if not self.data or 'substrates' not in self.data or len(self.data['substrates']) == 0:
            print("No microenvironment data to visualize")
            return
        
        # Check if substrate index is valid
        if substrate_idx < 0 or substrate_idx >= len(self.data['substrates']):
            print(f"Invalid substrate index: {substrate_idx}")
            return
        
        # Extract substrate data
        substrate_data = self.data['substrates'][substrate_idx]
        substrate_name = self.substrate_names[substrate_idx] if substrate_idx < len(self.substrate_names) else f"Substrate {substrate_idx}"
        
        # Extract grid information
        shape = self.data.get('shape', (0, 0, 0))
        origin = self.data.get('origin', (0, 0, 0))
        spacing = self.data.get('spacing', (1, 1, 1))
        
        # Create a structured points dataset for the volume
        volume_data = vtk.vtkStructuredPoints()
        volume_data.SetDimensions(shape)
        volume_data.SetOrigin(origin)
        volume_data.SetSpacing(spacing)
        
        # Create scalar data for the volume
        scalars = vtk.vtkDoubleArray()
        scalars.SetNumberOfValues(shape[0] * shape[1] * shape[2])
        
        # Fill scalar data (assuming substrate_data is a flattened 3D array)
        for i in range(shape[0] * shape[1] * shape[2]):
            if i < len(substrate_data):
                scalars.SetValue(i, substrate_data[i])
            else:
                scalars.SetValue(i, 0.0)
        
        # Add scalars to volume data
        volume_data.GetPointData().SetScalars(scalars)
        
        # Remove existing volume actor and scalar bar if they exist
        if self.volume_actor:
            self.renderer.RemoveActor(self.volume_actor)
        
        if self.scalar_bar:
            self.renderer.RemoveActor(self.scalar_bar)
        
        # Create a volume mapper
        volume_mapper = vtk.vtkSmartVolumeMapper()
        volume_mapper.SetInputData(volume_data)
        
        # Create transfer functions for opacity and color
        color_transfer_function = vtk.vtkColorTransferFunction()
        opacity_transfer_function = vtk.vtkPiecewiseFunction()
        
        # Get data range
        data_range = [np.min(substrate_data), np.max(substrate_data)]
        if data_range[0] == data_range[1]:
            # Avoid division by zero
            data_range[1] = data_range[0] + 1.0
        
        # Setup transfer functions
        # Use our lookup table for colors
        for i in range(256):
            val = data_range[0] + (i/255.0) * (data_range[1] - data_range[0])
            r, g, b, _ = self.lut.GetTableValue(i)
            color_transfer_function.AddRGBPoint(val, r, g, b)
        
        # Set opacity transfer function
        # Linear opacity from min to max
        opacity_transfer_function.AddPoint(data_range[0], 0.0)
        opacity_transfer_function.AddPoint(data_range[0] + 0.25 * (data_range[1] - data_range[0]), 0.05 * self.opacity)
        opacity_transfer_function.AddPoint(data_range[0] + 0.5 * (data_range[1] - data_range[0]), 0.2 * self.opacity)
        opacity_transfer_function.AddPoint(data_range[1], self.opacity)
        
        # Create volume properties
        volume_property = vtk.vtkVolumeProperty()
        volume_property.SetColor(color_transfer_function)
        volume_property.SetScalarOpacity(opacity_transfer_function)
        volume_property.SetInterpolationTypeToLinear()
        volume_property.ShadeOn()
        volume_property.SetAmbient(0.4)
        volume_property.SetDiffuse(0.6)
        volume_property.SetSpecular(0.2)
        
        # Create volume actor
        volume_actor = vtk.vtkVolume()
        volume_actor.SetMapper(volume_mapper)
        volume_actor.SetProperty(volume_property)
        
        # Store volume actor
        self.volume_actor = volume_actor
        
        # If wireframe is visible, reduce volume opacity
        if self.wireframe_visible:
            # Reduce opacity for better wireframe visibility
            for i in range(opacity_transfer_function.GetSize()):
                x, y = opacity_transfer_function.GetNodeValue(i)
                opacity_transfer_function.AddPoint(x, y * 0.5)
        
        # Add volume actor to renderer
        self.renderer.AddActor(self.volume_actor)
        
        # Create scalar bar
        self.scalar_bar = self._create_scalar_bar(substrate_name)
        self.scalar_bar.SetVisibility(self.show_scalar_bar)
        
        # Add scalar bar to renderer
        self.renderer.AddActor(self.scalar_bar)
        
        # Force renderer update
        self.renderer.GetRenderWindow().Render() 