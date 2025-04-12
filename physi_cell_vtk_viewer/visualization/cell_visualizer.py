"""
Cell visualization module for PhysiCell data
"""

import vtk
import numpy as np


class CellVisualizer:
    """
    Class for visualizing cells from PhysiCell data
    """
    
    def __init__(self, renderer):
        """
        Initialize the cell visualizer
        
        Parameters:
        -----------
        renderer : vtkRenderer
            The VTK renderer to use for visualization
        """
        self.renderer = renderer
        self.cell_actors = []
        self.cell_data = None
        self.cell_type_colors = {}
        self.default_cell_color = [1.0, 0.0, 0.0]  # Default red color
        self.show_nuclei = False
        self.show_axes = True
        self.cell_opacity = 1.0
        
        # Setup default cell type colors
        self._setup_default_cell_colors()
        
        # Add cell type legend
        self.legend_actor = None
        
    def _setup_default_cell_colors(self):
        """Set up default colors for different cell types"""
        # Default color scheme for PhysiCell cell types
        # These can be overridden by the user
        self.cell_type_colors = {
            0: [0.5, 0.5, 0.5],   # Default gray
            1: [1.0, 0.0, 0.0],   # Type 1: Red
            2: [0.0, 1.0, 0.0],   # Type 2: Green
            3: [0.0, 0.0, 1.0],   # Type 3: Blue
            4: [1.0, 1.0, 0.0],   # Type 4: Yellow
            5: [1.0, 0.0, 1.0],   # Type 5: Magenta
            6: [0.0, 1.0, 1.0],   # Type 6: Cyan
            7: [1.0, 0.5, 0.0],   # Type 7: Orange
            8: [0.5, 0.0, 1.0],   # Type 8: Purple
            9: [0.0, 0.5, 0.0],   # Type 9: Dark green
            10: [0.5, 0.5, 1.0],  # Type 10: Light blue
        }
    
    def set_cell_color(self, cell_type, color):
        """
        Set the color for a specific cell type
        
        Parameters:
        -----------
        cell_type : int
            The cell type ID
        color : list
            RGB color values as [r, g, b] with values from 0.0 to 1.0
        """
        self.cell_type_colors[cell_type] = color
        
        # Update existing cells if they exist
        self._update_cell_colors()
    
    def _update_cell_colors(self):
        """Update the colors of existing cell actors"""
        if not self.cell_data or not self.cell_actors:
            return
            
        for i, cell in enumerate(self.cell_data.get('cells', [])):
            if i < len(self.cell_actors):
                cell_type = cell.get('type', 0)
                color = self.cell_type_colors.get(cell_type, self.default_cell_color)
                
                # Set the cell color
                self.cell_actors[i].GetProperty().SetColor(color)
    
    def set_opacity(self, opacity):
        """
        Set the opacity for all cells
        
        Parameters:
        -----------
        opacity : float
            Opacity value between 0.0 (transparent) and 1.0 (opaque)
        """
        self.cell_opacity = opacity
        
        # Update existing actors
        for actor in self.cell_actors:
            actor.GetProperty().SetOpacity(opacity)
    
    def toggle_nuclei(self, show):
        """
        Toggle showing cell nuclei
        
        Parameters:
        -----------
        show : bool
            Whether to show nuclei
        """
        self.show_nuclei = show
        
        # If we have cell data, re-visualize with the new setting
        if self.cell_data:
            self.visualize_cells(self.cell_data)
    
    def toggle_axes(self, show):
        """
        Toggle showing axes
        
        Parameters:
        -----------
        show : bool
            Whether to show axes
        """
        self.show_axes = show
        
        # Remove existing axes if present
        for actor in self.renderer.GetActors():
            if hasattr(actor, 'is_axis') and actor.is_axis:
                self.renderer.RemoveActor(actor)
        
        # Add axes if needed
        if show:
            self._add_axes()
    
    def clear(self):
        """Clear all cell actors from the renderer"""
        for actor in self.cell_actors:
            self.renderer.RemoveActor(actor)
        
        # Remove legend if present
        if self.legend_actor:
            self.renderer.RemoveActor(self.legend_actor)
            self.legend_actor = None
            
        self.cell_actors = []
        self.cell_data = None
        
        # Force renderer update
        self.renderer.GetRenderWindow().Render()
    
    def _create_sphere(self, x, y, z, radius):
        """Create a VTK sphere at the specified position with the given radius"""
        sphere_source = vtk.vtkSphereSource()
        sphere_source.SetCenter(x, y, z)
        sphere_source.SetRadius(radius)
        sphere_source.SetPhiResolution(20)
        sphere_source.SetThetaResolution(20)
        sphere_source.Update()
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere_source.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        return actor
    
    def _add_axes(self):
        """Add coordinate axes to the visualization"""
        # Create axes actors
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(100, 100, 100)
        axes.SetShaftType(0)
        axes.SetAxisLabels(1)
        axes.SetCylinderRadius(0.02)
        axes.is_axis = True  # Custom property for identification
        
        # Add actor to renderer
        self.renderer.AddActor(axes)
    
    def _update_legend(self):
        """Create or update the cell type legend"""
        # Remove existing legend if present
        if self.legend_actor:
            self.renderer.RemoveActor(self.legend_actor)
        
        # Create a new legend
        legend = vtk.vtkLegendBoxActor()
        legend.SetNumberOfEntries(len(self.cell_type_colors))
        
        # Add each cell type to legend
        i = 0
        for cell_type, color in self.cell_type_colors.items():
            # Create a symbol for this cell type
            sphere = vtk.vtkSphereSource()
            sphere.SetRadius(0.5)
            sphere.SetPhiResolution(10)
            sphere.SetThetaResolution(10)
            sphere.Update()
            
            # Add to legend
            legend.SetEntry(i, sphere.GetOutput(), f"Type {cell_type}", color)
            i += 1
        
        # Set legend properties
        legend.SetPadding(2)
        legend.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        legend.GetPositionCoordinate().SetValue(0.8, 0.8)
        legend.GetPosition2Coordinate().SetCoordinateSystemToNormalizedViewport()
        legend.GetPosition2Coordinate().SetValue(0.2, 0.2)
        
        # Store and add the legend
        self.legend_actor = legend
        self.renderer.AddActor(legend)
    
    def visualize_cells(self, cell_data):
        """
        Visualize cells from PhysiCell data
        
        Parameters:
        -----------
        cell_data : dict
            Dictionary containing cell data with 'cells' key holding a list of cell properties
        """
        # Store the cell data for later reference
        self.cell_data = cell_data
        
        # Clear existing cells
        self.clear()
        
        # If no data or no cells, return
        if not cell_data or 'cells' not in cell_data or not cell_data['cells']:
            return
        
        # Process each cell
        for cell in cell_data['cells']:
            # Extract cell position
            position = cell.get('position')
            if position is None:
                continue
                
            x, y, z = position
            
            # Extract or calculate cell radius
            radius = cell.get('radius')
            if radius is None:
                # Try to calculate from volume if available
                volume = cell.get('volume')
                if volume:
                    radius = (3.0 * volume / (4.0 * np.pi)) ** (1.0/3.0)
                else:
                    # Default radius
                    radius = 10.0
            
            # Create cell representation
            cell_actor = self._create_sphere(x, y, z, radius)
            
            # Set cell color based on type
            cell_type = cell.get('type', 0)
            color = self.cell_type_colors.get(cell_type, self.default_cell_color)
            cell_actor.GetProperty().SetColor(color)
            
            # Set opacity
            cell_actor.GetProperty().SetOpacity(self.cell_opacity)
            
            # Add actor to renderer and store
            self.renderer.AddActor(cell_actor)
            self.cell_actors.append(cell_actor)
            
            # Add nucleus if enabled
            if self.show_nuclei:
                nuclear_radius = radius * 0.5  # Typical nucleus is about half cell radius
                nucleus_actor = self._create_sphere(x, y, z, nuclear_radius)
                nucleus_color = [min(c + 0.2, 1.0) for c in color]  # Lighter color for nucleus
                nucleus_actor.GetProperty().SetColor(nucleus_color)
                self.renderer.AddActor(nucleus_actor)
                self.cell_actors.append(nucleus_actor)
        
        # Add axes if enabled
        if self.show_axes:
            self._add_axes()
            
        # Update legend
        self._update_legend()
        
        # Force renderer update
        self.renderer.GetRenderWindow().Render()
    
    def visualize_cells_from_mat(self, cell_mat_data):
        """
        Visualize cells from a PhysiCell MAT file
        
        Parameters:
        -----------
        cell_mat_data : dict
            Dictionary containing cell data from a MAT file
        """
        # Check if we have cell data
        if not cell_mat_data or 'cells' not in cell_mat_data:
            return
            
        cells_matrix = cell_mat_data['cells']
        
        # Process the cells matrix
        # For PhysiCell, this is typically a (87 x N) matrix where:
        # - Each column represents a cell
        # - Each row represents a property
        # - Rows 1-3 are x,y,z position
        # - Row 4 is total volume
        
        # Convert to proper format for visualization
        cell_data = {'cells': []}
        
        # Check if we have a single cell or multiple cells
        if cells_matrix.ndim == 1 or (cells_matrix.ndim == 2 and cells_matrix.shape[1] == 1):
            # Single cell with 87 properties
            # Extract the properties we need
            try:
                if cells_matrix.ndim == 1:
                    # Handle 1D array
                    x = cells_matrix[1]
                    y = cells_matrix[2]
                    z = cells_matrix[3]
                    volume = cells_matrix[4]
                    cell_type = int(cells_matrix[5]) if len(cells_matrix) > 5 else 0
                else:
                    # Handle 2D array with shape (87, 1)
                    x = cells_matrix[1, 0]
                    y = cells_matrix[2, 0]
                    z = cells_matrix[3, 0]
                    volume = cells_matrix[4, 0]
                    cell_type = int(cells_matrix[5, 0]) if cells_matrix.shape[0] > 5 else 0
                
                # Calculate radius from volume
                radius = (3.0 * volume / (4.0 * np.pi)) ** (1.0/3.0)
                
                # Create cell entry
                cell = {
                    'position': np.array([x, y, z]),
                    'radius': radius,
                    'volume': volume,
                    'type': cell_type
                }
                
                cell_data['cells'].append(cell)
                
            except IndexError as e:
                print(f"Error extracting cell properties: {e}")
                return
        
        else:
            # Multiple cells
            # Each column is a cell, rows are properties
            for i in range(cells_matrix.shape[1]):
                try:
                    x = cells_matrix[1, i]
                    y = cells_matrix[2, i]
                    z = cells_matrix[3, i]
                    volume = cells_matrix[4, i]
                    cell_type = int(cells_matrix[5, i]) if cells_matrix.shape[0] > 5 else 0
                    
                    # Calculate radius from volume
                    radius = (3.0 * volume / (4.0 * np.pi)) ** (1.0/3.0)
                    
                    # Create cell entry
                    cell = {
                        'position': np.array([x, y, z]),
                        'radius': radius,
                        'volume': volume,
                        'type': cell_type
                    }
                    
                    cell_data['cells'].append(cell)
                    
                except IndexError as e:
                    print(f"Error extracting properties for cell {i}: {e}")
                    continue
        
        # Now visualize the cells
        self.visualize_cells(cell_data) 