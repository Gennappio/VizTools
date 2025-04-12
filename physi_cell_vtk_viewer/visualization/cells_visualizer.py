"""
Visualization components for PhysiCell cell data
"""

import vtk
import numpy as np
from physi_cell_vtk_viewer.utils.vtk_utils import create_cell_color, create_cell_rgb_color, calculate_cell_radius

class CellsVisualizer:
    """Class for visualizing PhysiCell cell data"""
    
    def __init__(self, renderer):
        """Initialize with a VTK renderer"""
        self.renderer = renderer
        self.cell_actors = []
        self.info_actors = []
        self.cell_properties = {}
        self.visible = True
        self.opacity = 1.0
        
        # Create lookup table for cell coloring
        self.lut = vtk.vtkLookupTable()
        self.setup_default_lut()
    
    def setup_default_lut(self):
        """Set up default lookup table for cell types"""
        # Set up a colorful lookup table for cell types
        self.lut.SetNumberOfTableValues(20)
        self.lut.SetTableRange(0, 19)
        self.lut.SetNanColor(0.7, 0.7, 0.7, 0.7)  # Gray for NaN
        
        # Default cell type colors (index 0-19)
        # These match PhysiCell GUI color scheme
        cell_colors = [
            (0.5, 0.5, 0.5),    # 0: Gray (default)
            (1.0, 0.0, 0.0),    # 1: Red (cancer cell)
            (1.0, 0.4, 0.0),    # 2: Orange
            (0.0, 1.0, 0.0),    # 3: Green
            (0.0, 0.0, 1.0),    # 4: Blue
            (1.0, 1.0, 0.0),    # 5: Yellow
            (1.0, 0.0, 1.0),    # 6: Magenta
            (0.0, 1.0, 1.0),    # 7: Cyan
            (0.5, 0.0, 0.0),    # 8: Dark red
            (0.0, 0.5, 0.0),    # 9: Dark green
            (0.0, 0.0, 0.5),    # 10: Dark blue
            (0.5, 0.5, 0.0),    # 11: Olive
            (0.5, 0.0, 0.5),    # 12: Purple
            (0.0, 0.5, 0.5),    # 13: Teal
            (0.3, 0.6, 0.9),    # 14: Light blue
            (0.9, 0.6, 0.3),    # 15: Tan
            (0.6, 0.3, 0.9),    # 16: Lavender
            (0.9, 0.3, 0.6),    # 17: Pink
            (0.4, 0.8, 0.2),    # 18: Light green
            (0.8, 0.4, 0.2),    # 19: Brown
        ]
        
        # Set the colors in the lookup table
        for i, color in enumerate(cell_colors):
            r, g, b = color
            self.lut.SetTableValue(i, r, g, b, 1.0)
        
        self.lut.Build()
    
    def setup_variable_lut(self, min_val, max_val):
        """Set up continuous lookup table for a variable range"""
        # Create a new lookup table for continuous variables
        self.var_lut = vtk.vtkLookupTable()
        self.var_lut.SetNumberOfTableValues(256)
        self.var_lut.SetHueRange(0.667, 0.0)  # Blue to red
        self.var_lut.SetTableRange(min_val, max_val)
        self.var_lut.SetNanColor(0.7, 0.7, 0.7, 0.7)  # Gray for NaN
        self.var_lut.Build()
        
        return self.var_lut
    
    def clear(self):
        """Remove all cell actors from the renderer"""
        for actor in self.cell_actors:
            self.renderer.RemoveActor(actor)
        self.cell_actors = []
        
        for actor in self.info_actors:
            self.renderer.RemoveActor(actor)
        self.info_actors = []
        
        self.cell_properties = {}
    
    def visualize_single_cell(self, cells_data, opacity=0.7):
        """Visualize a single cell from cell data"""
        # Extract cell position (typically stored at indices 1, 2, 3)
        x = float(cells_data[1, 0])
        y = float(cells_data[2, 0])
        z = float(cells_data[3, 0])
        
        # Extract cell radius (convert from volume)
        volume = float(cells_data[4, 0])  # Assuming volume is at index 4
        radius = calculate_cell_radius(volume)
        
        # Cell type
        cell_type = 1  # Default type if not specified
        if cells_data.shape[0] > 5:
            cell_type = int(cells_data[5, 0])  # Assuming type is at index 5
        
        # Create a sphere for the cell
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(x, y, z)
        sphere.SetRadius(radius)  # Use actual radius without scaling down
        sphere.SetPhiResolution(16)
        sphere.SetThetaResolution(16)
        sphere.Update()
        
        # Create a mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())
        
        # Create an actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        # Set color based on cell type
        color = create_cell_color(cell_type)
        actor.GetProperty().SetColor(color)
        
        # Set opacity
        actor.GetProperty().SetOpacity(opacity)
        
        # Add the actor to the renderer
        self.renderer.AddActor(actor)
        self.cell_actors.append(actor)
        
        # Add info text about the cell
        text_actor = vtk.vtkTextActor()
        text_actor.SetInput(f"Single cell at ({x:.2f}, {y:.2f}, {z:.2f})\nRadius: {radius:.2f}, Type: {cell_type}")
        text_actor.GetTextProperty().SetFontSize(16)
        text_actor.GetTextProperty().SetColor(1, 1, 0)  # Yellow text
        text_actor.SetPosition(20, 30)
        self.renderer.AddActor2D(text_actor)
        self.info_actors.append(text_actor)
        
        # Store cell properties for future reference
        self.cell_properties[actor] = {
            'position': (x, y, z),
            'radius': radius,
            'cell_type': cell_type,
            'volume': volume,
            'data': cells_data
        }
        
        return True
    
    def visualize_multiple_cells(self, cells_data, opacity=0.7):
        """Visualize multiple cells from cell data"""
        num_cells = cells_data.shape[1]
        
        # Create a polydata object to hold all cells
        all_points = vtk.vtkPoints()
        all_cells = vtk.vtkCellArray()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName("Colors")
        
        # Process each cell
        for cell_idx in range(num_cells):
            # Extract cell position (typically stored at indices 1, 2, 3)
            x = float(cells_data[1, cell_idx])
            y = float(cells_data[2, cell_idx])
            z = float(cells_data[3, cell_idx])
            
            # Extract cell radius (convert from volume)
            volume = float(cells_data[4, cell_idx])  # Assuming volume is at index 4
            radius = calculate_cell_radius(volume)
            
            # Cell type
            cell_type = 1  # Default type if not specified
            if cells_data.shape[0] > 5:
                cell_type = int(cells_data[5, cell_idx])  # Assuming type is at index 5
            
            # Create sphere for this cell
            sphere_source = vtk.vtkSphereSource()
            sphere_source.SetCenter(x, y, z)
            sphere_source.SetRadius(radius)  # Use actual radius without scaling
            sphere_source.SetPhiResolution(16)
            sphere_source.SetThetaResolution(16)
            sphere_source.Update()
            
            # Get polydata from the sphere source
            sphere_polydata = sphere_source.GetOutput()
            
            # Get number of points in the current polydata
            num_points = all_points.GetNumberOfPoints()
            
            # Get cell color based on type
            color = create_cell_rgb_color(cell_type)
            
            # Add sphere points to the combined polydata
            for j in range(sphere_polydata.GetNumberOfPoints()):
                point = sphere_polydata.GetPoint(j)
                all_points.InsertNextPoint(point)
                
                # Add the same color for each point of this cell
                colors.InsertNextTuple3(*color)
            
            # Add sphere cells (polygons) to the combined polydata
            for j in range(sphere_polydata.GetNumberOfCells()):
                cell = sphere_polydata.GetCell(j)
                polygon = vtk.vtkPolygon()
                polygon.GetPointIds().SetNumberOfIds(cell.GetNumberOfPoints())
                
                for k in range(cell.GetNumberOfPoints()):
                    polygon.GetPointIds().SetId(k, cell.GetPointId(k) + num_points)
                
                all_cells.InsertNextCell(polygon)
        
        # Create combined polydata for all cells
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(all_points)
        polydata.SetPolys(all_cells)
        polydata.GetPointData().SetScalars(colors)
        
        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        # Set opacity from slider
        actor.GetProperty().SetOpacity(opacity)
        
        # Add the actor to the renderer
        self.renderer.AddActor(actor)
        self.cell_actors.append(actor)
        
        # Add info text about the cells
        text_actor = vtk.vtkTextActor()
        text_actor.SetInput(f"Multiple cells: {num_cells} total cells")
        text_actor.GetTextProperty().SetFontSize(16)
        text_actor.GetTextProperty().SetColor(1, 1, 0)  # Yellow text
        text_actor.SetPosition(20, 30)
        self.renderer.AddActor2D(text_actor)
        self.info_actors.append(text_actor)
        
        # Store cell properties for future reference
        for i in range(num_cells):
            cell_data = cells_data[:, i:i+1]
            self.cell_properties[actor] = {
                'position': (float(cell_data[1, 0]), float(cell_data[2, 0]), float(cell_data[3, 0])),
                'radius': calculate_cell_radius(float(cell_data[4, 0])),
                'cell_type': int(cell_data[5, 0]) if cell_data.shape[0] > 5 else 0,
                'volume': float(cell_data[4, 0]),
                'data': cell_data
            }
        
        return True
    
    def visualize_cells_from_xml(self, mcds, opacity=0.7):
        """Visualize cells from a pyMCDS_cells object"""
        # Get cell positions and types
        cell_df = mcds.get_cell_df()
        positions_x = cell_df['position_x'].values
        positions_y = cell_df['position_y'].values
        
        if 'position_z' in cell_df.columns:
            positions_z = cell_df['position_z'].values
        else:
            positions_z = np.zeros_like(positions_x)
        
        # Make sure there are cells to visualize
        if len(positions_x) == 0:
            return False
        
        # Create a polydata object to hold all cells
        all_points = vtk.vtkPoints()
        all_cells = vtk.vtkCellArray()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName("Colors")
        
        # Process each cell
        for idx in range(len(positions_x)):
            x = positions_x[idx]
            y = positions_y[idx]
            z = positions_z[idx]
            
            # Get cell type if available
            cell_type = 1  # Default
            if 'cell_type' in cell_df.columns:
                cell_type = int(cell_df['cell_type'].values[idx])
            
            # Get cell radius if available, otherwise use a default
            radius = 10.0  # Default radius
            if 'total_volume' in cell_df.columns:
                volume = cell_df['total_volume'].values[idx]
                radius = calculate_cell_radius(volume)
            
            # Create sphere for this cell
            sphere_source = vtk.vtkSphereSource()
            sphere_source.SetCenter(x, y, z)
            sphere_source.SetRadius(radius)
            sphere_source.SetPhiResolution(16)
            sphere_source.SetThetaResolution(16)
            sphere_source.Update()
            
            # Get polydata from the sphere source
            sphere_polydata = sphere_source.GetOutput()
            
            # Get number of points in the current polydata
            num_points = all_points.GetNumberOfPoints()
            
            # Get cell color based on type
            color = create_cell_rgb_color(cell_type)
            
            # Add sphere points to the combined polydata
            for j in range(sphere_polydata.GetNumberOfPoints()):
                point = sphere_polydata.GetPoint(j)
                all_points.InsertNextPoint(point)
                
                # Add the same color for each point of this cell
                colors.InsertNextTuple3(*color)
            
            # Add sphere cells (polygons) to the combined polydata
            for j in range(sphere_polydata.GetNumberOfCells()):
                cell = sphere_polydata.GetCell(j)
                polygon = vtk.vtkPolygon()
                polygon.GetPointIds().SetNumberOfIds(cell.GetNumberOfPoints())
                
                for k in range(cell.GetNumberOfPoints()):
                    polygon.GetPointIds().SetId(k, cell.GetPointId(k) + num_points)
                
                all_cells.InsertNextCell(polygon)
        
        # Create combined polydata for all cells
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(all_points)
        polydata.SetPolys(all_cells)
        polydata.GetPointData().SetScalars(colors)
        
        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        # Set opacity
        actor.GetProperty().SetOpacity(opacity)
        
        # Add the actor to the renderer
        self.renderer.AddActor(actor)
        self.cell_actors.append(actor)
        
        # Add info text about the cells
        text_actor = vtk.vtkTextActor()
        text_actor.SetInput(f"XML cells: {len(positions_x)} total cells")
        text_actor.GetTextProperty().SetFontSize(16)
        text_actor.GetTextProperty().SetColor(1, 1, 0)  # Yellow text
        text_actor.SetPosition(20, 60)
        self.renderer.AddActor2D(text_actor)
        self.info_actors.append(text_actor)
        
        # Store cell properties for future reference
        for idx in range(len(positions_x)):
            cell_data = np.array([[positions_x[idx], positions_y[idx], positions_z[idx], 0, 0, 0]])
            self.cell_properties[actor] = {
                'position': (positions_x[idx], positions_y[idx], positions_z[idx]),
                'radius': calculate_cell_radius(0),
                'cell_type': 0,
                'volume': 0,
                'data': cell_data
            }
        
        return True
    
    def create_spiral_viz_from_values(self, values, opacity=0.7):
        """Create a spiral visualization from scalar values (for non-standard cells data)"""
        # Create a spiral layout to visualize the values
        n_values = len(values)
        
        # Set up constants for the spiral
        radius_step = 0.5  # Distance between spiral turns
        phi_step = 0.3     # Angle step between points
        height_scale = 2.0  # Scaling factor for height
        
        # Create a polydata object to hold the spiral
        points = vtk.vtkPoints()
        
        # Create cells for lines
        lines = vtk.vtkCellArray()
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(n_values)
        
        # Create scalar data for coloring
        scalar_data = vtk.vtkDoubleArray()
        scalar_data.SetName("Values")
        
        # Generate points on a spiral
        radius = 1.0
        phi = 0.0
        min_val = min(values)
        max_val = max(values)
        val_range = max_val - min_val if max_val > min_val else 1.0
        
        for i in range(n_values):
            # Calculate position on spiral
            x = radius * np.cos(phi)
            y = radius * np.sin(phi)
            
            # Normalize value for z-height
            norm_val = (values[i] - min_val) / val_range if val_range > 0 else 0.5
            z = norm_val * height_scale
            
            # Add point
            points.InsertNextPoint(x, y, z)
            
            # Set point ID in line
            line.GetPointIds().SetId(i, i)
            
            # Add scalar value
            scalar_data.InsertNextValue(values[i])
            
            # Update radius and angle for next point
            radius += radius_step / (2 * np.pi)
            phi += phi_step
        
        # Add the line to the cell array
        lines.InsertNextCell(line)
        
        # Create polydata for the spiral
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetLines(lines)
        polydata.GetPointData().SetScalars(scalar_data)
        
        # Create tubes around the lines for better visualization
        tube_filter = vtk.vtkTubeFilter()
        tube_filter.SetInputData(polydata)
        tube_filter.SetRadius(0.1)
        tube_filter.SetNumberOfSides(12)
        tube_filter.Update()
        
        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tube_filter.GetOutputPort())
        mapper.SetScalarRange(min_val, max_val)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(opacity)
        
        # Add the actor to the renderer
        self.renderer.AddActor(actor)
        self.cell_actors.append(actor)
        
        # Also add spheres at each data point
        for i in range(n_values):
            sphere = vtk.vtkSphereSource()
            point = points.GetPoint(i)
            sphere.SetCenter(point[0], point[1], point[2])
            sphere.SetRadius(0.15)
            sphere.SetPhiResolution(12)
            sphere.SetThetaResolution(12)
            sphere.Update()
            
            sphere_mapper = vtk.vtkPolyDataMapper()
            sphere_mapper.SetInputConnection(sphere.GetOutputPort())
            
            # Map value to color
            norm_val = (values[i] - min_val) / val_range if val_range > 0 else 0.5
            r = norm_val
            g = 0.5 * (1.0 - norm_val)
            b = 1.0 - norm_val
            
            sphere_actor = vtk.vtkActor()
            sphere_actor.SetMapper(sphere_mapper)
            sphere_actor.GetProperty().SetColor(r, g, b)
            sphere_actor.GetProperty().SetOpacity(opacity)
            
            self.renderer.AddActor(sphere_actor)
            self.cell_actors.append(sphere_actor)
        
        return self.cell_actors
    
    def visualize_cells_with_variable(self, cells_data, var_index, color_var_index, opacity=0.7, as_spheres=True, auto_range=True, min_val=0.0, max_val=1.0):
        """
        Visualize cells with a specific variable and color by another variable.
        
        Args:
            cells_data: The cells data matrix
            var_index: Index of the variable to visualize (determines position, size)
            color_var_index: Index of the variable to use for coloring
            opacity: Cell opacity (0-1)
            as_spheres: Whether to display cells as spheres (True) or points (False)
            auto_range: Whether to automatically determine color range
            min_val: Minimum value for color range (if auto_range is False)
            max_val: Maximum value for color range (if auto_range is False)
        """
        # Clear existing visualization
        self.clear()
        
        # We always use fixed indices for position (1,2,3) regardless of var_index
        # Extract cell positions (always at indices 1, 2, 3)
        positions = cells_data[1:4, :].T  # Transpose to get one position per row
        
        # Extract coloring variable
        color_values = cells_data[color_var_index, :]
        
        # Get data range for coloring
        self.color_min = np.min(color_values) if auto_range else min_val
        self.color_max = np.max(color_values) if auto_range else max_val
        
        # Store color values for later reference
        self.color_values = color_values
        self.color_var_index = color_var_index
        
        # Create a lookup table for coloring
        lut = vtk.vtkLookupTable()
        lut.SetHueRange(0.667, 0.0)  # Blue to red
        lut.SetRange(self.color_min, self.color_max)
        lut.Build()
        
        # Create points dataset
        points = vtk.vtkPoints()
        scalars = vtk.vtkDoubleArray()
        scalars.SetName("ColorValues")
        
        # Add all cell points with their actual positions
        for i in range(positions.shape[0]):
            pos = positions[i]
            points.InsertNextPoint(pos[0], pos[1], pos[2])
            scalars.InsertNextValue(color_values[i])
        
        # Create a polydata object with the points
        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        
        # Add the color values as scalars
        poly_data.GetPointData().SetScalars(scalars)
        
        if as_spheres:
            # Visualize as spheres
            # Create a sphere source to use as the glyph
            sphere = vtk.vtkSphereSource()
            sphere.SetRadius(1.0)  # Base radius, will be scaled
            sphere.SetPhiResolution(12)
            sphere.SetThetaResolution(12)
            
            # Create a glyph filter
            glyph = vtk.vtkGlyph3D()
            glyph.SetInputData(poly_data)
            glyph.SetSourceConnection(sphere.GetOutputPort())
            
            # Always get volumes from index 4 regardless of var_index
            volumes = cells_data[4, :]
            
            # Convert volumes to radii using volume formula for sphere
            radii = np.cbrt(volumes * 0.75 / np.pi)
            avg_radius = np.mean(radii)
            
            # Set fixed scaling based on average radius
            glyph.SetScaleModeToDataScalingOff()
            glyph.SetScaleFactor(avg_radius)
            
            # Create mapper and actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(glyph.GetOutputPort())
            mapper.ScalarVisibilityOn()
            mapper.SetLookupTable(lut)
            mapper.SetScalarRange(self.color_min, self.color_max)
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetOpacity(opacity)
            
            # Add to renderer
            self.renderer.AddActor(actor)
            self.cell_actors.append(actor)
        else:
            # Visualize as points
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(poly_data)
            mapper.ScalarVisibilityOn()
            mapper.SetLookupTable(lut)
            mapper.SetScalarRange(self.color_min, self.color_max)
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetOpacity(opacity)
            actor.GetProperty().SetPointSize(5)  # Larger points for visibility
            
            # Add to renderer
            self.renderer.AddActor(actor)
            self.cell_actors.append(actor)
        
        # Add a scalar bar for the color map
        scalar_bar = vtk.vtkScalarBarActor()
        scalar_bar.SetLookupTable(lut)
        scalar_bar.SetTitle(f"Cell Variable (index {color_var_index})")
        scalar_bar.SetNumberOfLabels(5)
        scalar_bar.SetPosition(0.05, 0.05)
        scalar_bar.SetWidth(0.15)
        scalar_bar.SetHeight(0.8)
        scalar_bar.GetTitleTextProperty().SetColor(1, 1, 1)
        scalar_bar.GetLabelTextProperty().SetColor(1, 1, 1)
        
        self.renderer.AddActor2D(scalar_bar)
        self.info_actors.append(scalar_bar)
        
        # Store cell properties for future reference - but safely to avoid index errors
        num_cells = cells_data.shape[1]
        # Store combined properties for the visualization instead of per-cell
        self.cell_properties[actor] = {
            'num_cells': num_cells,
            'mean_radius': avg_radius,
            'mean_position': tuple(np.mean(positions, axis=0)),
            'data_shape': cells_data.shape
        }
        
        return True
    
    def get_data_range(self):
        """Get the current data range for the color variable"""
        if hasattr(self, 'color_min') and hasattr(self, 'color_max'):
            return self.color_min, self.color_max
        elif hasattr(self, 'color_values') and len(self.color_values) > 0:
            return np.min(self.color_values), np.max(self.color_values)
        else:
            return 0.0, 1.0
    
    def set_visibility(self, visible):
        """Set the visibility of all cell actors"""
        self.visible = visible
        for actor in self.cell_actors:
            actor.SetVisibility(visible)
        for actor in self.info_actors:
            actor.SetVisibility(visible)
    
    def set_opacity(self, opacity):
        """Set the opacity of all cell actors"""
        opacity_value = opacity / 100.0  # Convert from slider value
        for actor in self.cell_actors:
            actor.GetProperty().SetOpacity(opacity_value)
    
    def get_visibility(self):
        """Get current visibility state"""
        return self.visible
    
    def clear(self):
        """Remove all cell actors from the renderer"""
        for actor in self.cell_actors:
            self.renderer.RemoveActor(actor)
        
        self.cell_actors = []
        self.cell_properties = {}
        for actor in self.info_actors:
            self.renderer.RemoveActor(actor)
        self.info_actors = [] 