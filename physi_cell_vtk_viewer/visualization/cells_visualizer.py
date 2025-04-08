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
    
    def clear(self):
        """Remove all cell actors from the renderer"""
        for actor in self.cell_actors:
            self.renderer.RemoveActor(actor)
        self.cell_actors = []
        
        for actor in self.info_actors:
            self.renderer.RemoveActor(actor)
        self.info_actors = []
    
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
    
    def set_visibility(self, visible):
        """Set the visibility of all cell actors"""
        for actor in self.cell_actors:
            actor.SetVisibility(visible)
        for actor in self.info_actors:
            actor.SetVisibility(visible)
    
    def set_opacity(self, opacity):
        """Set the opacity of all cell actors"""
        opacity_value = opacity / 100.0  # Convert from slider value
        for actor in self.cell_actors:
            actor.GetProperty().SetOpacity(opacity_value) 