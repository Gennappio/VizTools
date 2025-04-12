"""
CellVisualizer module for visualizing cell data from PhysiCell.
"""

import vtk
import numpy as np
import os
import xml.etree.ElementTree as ET
import scipy.io as sio
import math


class CellVisualizer:
    """Class for visualizing cell data from PhysiCell simulations."""
    
    def __init__(self, renderer):
        """
        Initialize the cell visualizer.
        
        Args:
            renderer: VTK renderer to which cell actors will be added
        """
        self.renderer = renderer
        self.cell_actors = []
        self.cell_data = None
        self.selected_cell_id = None
        self.cell_colors = {}  # Map cell types to colors
        self.cell_scalar_array = None
        self.cell_id_to_actor = {}  # Map cell IDs to actors for selection
        
        # Default colors for cell types
        self.default_colors = {
            0: [0.5, 0.5, 1.0],  # Default blue
            1: [1.0, 0.0, 0.0],  # Red
            2: [0.0, 1.0, 0.0],  # Green
            3: [1.0, 1.0, 0.0],  # Yellow
            4: [1.0, 0.5, 0.0],  # Orange
            5: [0.5, 0.0, 0.5],  # Purple
        }
        
        # Default opacity for cells
        self.opacity = 0.7
        
        # Default resolution for cell spheres
        self.sphere_resolution = 12
        
    def clear(self):
        """Remove all cell actors from the renderer."""
        for actor in self.cell_actors:
            self.renderer.RemoveActor(actor)
        self.cell_actors = []
        self.cell_id_to_actor = {}
        self.cell_data = None
        
    def set_cell_colors(self, colors):
        """
        Set custom colors for cell types.
        
        Args:
            colors: Dictionary mapping cell types to RGB colors
        """
        self.cell_colors = colors
        
    def get_color_for_cell_type(self, cell_type):
        """
        Get the color for a specific cell type.
        
        Args:
            cell_type: Integer representing the cell type
            
        Returns:
            RGB color list for the specified cell type
        """
        if cell_type in self.cell_colors:
            return self.cell_colors[cell_type]
        elif cell_type in self.default_colors:
            return self.default_colors[cell_type]
        else:
            # Return a hash-based color for unknown types
            hash_val = hash(str(cell_type)) % 1000
            r = (hash_val % 255) / 255.0
            g = ((hash_val * 7) % 255) / 255.0
            b = ((hash_val * 13) % 255) / 255.0
            return [r, g, b]
    
    def set_opacity(self, opacity):
        """
        Set the opacity for all cell actors.
        
        Args:
            opacity: Float between 0.0 and 1.0
        """
        self.opacity = opacity
        for actor in self.cell_actors:
            actor.GetProperty().SetOpacity(opacity)
            
    def set_sphere_resolution(self, resolution):
        """
        Set the resolution for cell spheres.
        
        Args:
            resolution: Integer representing the sphere resolution
        """
        self.sphere_resolution = resolution
        
    def visualize_cells_from_xml(self, xml_file):
        """
        Visualize cells from a PhysiCell XML file.
        
        Args:
            xml_file: Path to the XML file containing cell data
            
        Returns:
            True if visualization was successful, False otherwise
        """
        if not os.path.exists(xml_file):
            print(f"XML file not found: {xml_file}")
            return False
            
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Clear existing cells
            self.clear()
            
            # Find all cell elements
            cells = root.findall('.//cell')
            if not cells:
                print("No cells found in XML file")
                return False
                
            print(f"Found {len(cells)} cells in {xml_file}")
            self.cell_data = []
            
            # Process each cell
            for cell in cells:
                cell_id = int(cell.get('ID', '-1'))
                cell_type = int(cell.get('type', '0'))
                
                # Get position
                position = cell.find('position')
                if position is None:
                    continue
                    
                x = float(position.get('x', '0'))
                y = float(position.get('y', '0'))
                z = float(position.get('z', '0'))
                
                # Get custom data (optional)
                custom_data = {}
                custom_data_elem = cell.find('custom')
                if custom_data_elem is not None:
                    for var in custom_data_elem.findall('var'):
                        name = var.get('name', '')
                        value = float(var.get('value', '0'))
                        custom_data[name] = value
                
                # Add cell data
                cell_info = {
                    'id': cell_id,
                    'type': cell_type,
                    'position': [x, y, z],
                    'custom_data': custom_data,
                }
                
                # Check for radius/volume information
                radius = 5.0  # Default radius
                volume_elem = cell.find('phenotype/volume')
                if volume_elem is not None:
                    total_volume = float(volume_elem.find('total').get('value', '0'))
                    # Calculate radius from volume (assuming spherical cells)
                    if total_volume > 0:
                        radius = (total_volume * 0.75 / math.pi) ** (1/3)
                
                cell_info['radius'] = radius
                self.cell_data.append(cell_info)
                
                # Create and add a sphere for this cell
                self._add_cell_sphere(cell_info)
            
            self.renderer.ResetCamera()
            return True
            
        except Exception as e:
            print(f"Error visualizing XML file: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def visualize_cells_from_mat(self, mat_file):
        """
        Visualize cells from a PhysiCell MAT file.
        
        Args:
            mat_file: Path to the MAT file containing cell data
            
        Returns:
            True if visualization was successful, False otherwise
        """
        if not os.path.exists(mat_file):
            print(f"MAT file not found: {mat_file}")
            return False
            
        try:
            # Load MAT file
            mat_data = sio.loadmat(mat_file)
            
            # Clear existing cells
            self.clear()
            
            # Check if 'cells' exists in the MAT file
            if 'cells' not in mat_data:
                print("No 'cells' array found in MAT file")
                return False
                
            cells = mat_data['cells']
            
            # Handle PhysiCell MAT format (87 attributes per cell)
            if len(cells.shape) == 2 and cells.shape[0] == 87:
                self.cell_data = []
                
                # Transpose if needed for proper indexing
                if cells.shape[1] > cells.shape[0]:
                    cells = cells.T
                
                # Get number of cells
                num_cells = cells.shape[1]
                print(f"Found {num_cells} cells in {mat_file}")
                
                for i in range(num_cells):
                    # Extract position (indices 1, 2, 3)
                    x = float(cells[1, i])
                    y = float(cells[2, i])
                    z = float(cells[3, i])
                    
                    # Extract volume and calculate radius (index 4)
                    volume = float(cells[4, i])
                    radius = (3.0 * volume / (4.0 * math.pi)) ** (1.0/3.0)
                    
                    # Extract cell type (index 5)
                    cell_type = int(cells[5, i]) if cells.shape[0] > 5 else 0
                    
                    # Extract cell ID (index 0)
                    cell_id = int(cells[0, i])
                    
                    # Create cell info dictionary
                    cell_info = {
                        'id': cell_id,
                        'type': cell_type,
                        'position': [x, y, z],
                        'radius': radius,
                        'volume': volume,
                        'custom_data': {}
                    }
                    
                    # Add more properties if needed
                    # Examples of indices in PhysiCell cells matrix:
                    # 6: phase
                    # 7: elapsed time in phase
                    # 8-10: nuclear position (x,y,z)
                    # 11: nuclear volume
                    # ...and so on
                    
                    self.cell_data.append(cell_info)
                    
                    # Create and add a sphere for this cell
                    self._add_cell_sphere(cell_info)
                
                self.renderer.ResetCamera()
                return True
                
            # Standard cells array with cell data
            elif len(cells.shape) == 1 and cells.shape[0] == 87:
                # Single cell with 87 attributes
                self.cell_data = []
                
                # Extract position (indices 1, 2, 3)
                x = float(cells[1])
                y = float(cells[2])
                z = float(cells[3])
                
                # Extract volume and calculate radius (index 4)
                volume = float(cells[4])
                radius = (3.0 * volume / (4.0 * math.pi)) ** (1.0/3.0)
                
                # Extract cell type (index 5)
                cell_type = int(cells[5]) if cells.shape[0] > 5 else 0
                
                # Extract cell ID (index 0)
                cell_id = int(cells[0])
                
                # Create cell info dictionary
                cell_info = {
                    'id': cell_id,
                    'type': cell_type,
                    'position': [x, y, z],
                    'radius': radius,
                    'volume': volume,
                    'custom_data': {}
                }
                
                self.cell_data = [cell_info]
                
                # Create and add a sphere for this cell
                self._add_cell_sphere(cell_info)
                
                print(f"Visualized single cell from {mat_file}")
                print(f"Cell properties: ID={cell_id}, Type={cell_type}, Position=({x}, {y}, {z}), Radius={radius}")
                
                self.renderer.ResetCamera()
                return True
            
            else:
                print(f"Unexpected cells shape: {cells.shape}")
                return False
                
        except Exception as e:
            print(f"Error visualizing MAT file: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _add_cell_sphere(self, cell_info):
        """
        Add a sphere actor representing a cell to the renderer.
        
        Args:
            cell_info: Dictionary containing cell properties
        """
        # Create sphere source
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(cell_info['position'])
        sphere.SetRadius(cell_info['radius'])
        sphere.SetPhiResolution(self.sphere_resolution)
        sphere.SetThetaResolution(self.sphere_resolution)
        sphere.Update()
        
        # Create mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())
        
        # Create actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        # Set color based on cell type
        color = self.get_color_for_cell_type(cell_info['type'])
        actor.GetProperty().SetColor(color)
        actor.GetProperty().SetOpacity(self.opacity)
        
        # Set specular properties for better rendering
        actor.GetProperty().SetSpecular(0.3)
        actor.GetProperty().SetSpecularPower(30)
        
        # Store cell ID with actor (for selection)
        actor.SetProperty(vtk.vtkProperty())
        actor.GetProperty().SetEdgeVisibility(False)
        
        # Add to renderer
        self.renderer.AddActor(actor)
        
        # Store actor
        self.cell_actors.append(actor)
        self.cell_id_to_actor[cell_info['id']] = actor
        
    def get_cell_info(self, cell_id):
        """
        Get information about a specific cell.
        
        Args:
            cell_id: ID of the cell to get info for
            
        Returns:
            Dictionary with cell information or None if not found
        """
        if not self.cell_data:
            return None
            
        for cell in self.cell_data:
            if cell['id'] == cell_id:
                return cell
                
        return None
        
    def select_cell(self, cell_id):
        """
        Highlight a selected cell.
        
        Args:
            cell_id: ID of the cell to select
            
        Returns:
            True if the cell was found and selected, False otherwise
        """
        # Reset previous selection
        if self.selected_cell_id is not None and self.selected_cell_id in self.cell_id_to_actor:
            prev_actor = self.cell_id_to_actor[self.selected_cell_id]
            prev_actor.GetProperty().SetEdgeVisibility(False)
            
        # Set new selection
        if cell_id in self.cell_id_to_actor:
            self.selected_cell_id = cell_id
            actor = self.cell_id_to_actor[cell_id]
            actor.GetProperty().SetEdgeVisibility(True)
            actor.GetProperty().SetEdgeColor(1, 1, 1)  # White highlight
            actor.GetProperty().SetLineWidth(2)
            return True
            
        self.selected_cell_id = None
        return False
        
    def get_cell_count(self):
        """Get the number of cells currently visualized."""
        return len(self.cell_data) if self.cell_data else 0 