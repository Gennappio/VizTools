"""
Data loader for PhysiCell output files
"""

import os
import re
import gzip
import numpy as np
import scipy.io as sio
from lxml import etree
import vtk
import xml.etree.ElementTree as ET


class DataLoader:
    """
    Class for loading and parsing PhysiCell output data
    """
    
    def __init__(self):
        """Initialize the data loader"""
        self.current_directory = None
        self.current_frame = None
        self.frames = []
        self.output_file_pattern = r'output(\d+).xml'
        self.mesh_file_pattern = r'mesh(\d+).mat'
        self.cells_file_pattern = r'cells(\d+).mat'
        
    def load_directory(self, directory_path):
        """
        Load all PhysiCell data from a directory
        
        Parameters:
        -----------
        directory_path : str
            Path to the directory containing PhysiCell output files
            
        Returns:
        --------
        list
            List of frame numbers found in the directory
        """
        if not os.path.isdir(directory_path):
            raise ValueError(f"Directory not found: {directory_path}")
        
        self.current_directory = directory_path
        self.frames = []
        
        # Find all output XML files in the directory
        xml_files = [f for f in os.listdir(directory_path) if f.startswith('output') and f.endswith('.xml')]
        
        # Extract frame numbers from file names
        for file_name in xml_files:
            match = re.match(self.output_file_pattern, file_name)
            if match:
                frame_number = int(match.group(1))
                self.frames.append(frame_number)
        
        # Sort frames
        self.frames.sort()
        
        return self.frames
    
    def get_frame_data(self, frame_number):
        """
        Get data for a specific frame
        
        Parameters:
        -----------
        frame_number : int
            Frame number to load
            
        Returns:
        --------
        dict
            Dictionary containing data for the specified frame
        """
        if not self.current_directory:
            raise ValueError("No directory loaded. Call load_directory first.")
        
        if frame_number not in self.frames:
            raise ValueError(f"Frame {frame_number} not found in loaded frames.")
        
        self.current_frame = frame_number
        
        # Construct file paths
        xml_file = os.path.join(self.current_directory, f"output{frame_number}.xml")
        mesh_file = os.path.join(self.current_directory, f"mesh{frame_number}.mat")
        cells_file = os.path.join(self.current_directory, f"cells{frame_number}.mat")
        
        # Initialize data dictionary
        data = {
            'frame': frame_number,
            'cells': None,
            'microenvironment': None,
            'metadata': {},
        }
        
        # Load XML file
        if os.path.exists(xml_file):
            try:
                xml_data = self._load_xml_file(xml_file)
                data.update(xml_data)
            except Exception as e:
                print(f"Error loading XML file: {e}")
        
        # Load mesh data
        if os.path.exists(mesh_file):
            try:
                mesh_data = self._load_mesh_file(mesh_file)
                data['microenvironment'] = mesh_data
            except Exception as e:
                print(f"Error loading mesh file: {e}")
        
        # Load cells data
        if os.path.exists(cells_file):
            try:
                cells_data = self._load_cells_file(cells_file)
                data['cells'] = cells_data
            except Exception as e:
                print(f"Error loading cells file: {e}")
        
        return data
    
    def _load_xml_file(self, file_path):
        """
        Load data from an XML file
        
        Parameters:
        -----------
        file_path : str
            Path to the XML file
            
        Returns:
        --------
        dict
            Dictionary containing data from the XML file
        """
        # Check if file is gzipped
        if file_path.endswith('.gz'):
            with gzip.open(file_path, 'rb') as f:
                tree = etree.parse(f)
        else:
            tree = etree.parse(file_path)
        
        root = tree.getroot()
        
        # Initialize data structures
        data = {
            'metadata': {},
            'cells': {},
            'microenvironment': {},
        }
        
        # Extract metadata
        metadata_node = root.find('metadata')
        if metadata_node is not None:
            for child in metadata_node:
                data['metadata'][child.tag] = child.text
        
        # Extract microenvironment data
        micro_node = root.find('microenvironment')
        if micro_node is not None:
            domain_node = micro_node.find('domain')
            if domain_node is not None:
                # Get mesh information
                mesh = {}
                mesh_node = domain_node.find('mesh')
                if mesh_node is not None:
                    x_coords = mesh_node.findtext('x_coordinates')
                    y_coords = mesh_node.findtext('y_coordinates')
                    z_coords = mesh_node.findtext('z_coordinates')
                    
                    if x_coords and y_coords and z_coords:
                        mesh['x'] = np.array([float(x) for x in x_coords.split()])
                        mesh['y'] = np.array([float(y) for y in y_coords.split()])
                        mesh['z'] = np.array([float(z) for z in z_coords.split()])
                        
                        mesh['shape'] = (len(mesh['x']), len(mesh['y']), len(mesh['z']))
                        mesh['origin'] = (mesh['x'][0], mesh['y'][0], mesh['z'][0])
                        
                        if len(mesh['x']) > 1 and len(mesh['y']) > 1 and len(mesh['z']) > 1:
                            mesh['spacing'] = (
                                (mesh['x'][-1] - mesh['x'][0]) / (len(mesh['x']) - 1),
                                (mesh['y'][-1] - mesh['y'][0]) / (len(mesh['y']) - 1),
                                (mesh['z'][-1] - mesh['z'][0]) / (len(mesh['z']) - 1)
                            )
                        else:
                            mesh['spacing'] = (1, 1, 1)
                        
                        data['microenvironment']['mesh'] = mesh
                
                # Get variables information
                variables = []
                variables_node = domain_node.find('variables')
                if variables_node is not None:
                    for var_node in variables_node.findall('variable'):
                        var = {
                            'name': var_node.findtext('name'),
                            'units': var_node.findtext('units'),
                        }
                        variables.append(var)
                    
                    data['microenvironment']['variables'] = variables
                    
                # Get data
                data_node = domain_node.find('data')
                if data_node is not None:
                    data_text = data_node.text
                    if data_text:
                        # Parse the data matrix
                        rows = data_text.strip().split('\n')
                        if rows:
                            data_matrix = []
                            for row in rows:
                                values = [float(v) for v in row.split()]
                                if values:
                                    data_matrix.append(values)
                            
                            data_matrix = np.array(data_matrix)
                            data['microenvironment']['data'] = data_matrix
        
        # Extract cellular data
        cells_node = root.find('cellular_information')
        if cells_node is not None:
            cell_populations = cells_node.find('cell_populations')
            if cell_populations is not None:
                cells = []
                for cell_pop in cell_populations:
                    for cell_node in cell_pop.findall('cell'):
                        cell = {'type': cell_pop.get('type')}
                        
                        # Extract custom data for each cell
                        phenotype_node = cell_node.find('phenotype')
                        if phenotype_node is not None:
                            for phenotype_element in phenotype_node:
                                cell[f"phenotype_{phenotype_element.tag}"] = phenotype_element.text
                        
                        # Extract position and other data
                        for data_element in cell_node:
                            if data_element.tag != 'phenotype':
                                cell[data_element.tag] = data_element.text
                        
                        # Convert position data to coordinates
                        if 'position' in cell:
                            pos_values = cell['position'].split()
                            if len(pos_values) >= 3:
                                cell['x'] = float(pos_values[0])
                                cell['y'] = float(pos_values[1])
                                cell['z'] = float(pos_values[2])
                        
                        # Convert volume to radius
                        if 'volume' in cell:
                            volume = float(cell['volume'])
                            # Assuming spherical cell: V = (4/3) * π * r³
                            cell['radius'] = (3 * volume / (4 * np.pi)) ** (1/3)
                        
                        cells.append(cell)
                
                data['cells'] = cells
        
        return data
    
    def _load_mesh_file(self, file_path):
        """
        Load mesh data from a MAT file
        
        Parameters:
        -----------
        file_path : str
            Path to the mesh MAT file
            
        Returns:
        --------
        dict
            Dictionary containing microenvironment data
        """
        try:
            mat_data = sio.loadmat(file_path)
            
            # Initialize microenvironment data structure
            micro_env = {}
            
            # Extract mesh size and coordinates
            if 'mesh' in mat_data:
                mesh_data = mat_data['mesh']
                
                if mesh_data.ndim >= 2:
                    # Extract X, Y, Z coordinates
                    x_coords = np.unique(mesh_data[:, 0])
                    y_coords = np.unique(mesh_data[:, 1])
                    z_coords = np.unique(mesh_data[:, 2])
                    
                    # Determine shape
                    shape = (len(x_coords), len(y_coords), len(z_coords))
                    
                    # Get origin
                    origin = (x_coords[0], y_coords[0], z_coords[0])
                    
                    # Calculate spacing
                    if len(x_coords) > 1 and len(y_coords) > 1 and len(z_coords) > 1:
                        spacing = (
                            (x_coords[-1] - x_coords[0]) / (len(x_coords) - 1),
                            (y_coords[-1] - y_coords[0]) / (len(y_coords) - 1),
                            (z_coords[-1] - z_coords[0]) / (len(z_coords) - 1)
                        )
                    else:
                        spacing = (1, 1, 1)
                    
                    micro_env['shape'] = shape
                    micro_env['origin'] = origin
                    micro_env['spacing'] = spacing
                    micro_env['x_coords'] = x_coords
                    micro_env['y_coords'] = y_coords
                    micro_env['z_coords'] = z_coords
            
            # Extract substrate data
            substrates = []
            substrate_names = []
            
            # Look for variables that contain substrate data
            for key in mat_data.keys():
                # Skip standard variables
                if key in ['__header__', '__version__', '__globals__', 'mesh']:
                    continue
                
                # Add substrate data
                substrate_names.append(key)
                substrate_data = mat_data[key]
                
                # Reshape data if needed
                if substrate_data.ndim == 2 and substrate_data.shape[1] == 1:
                    # Handle the case where data is a column vector
                    substrates.append(substrate_data[:, 0])
                else:
                    substrates.append(substrate_data.flatten())
            
            micro_env['substrates'] = substrates
            micro_env['substrate_names'] = substrate_names
            
            return micro_env
            
        except Exception as e:
            print(f"Error loading mesh file {file_path}: {e}")
            return None
    
    def _load_cells_file(self, file_path):
        """
        Load cells data from a MAT file
        
        Parameters:
        -----------
        file_path : str
            Path to the cells MAT file
            
        Returns:
        --------
        dict
            Dictionary containing cell data
        """
        try:
            mat_data = sio.loadmat(file_path)
            
            # Initialize cells data structure
            cells_data = {}
            
            # Check for cells data
            if 'cells' in mat_data:
                cells = mat_data['cells']
                
                # PhysiCell cells.mat file format has 87 attributes for each cell
                # The first few entries contain the most important information:
                # - index 0: ID
                # - indices 1-3: position (x,y,z)
                # - index 4: total volume
                # - index 5: cell type
                
                # Check dimensions to determine format
                if cells.ndim == 2:
                    # Multiple cells case (common format)
                    if cells.shape[0] == 87:
                        # 87 rows (attributes) x N columns (cells)
                        num_cells = cells.shape[1]
                        
                        # Create structured cell data
                        cell_list = []
                        
                        for i in range(num_cells):
                            cell = {
                                'ID': int(cells[0, i]),
                                'position': {
                                    'x': cells[1, i],
                                    'y': cells[2, i],
                                    'z': cells[3, i]
                                },
                                'volume': cells[4, i],
                                'type': int(cells[5, i]) if cells.shape[0] > 5 else 0,
                                # Calculate radius from volume (assuming spherical cell)
                                'radius': (3 * cells[4, i] / (4 * np.pi)) ** (1/3)
                            }
                            cell_list.append(cell)
                        
                        cells_data['cell_list'] = cell_list
                        cells_data['num_cells'] = num_cells
                        
                    elif cells.shape[1] == 87:
                        # N rows (cells) x 87 columns (attributes)
                        num_cells = cells.shape[0]
                        
                        # Create structured cell data
                        cell_list = []
                        
                        for i in range(num_cells):
                            cell = {
                                'ID': int(cells[i, 0]),
                                'position': {
                                    'x': cells[i, 1],
                                    'y': cells[i, 2],
                                    'z': cells[i, 3]
                                },
                                'volume': cells[i, 4],
                                'type': int(cells[i, 5]) if cells.shape[1] > 5 else 0,
                                # Calculate radius from volume (assuming spherical cell)
                                'radius': (3 * cells[i, 4] / (4 * np.pi)) ** (1/3)
                            }
                            cell_list.append(cell)
                        
                        cells_data['cell_list'] = cell_list
                        cells_data['num_cells'] = num_cells
                        
                    else:
                        # Unusual format - return raw data
                        cells_data['raw_data'] = cells
                
                elif cells.ndim == 1 and cells.size == 87:
                    # Single cell case (87 attributes for one cell)
                    cell = {
                        'ID': int(cells[0]),
                        'position': {
                            'x': cells[1],
                            'y': cells[2],
                            'z': cells[3]
                        },
                        'volume': cells[4],
                        'type': int(cells[5]) if cells.size > 5 else 0,
                        # Calculate radius from volume (assuming spherical cell)
                        'radius': (3 * cells[4] / (4 * np.pi)) ** (1/3)
                    }
                    
                    cells_data['cell_list'] = [cell]
                    cells_data['num_cells'] = 1
                
                else:
                    # Unusual format - return raw data
                    cells_data['raw_data'] = cells
            
            # Store raw data for reference
            cells_data['mat_data'] = mat_data
            
            return cells_data
            
        except Exception as e:
            print(f"Error loading cells file {file_path}: {e}")
            return None
    
    @staticmethod
    def cells_to_vtk_polydata(cells_data):
        """
        Convert cells data to VTK polydata for visualization
        
        Parameters:
        -----------
        cells_data : dict
            Dictionary containing cell data
            
        Returns:
        --------
        vtkPolyData
            VTK polydata representing the cells
        """
        if not cells_data or 'cell_list' not in cells_data:
            return None
        
        cell_list = cells_data['cell_list']
        
        # Create points for the cells
        points = vtk.vtkPoints()
        
        # Create cell array
        vertices = vtk.vtkCellArray()
        
        # Create arrays for cell properties
        id_array = vtk.vtkIntArray()
        id_array.SetName("ID")
        
        type_array = vtk.vtkIntArray()
        type_array.SetName("Type")
        
        radius_array = vtk.vtkFloatArray()
        radius_array.SetName("Radius")
        
        volume_array = vtk.vtkFloatArray()
        volume_array.SetName("Volume")
        
        # Add cells to the polydata
        for i, cell in enumerate(cell_list):
            # Add point
            points.InsertNextPoint(
                cell['position']['x'],
                cell['position']['y'],
                cell['position']['z']
            )
            
            # Add vertex
            vertex = vtk.vtkVertex()
            vertex.GetPointIds().SetId(0, i)
            vertices.InsertNextCell(vertex)
            
            # Add data
            id_array.InsertNextValue(cell['ID'])
            type_array.InsertNextValue(cell['type'])
            radius_array.InsertNextValue(cell['radius'])
            volume_array.InsertNextValue(cell['volume'])
        
        # Create polydata
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetVerts(vertices)
        
        # Add data arrays
        polydata.GetPointData().AddArray(id_array)
        polydata.GetPointData().AddArray(type_array)
        polydata.GetPointData().AddArray(radius_array)
        polydata.GetPointData().AddArray(volume_array)
        
        return polydata
    
    @staticmethod
    def microenvironment_to_vtk_grid(micro_env_data):
        """
        Convert microenvironment data to VTK grid for visualization
        
        Parameters:
        -----------
        micro_env_data : dict
            Dictionary containing microenvironment data
            
        Returns:
        --------
        vtkRectilinearGrid
            VTK rectilinear grid representing the microenvironment
        """
        if not micro_env_data:
            return None
        
        # Extract data
        shape = micro_env_data.get('shape', None)
        if not shape:
            return None
        
        # Create a rectilinear grid
        grid = vtk.vtkRectilinearGrid()
        
        # Set dimensions
        grid.SetDimensions(shape)
        
        # Create coordinate arrays
        x_coords = vtk.vtkDoubleArray()
        y_coords = vtk.vtkDoubleArray()
        z_coords = vtk.vtkDoubleArray()
        
        # Fill coordinate arrays with actual values if available
        if 'x_coords' in micro_env_data:
            for x in micro_env_data['x_coords']:
                x_coords.InsertNextValue(x)
        else:
            origin = micro_env_data.get('origin', (0, 0, 0))
            spacing = micro_env_data.get('spacing', (1, 1, 1))
            for i in range(shape[0]):
                x_coords.InsertNextValue(origin[0] + i * spacing[0])
        
        if 'y_coords' in micro_env_data:
            for y in micro_env_data['y_coords']:
                y_coords.InsertNextValue(y)
        else:
            origin = micro_env_data.get('origin', (0, 0, 0))
            spacing = micro_env_data.get('spacing', (1, 1, 1))
            for j in range(shape[1]):
                y_coords.InsertNextValue(origin[1] + j * spacing[1])
        
        if 'z_coords' in micro_env_data:
            for z in micro_env_data['z_coords']:
                z_coords.InsertNextValue(z)
        else:
            origin = micro_env_data.get('origin', (0, 0, 0))
            spacing = micro_env_data.get('spacing', (1, 1, 1))
            for k in range(shape[2]):
                z_coords.InsertNextValue(origin[2] + k * spacing[2])
        
        # Set coordinates
        grid.SetXCoordinates(x_coords)
        grid.SetYCoordinates(y_coords)
        grid.SetZCoordinates(z_coords)
        
        # Add substrate data as point data
        substrates = micro_env_data.get('substrates', [])
        substrate_names = micro_env_data.get('substrate_names', [])
        
        for i, substrate in enumerate(substrates):
            # Create array for substrate data
            substrate_array = vtk.vtkDoubleArray()
            
            # Set name
            name = f"Substrate {i}"
            if i < len(substrate_names):
                name = substrate_names[i]
            substrate_array.SetName(name)
            
            # Allocate memory for the array
            total_points = shape[0] * shape[1] * shape[2]
            substrate_array.SetNumberOfValues(total_points)
            
            # Fill the array
            if len(substrate) == total_points:
                # Data is already flattened
                for j in range(total_points):
                    substrate_array.SetValue(j, substrate[j])
            else:
                # Need to reshape data - fill with zeros
                for j in range(total_points):
                    substrate_array.SetValue(j, 0.0)
            
            # Add array to grid
            grid.GetPointData().AddArray(substrate_array)
        
        # Set active scalar
        if substrates:
            if substrate_names:
                grid.GetPointData().SetActiveScalars(substrate_names[0])
            else:
                grid.GetPointData().SetActiveScalars("Substrate 0")
        
        return grid

    def _map_custom_vars_to_indices(self, data, xml_file):
        """Map custom variable names from XML to indices in MAT file cells data"""
        if not os.path.exists(xml_file):
            return
            
        try:
            # Parse XML file
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Get cell data from the MAT file
            cells_data = data['cells_mat']['cells']
            
            # Standard variables take up first 17 indices
            next_idx = 17
            
            # Extract custom variable names 
            custom_vars = []
            
            # Check custom_data sections in cell_population
            for custom_section in root.findall('.//cellular_information//cell_population//custom_data'):
                for var in custom_section:
                    var_name = var.tag
                    if var_name:
                        custom_vars.append(var_name)
            
            # Print debug info
            print(f"Found custom variables in XML: {custom_vars}")
            print(f"Data shape from MAT file: {cells_data.shape}")
            
            # See if there are more variables in the MAT file than we have names for
            if cells_data.shape[0] > (next_idx + len(custom_vars)):
                # There are more variables than we have names for
                remaining = cells_data.shape[0] - (next_idx + len(custom_vars))
                print(f"Warning: {remaining} variables in MAT file don't have names")
                
                # Generate names for the remaining variables
                for i in range(remaining):
                    custom_vars.append(f"unknown_var_{next_idx + len(custom_vars)}")
            
            # Create a mapping of variable names to indices
            var_index_map = {}
            for i, var_name in enumerate(custom_vars):
                var_index_map[var_name] = next_idx + i
            
            # Store the mapping in the data
            if 'metadata' not in data['cells_mat']:
                data['cells_mat']['metadata'] = {}
            
            data['cells_mat']['metadata']['var_index_map'] = var_index_map
            print(f"Variable index map: {var_index_map}")
            
            # Explicitly check for cell_ATP_source
            if 'cell_ATP_source' in var_index_map:
                print(f"cell_ATP_source found at index {var_index_map['cell_ATP_source']}")
                
        except Exception as e:
            print(f"Error mapping custom variables: {e}")

    def load_frame(self, frame_number):
        """Load data for a specified frame"""
        if frame_number < 0 or frame_number > self.max_frame:
            return None
        
        # Construct file paths
        frame_str = f"{frame_number:08d}"
        xml_file = f"{self.output_dir}/output{frame_str}.xml"
        cells_file = f"{self.output_dir}/output{frame_str}_cells.mat"
        
        # Load data
        data = {
            'frame': frame_number,
            'cells_xml': None,
            'cells_mat': None,
            'microenv': None
        }
        
        # Try to load XML file
        if os.path.exists(xml_file):
            try:
                data['cells_xml'] = self._load_xml_file(xml_file)
            except Exception as e:
                if self.debug:
                    print(f"Error loading XML file: {e}")
        
        # Try to load cells MAT file
        if os.path.exists(cells_file):
            try:
                data['cells_mat'] = self._load_cells_mat_file(cells_file)
                
                # Map custom variables from XML to MAT file
                if 'cells_xml' in data and data['cells_xml'] and 'cells_mat' in data and 'cells' in data['cells_mat']:
                    self._map_custom_vars_to_indices(data, xml_file)
                
            except Exception as e:
                if self.debug:
                    print(f"Error loading cells MAT file: {e}")
        
        # Try to load microenvironment data from XML
        if 'cells_xml' in data and data['cells_xml'] and 'microenvironment' in data['cells_xml']:
            data['microenv'] = data['cells_xml']['microenvironment'].get('data', None)
        
        # Update current frame
        self.current_frame = frame_number
        
        return data 