"""
Data models for PhysiCell visualization
"""

import os
import numpy as np
import scipy.io as sio
from physi_cell_vtk_viewer.utils.file_utils import find_output_files, find_max_frame

class PhysiCellData:
    """Model for PhysiCell output data"""
    
    def __init__(self, debug=False):
        """Initialize the data model"""
        self.output_dir = ""
        self.current_frame = 0
        self.max_frame = -1
        self.debug = debug
        
        # Optional pyMCDS module for XML parsing
        try:
            from pyMCDS_cells import pyMCDS_cells
            self.pyMCDS_cells = pyMCDS_cells
        except ImportError:
            self.pyMCDS_cells = None
            if debug:
                print("Warning: pyMCDS_cells module not found. XML visualization will be limited.")
    
    def set_output_directory(self, directory):
        """Set the output directory and find the maximum frame number"""
        self.output_dir = directory
        self.max_frame = find_max_frame(directory)
        return self.max_frame
    
    def get_output_directory(self):
        """Get the current output directory"""
        return self.output_dir
    
    def get_current_frame(self):
        """Get the current frame number"""
        return self.current_frame
    
    def get_max_frame(self):
        """Get the maximum frame number"""
        return self.max_frame
    
    def load_frame(self, frame_number):
        """Load data for a specific frame number"""
        if not self.output_dir or frame_number < 0 or frame_number > self.max_frame:
            return None
        
        self.current_frame = frame_number
        
        # Find all the files for this frame
        files = find_output_files(self.output_dir, frame_number)
        
        # Load the data from each file type
        data = {
            'cells_mat': self.load_cells_mat(files['mat_file']),
            'cells_xml': self.load_cells_xml(files['xml_file']),
            'microenv': self.load_microenv(files['microenv_file'])
        }
        
        return data
    
    def load_cells_mat(self, file_path):
        """Load cell data from a .mat file"""
        if not file_path or not os.path.exists(file_path):
            return None
        
        try:
            mat_contents = sio.loadmat(file_path)
            mat_contents = {k:v for k, v in mat_contents.items() 
                           if not k.startswith('__')}
            
            # Print debug info if enabled
            if self.debug:
                self._print_mat_debug_info(file_path, mat_contents)
            
            return mat_contents
        
        except Exception as e:
            if self.debug:
                print(f"Error loading {file_path}: {e}")
            return None
    
    def load_cells_xml(self, file_path):
        """Load cell data from an XML file using pyMCDS_cells if available"""
        if not file_path or not os.path.exists(file_path) or not self.pyMCDS_cells:
            return None
        
        try:
            mcds = self.pyMCDS_cells(os.path.basename(file_path), os.path.dirname(file_path))
            
            # Print debug info if enabled
            if self.debug:
                self._print_xml_debug_info(file_path, mcds)
            
            return mcds
        
        except Exception as e:
            if self.debug:
                print(f"Error loading {file_path}: {e}")
            return None
    
    def load_microenv(self, file_path):
        """Load microenvironment data from a .mat file"""
        if not file_path or not os.path.exists(file_path):
            return None
        
        try:
            mat_contents = sio.loadmat(file_path)
            mat_contents = {k:v for k, v in mat_contents.items() 
                           if not k.startswith('__')}
            
            # Get the microenvironment data
            if 'multiscale_microenvironment' in mat_contents:
                microenv_data = mat_contents['multiscale_microenvironment']
                
                # Print debug info if enabled
                if self.debug:
                    self._print_microenv_debug_info(file_path, microenv_data)
                
                return microenv_data
            
            return None
        
        except Exception as e:
            if self.debug:
                print(f"Error loading {file_path}: {e}")
            return None
    
    def _print_mat_debug_info(self, file_path, mat_contents):
        """Print debug information about a .mat file"""
        print(f"\n==== Contents of {file_path} ====")
        for key, value in mat_contents.items():
            print(f"Key: {key}")
            print(f"Type: {type(value)}")
            print(f"Shape: {value.shape if hasattr(value, 'shape') else 'N/A'}")
            
            # Always display full content of 'cells' array regardless of size
            if key == 'cells' and isinstance(value, np.ndarray):
                # Format array elements as fixed-point notation
                if np.issubdtype(value.dtype, np.number):
                    # Set a large threshold to ensure all elements are displayed
                    np.set_printoptions(threshold=np.inf, precision=6, suppress=True)
                    print("Content of cells array:")
                    print(value)
                    # Reset print options to default
                    np.set_printoptions(threshold=1000)
                else:
                    print(f"Content: {value}")
            elif isinstance(value, np.ndarray) and value.size < 20:  # Only print small arrays fully
                # Format array elements as fixed-point notation
                if np.issubdtype(value.dtype, np.number):
                    value_str = np.array2string(value, precision=6, suppress_small=True, formatter={'float_kind': lambda x: f"{x:.6f}"})
                    print(f"Content: {value_str}")
                else:
                    print(f"Content: {value}")
            else:
                print(f"Content: {type(value)} (too large to display)")
            print("-" * 50)
    
    def _print_xml_debug_info(self, file_path, mcds):
        """Print debug information about an XML file"""
        print(f"\n==== Contents of {file_path} ====")
        try:
            cell_df = mcds.get_cell_df()
            print(f"Number of cells: {len(cell_df)}")
            if not cell_df.empty:
                print(f"Cell columns: {list(cell_df.columns)}")
                print(f"First cell data:")
                print(cell_df.iloc[0])
                
                # Print position bounds
                positions_x = cell_df['position_x'].values
                positions_y = cell_df['position_y'].values
                print(f"X range: {positions_x.min():.6f} to {positions_x.max():.6f}")
                print(f"Y range: {positions_y.min():.6f} to {positions_y.max():.6f}")
                if 'position_z' in cell_df.columns:
                    positions_z = cell_df['position_z'].values
                    print(f"Z range: {positions_z.min():.6f} to {positions_z.max():.6f}")
        except Exception as e:
            print(f"Error analyzing XML data: {e}")
        print("-" * 50)
    
    def _print_microenv_debug_info(self, file_path, microenv_data):
        """Print debug information about a microenvironment file"""
        print(f"\n==== Contents of {file_path} ====")
        print(f"Shape: {microenv_data.shape}")
        
        # Extract coordinate info
        x = microenv_data[0, :].flatten()  # x coordinates
        y = microenv_data[1, :].flatten()  # y coordinates
        z = microenv_data[2, :].flatten()  # z coordinates
        
        print(f"X range: {x.min():.6f} to {x.max():.6f}, shape: {x.shape}")
        print(f"Y range: {y.min():.6f} to {y.max():.6f}, shape: {y.shape}")
        print(f"Z range: {z.min():.6f} to {z.max():.6f}, shape: {z.shape}")
        
        # Number of substrates (chemical species)
        substrate_count = microenv_data.shape[0] - 4
        print(f"Number of substrates: {substrate_count}")
        
        # Print substrate ranges
        for substrate_idx in range(substrate_count):
            substrate_data = microenv_data[4 + substrate_idx, :]
            print(f"Substrate {substrate_idx} range: {substrate_data.min():.6f} to {substrate_data.max():.6f}")
        
        print("-" * 50) 