"""
File utility functions for loading and processing PhysiCell output files
"""

import os
import numpy as np
import scipy.io as sio

def load_mat_file(file_path, debug=False):
    """Load data from a .mat file"""
    try:
        if not os.path.exists(file_path):
            return None
            
        if not file_path.lower().endswith('.mat'):
            return None
            
        mat_contents = sio.loadmat(file_path)
        mat_contents = {k:v for k, v in mat_contents.items() 
                       if not k.startswith('__')}
        
        # Print the entire content of the cells.mat file
        if debug:
            print("\n==== Contents of cells.mat file ====")
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
        
        return mat_contents
    
    except Exception as e:
        if debug:
            print(f"Error loading .mat file: {e}")
        return None

def find_output_files(directory, frame_number, cells_only=False):
    """Find PhysiCell output files for a given frame number"""
    # Define file patterns to look for
    mat_pattern = os.path.join(directory, f"output{frame_number:08d}_cells.mat")
    xml_pattern = os.path.join(directory, f"output{frame_number:08d}.xml")
    
    # Only look for microenvironment files if not in cells_only mode
    if cells_only:
        microenv_file = None
    else:
        # Try different naming patterns for microenvironment files
        microenv_pattern = os.path.join(directory, f"output{frame_number:08d}_microenvironment0.mat")
        microenv_alt_pattern = os.path.join(directory, f"output{frame_number:08d}_microenvironment.mat")
        
        # Check if either file exists
        if os.path.isfile(microenv_pattern):
            microenv_file = microenv_pattern
        elif os.path.isfile(microenv_alt_pattern):
            microenv_file = microenv_alt_pattern
        else:
            microenv_file = None
    
    # Create result dict
    files = {
        "mat_file": mat_pattern if os.path.isfile(mat_pattern) else None,
        "xml_file": xml_pattern if os.path.isfile(xml_pattern) else None,
        "microenv_file": microenv_file
    }
    
    return files

def find_max_frame(directory):
    """Find the maximum frame number in the given directory"""
    max_frame = -1
    
    if not directory or not os.path.isdir(directory):
        return max_frame
    
    # Find XML files matching the pattern: outputNNNNNNNN.xml
    for filename in os.listdir(directory):
        if filename.startswith("output") and filename.endswith(".xml"):
            try:
                # Extract frame number
                frame_str = filename.replace("output", "").replace(".xml", "")
                frame_num = int(frame_str)
                max_frame = max(max_frame, frame_num)
            except ValueError:
                # Not a valid frame number, skip
                continue
    
    return max_frame
