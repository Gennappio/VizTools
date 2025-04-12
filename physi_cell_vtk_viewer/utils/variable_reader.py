"""
Variable reader for PhysiCell output files.
This module extracts available variables from PhysiCell XML output files
and provides mappings to data in corresponding mat files.
"""

import os
import xml.etree.ElementTree as ET
import numpy as np
from scipy import io


def get_microenv_variables(xml_file):
    """
    Extract microenvironment variable names from a PhysiCell XML output file.
    
    Args:
        xml_file (str): Path to XML file
        
    Returns:
        list: List of dictionaries with variable information (name, units, ID)
    """
    if not os.path.exists(xml_file):
        return []
    
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        variables = []
        
        # Find all variable elements in the microenvironment section
        for var in root.findall('.//microenvironment//variable'):
            var_name = var.get('name')
            var_units = var.get('units')
            var_id = var.get('ID')
            
            if var_name:
                # Convert ID to integer correctly
                if var_id is not None:
                    var_id = int(var_id)
                else:
                    # Use sequential ID based on variable position if not specified
                    var_id = len(variables)
                
                # Debug output to help diagnose ID mapping
                print(f"Found microenvironment variable: {var_name}, ID: {var_id}")
                
                variables.append({
                    'name': var_name,
                    'units': var_units if var_units else 'dimensionless',
                    'id': var_id,
                })
        
        # Verify no duplicate IDs
        ids = [var['id'] for var in variables]
        if len(ids) != len(set(ids)):
            print("WARNING: Duplicate IDs found in microenvironment variables. Reassigning IDs.")
            # Reassign IDs to ensure uniqueness
            for i, var in enumerate(variables):
                var['id'] = i
                print(f"Reassigned {var['name']} to ID {i}")
        
        return variables
    except Exception as e:
        print(f"Error reading XML file: {e}")
        return []


def get_cell_variables(xml_file, cells_mat_file=None):
    """
    Extract cell variable names from PhysiCell output files.
    
    Args:
        xml_file (str): Path to XML file
        cells_mat_file (str, optional): Path to cells.mat file
        
    Returns:
        list: List of dictionaries with variable information
    """
    variables = []
    
    # Standard cell variables according to PhysiCell documentation
    standard_vars = [
        {'name': 'ID', 'index': 0, 'description': 'Cell ID'},
        {'name': 'position_x', 'index': 1, 'description': 'X position'},
        {'name': 'position_y', 'index': 2, 'description': 'Y position'},
        {'name': 'position_z', 'index': 3, 'description': 'Z position'},
        {'name': 'total_volume', 'index': 4, 'description': 'Total cell volume'},
        {'name': 'cell_type', 'index': 5, 'description': 'Cell type'},
        {'name': 'cycle_model', 'index': 6, 'description': 'Cycle model'},
        {'name': 'current_phase', 'index': 7, 'description': 'Current cycle phase'},
        {'name': 'elapsed_time_in_phase', 'index': 8, 'description': 'Time in current phase'},
        {'name': 'nuclear_volume', 'index': 9, 'description': 'Nuclear volume'},
        {'name': 'cytoplasmic_volume', 'index': 10, 'description': 'Cytoplasmic volume'},
        {'name': 'fluid_fraction', 'index': 11, 'description': 'Fluid fraction'},
        {'name': 'calcified_fraction', 'index': 12, 'description': 'Calcified fraction'},
        {'name': 'orientation_x', 'index': 13, 'description': 'X orientation'},
        {'name': 'orientation_y', 'index': 14, 'description': 'Y orientation'},
        {'name': 'orientation_z', 'index': 15, 'description': 'Z orientation'},
        {'name': 'polarity', 'index': 16, 'description': 'Polarity'},
    ]
    
    # Add standard variables to the list
    variables.extend(standard_vars)
    next_idx = len(standard_vars)
    
    # Create a dictionary to map variable names to indices
    var_index_map = {}
    
    # If cells mat file is provided, check for variable mapping in metadata
    if cells_mat_file and os.path.exists(cells_mat_file):
        try:
            # Load cells data
            mat_data = io.loadmat(cells_mat_file)
            
            # Check if we already have a variable mapping from the data loader
            if 'metadata' in mat_data and 'var_index_map' in mat_data['metadata']:
                var_index_map = mat_data['metadata']['var_index_map']
                print(f"Using variable mapping from metadata: {var_index_map}")
        except Exception as e:
            print(f"Error checking MAT file metadata: {e}")
    
    # Try to extract custom variables from XML first
    if xml_file and os.path.exists(xml_file):
        try:
            # Parse XML file
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Create a set to track all custom variables
            custom_vars = set()
            
            # 1. Check for 'custom' variables in the PhysiCell_settings section
            custom_data_nodes = root.findall('.//cell_definitions//cell_definition//custom_data')
            
            # Collect all custom variables from all cell definitions
            for custom_data in custom_data_nodes:
                for var in custom_data:
                    custom_vars.add(var.tag)
            
            # 2. Check for custom variables in cellular_information section
            cellular_info = root.find('.//cellular_information')
            if cellular_info is not None:
                # Check cell_population custom_data section (this is where cell_ATP_source might be)
                cell_pop_custom = cellular_info.findall('.//cell_population//custom_data')
                for custom_data in cell_pop_custom:
                    for var in custom_data:
                        custom_vars.add(var.tag)
                
                # Also check for variables list in cell_population
                variables_nodes = cellular_info.findall('.//cell_population//variables')
                for vars_node in variables_nodes:
                    for var in vars_node.findall('variable'):
                        var_name = var.get('name')
                        if var_name:
                            custom_vars.add(var_name)
                
                # Check for variables in the cell_population data records
                simplified_data = cellular_info.findall('.//cell_population//simplified_data')
                for data_node in simplified_data:
                    var_name = data_node.get('name')
                    if var_name:
                        custom_vars.add(var_name)
            
            # Also search for any XML elements containing the specific variable we're looking for
            for elem in root.findall('.//*[@name="cell_ATP_source"]'):
                custom_vars.add('cell_ATP_source')
            
            # Add custom variables to the list, with proper indexing from var_index_map if available
            for var_name in sorted(custom_vars):
                # Check if we have a mapping for this variable
                if var_name in var_index_map:
                    idx = var_index_map[var_name]
                    variables.append({
                        'name': var_name,
                        'index': idx,
                        'description': f'Custom variable: {var_name} (index {idx})'
                    })
                else:
                    # Use the next available index
                    variables.append({
                        'name': var_name,
                        'index': next_idx,
                        'description': f'Custom variable: {var_name}'
                    })
                    next_idx += 1
                
            # Debug output of found variables
            print(f"Found custom variables in XML: {custom_vars}")
                
        except Exception as e:
            print(f"Error extracting custom variables from XML: {e}")
    
    # Debug output of found variables
    var_names = [v['name'] for v in variables]
    print(f"All found variables: {var_names}")
    
    # If cells mat file is provided, check its dimensions to add any remaining variables
    if cells_mat_file and os.path.exists(cells_mat_file):
        try:
            # Load cells data
            mat_data = io.loadmat(cells_mat_file)
            if 'cells' in mat_data:
                cells = mat_data['cells']
                print(f"MAT file cells shape: {cells.shape}")
                
                # If there are more rows than variables we've already found, add them
                if cells.shape[0] > len(variables):
                    print(f"Adding {cells.shape[0] - len(variables)} additional variables from MAT file")
                    
                    for i in range(len(variables), cells.shape[0]):
                        # First check if we should use a known name for this index based on likely matches
                        # This helps match indices in the MAT file with names from XML
                        special_names = {
                            # Common PhysiCell custom data variable names and their typical indices
                            17: "relative_adhesion", 
                            18: "relative_repulsion",
                            19: "adhesion_affinities",
                            20: "uptake_rates",
                            21: "secretion_rates",
                            22: "oncoprotein",
                            # Corrected indices for cell_ATP_source and velocity variables
                            # Based on the user's specific data structure
                            # Update these indices to match your actual data structure
                            24: "cell_ATP_source",  # Changed from index 23
                            25: "cell_velocity_x",  # Changed from index 24
                            26: "cell_velocity_y",  # Changed from index 25
                            27: "cell_velocity_z",  # Changed from index 26
                            # Add more mappings as needed
                        }
                        
                        # If we have a special name for this index, or a mapping from metadata, use it
                        if i in special_names:
                            var_name = special_names[i]
                        else:
                            var_name = f"custom_var_{i}"
                            
                        # Check if we need to add this variable
                        exists = False
                        for var in variables:
                            if var['index'] == i:
                                exists = True
                                break
                                
                        if not exists:
                            variables.append({
                                'name': var_name,
                                'index': i,
                                'description': f'Custom variable from MAT file: {var_name}'
                            })
                    
                    # Debug output of added MAT variables
                    added_vars = [v['name'] for v in variables if v['index'] >= len(standard_vars)]
                    print(f"Added MAT variables: {added_vars}")
        except Exception as e:
            print(f"Error reading cells mat file: {e}")
    
    # One final check: if we have a variable mapping but couldn't find cell_ATP_source,
    # explicitly add it with the right index
    if "cell_ATP_source" not in [v['name'] for v in variables] and 'cell_ATP_source' in var_index_map:
        idx = var_index_map['cell_ATP_source']
        print(f"Explicitly adding cell_ATP_source at index {idx}")
        variables.append({
            'name': 'cell_ATP_source',
            'index': idx,
            'description': 'Cell ATP source'
        })
    
    return variables


def map_variable_to_data(variable_name, data, variable_list):
    """
    Map a variable name to its data in a numpy array.
    
    Args:
        variable_name (str): Name of the variable to extract
        data (numpy.ndarray): Data array
        variable_list (list): List of variable dictionaries
        
    Returns:
        numpy.ndarray: Array with the variable data
    """
    for var in variable_list:
        if var['name'] == variable_name:
            if 'id' in var:  # Microenvironment variable
                # Microenvironment data is structured as [x, y, z, time, var1, var2, ...]
                # So the variable data is at index var_id + 4
                var_id = var['id']
                # Include debug output to help diagnose which index is being used
                print(f"Mapping microenvironment variable '{variable_name}' to index {var_id + 4}")
                return data[var_id + 4]
            elif 'index' in var:  # Cell variable
                # Cell data is structured as a matrix with rows=cells, columns=variables
                idx = var['index']
                # Include debug output to help diagnose which index is being used
                print(f"Mapping cell variable '{variable_name}' to index {idx}")
                return data[:, idx]
    
    # If variable not found, return None
    print(f"Variable '{variable_name}' not found in variable list")
    return None 