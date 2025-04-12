"""
Visualization controller module for PhysiCell visualization
"""

import os
import vtk


class VisualizationController:
    """
    Manages all visualization components and controls variable selection
    """
    
    def __init__(self, renderer, cell_visualizer=None, microenv_visualizer=None):
        """Initialize with a VTK renderer and optional visualizers"""
        self.renderer = renderer
        self.cell_visualizer = cell_visualizer
        self.microenv_visualizer = microenv_visualizer
        
        # Variable tracking
        self.cell_variables = []
        self.microenv_variables = []
        self.current_cell_variable = None
        self.current_microenv_variable = None
        
        # Source data
        self.cell_data = None
        self.microenv_data = None
        self.frame_data = None
        
        # Keep track of available variable names and their indices
        self.cell_variable_dict = {}
        self.microenv_variable_dict = {}
        
    def clear_all(self):
        """Clear all visualizations"""
        if self.cell_visualizer:
            self.cell_visualizer.clear()
        
        if self.microenv_visualizer:
            self.microenv_visualizer.clear()
        
        # Clear variable records
        self.cell_variables = []
        self.microenv_variables = []
        self.cell_variable_dict = {}
        self.microenv_variable_dict = {}
        self.current_cell_variable = None
        self.current_microenv_variable = None
    
    def set_cell_data(self, cell_data):
        """Set cell data and update available variables"""
        self.cell_data = cell_data
        
        # Get available variables from cell data
        if cell_data is not None:
            if isinstance(cell_data, dict):
                # Handle dictionary-style data (XML format)
                if 'data' in cell_data and 'metadata' in cell_data:
                    # Extract variable names from metadata
                    if 'variables' in cell_data['metadata']:
                        self.cell_variables = cell_data['metadata']['variables']
                        
                        # Update the variable dictionary for lookup
                        self.cell_variable_dict = {
                            var: idx for idx, var in enumerate(self.cell_variables)
                        }
                        
                        # Set default variable (position)
                        if 'position_x' in self.cell_variable_dict:
                            self.current_cell_variable = 'position_x'
                        elif len(self.cell_variables) > 0:
                            self.current_cell_variable = self.cell_variables[0]
            
            elif hasattr(cell_data, 'shape') and len(cell_data.shape) > 0:
                # Handle array-style data (MAT format)
                # PhysiCell cells.mat typically has 87 attributes per cell
                if cell_data.shape[0] == 87:
                    # Define standard PhysiCell cell variables
                    std_vars = [
                        'ID', 'position_x', 'position_y', 'position_z', 
                        'total_volume', 'cell_type', 'cycle_model', 'current_phase',
                        'elapsed_time_in_phase', 'nuclear_volume', 'cytoplasmic_volume',
                        'fluid_fraction', 'calcified_fraction', 'orientation_x', 'orientation_y', 'orientation_z',
                        'polarity', 'migration_speed', 'motility_vector_x', 'motility_vector_y', 'motility_vector_z',
                        'migration_bias', 'motility_bias_direction_x', 'motility_bias_direction_y', 'motility_bias_direction_z',
                        'persistence_time', 'motility_reserved'
                    ]
                    
                    # Add standard variables and remaining indices
                    self.cell_variables = std_vars.copy()
                    for i in range(len(std_vars), 87):
                        self.cell_variables.append(f"attribute_{i}")
                    
                    # Update the variable dictionary for lookup
                    self.cell_variable_dict = {
                        var: idx for idx, var in enumerate(self.cell_variables)
                    }
                    
                    # Set default variable
                    self.current_cell_variable = 'cell_type'
    
    def set_microenv_data(self, microenv_data):
        """Set microenvironment data and update available variables"""
        self.microenv_data = microenv_data
        
        # Get available variables from microenvironment data
        if microenv_data is not None:
            if isinstance(microenv_data, dict):
                # Handle dictionary-style data
                if 'data' in microenv_data and 'metadata' in microenv_data:
                    # Extract variable names from metadata
                    if 'variables' in microenv_data['metadata']:
                        self.microenv_variables = microenv_data['metadata']['variables']
                        
                        # Update the variable dictionary for lookup
                        self.microenv_variable_dict = {
                            var: idx for idx, var in enumerate(self.microenv_variables)
                        }
                        
                        # Set default variable (first chemical species)
                        if len(self.microenv_variables) > 0:
                            self.current_microenv_variable = self.microenv_variables[0]
                
                # Simpler dictionary format
                elif 'data' in microenv_data and 'variables' in microenv_data:
                    self.microenv_variables = microenv_data['variables']
                    
                    # Update the variable dictionary for lookup
                    self.microenv_variable_dict = {
                        var: idx for idx, var in enumerate(self.microenv_variables)
                    }
                    
                    # Set default variable (first chemical species)
                    if len(self.microenv_variables) > 0:
                        self.current_microenv_variable = self.microenv_variables[0]
    
    def visualize_with_current_settings(self):
        """Visualize data with current variable selections"""
        # Visualize cells if available
        if self.cell_visualizer and self.cell_data is not None:
            if self.current_cell_variable:
                var_idx = self.cell_variable_dict.get(self.current_cell_variable, 0)
                self.cell_visualizer.visualize_cells_with_variable(
                    self.cell_data, 
                    variable_name=self.current_cell_variable,
                    variable_idx=var_idx
                )
            else:
                # Default visualization
                self.cell_visualizer.visualize_cells_with_variable(
                    self.cell_data,
                    variable_name=None,
                    variable_idx=None
                )
        
        # Visualize microenvironment if available
        if self.microenv_visualizer and self.microenv_data is not None:
            if self.current_microenv_variable:
                var_idx = self.microenv_variable_dict.get(self.current_microenv_variable, 0)
                self.microenv_visualizer.visualize_microenvironment(
                    self.microenv_data,
                    variable_index=var_idx
                )
    
    def set_current_cell_variable(self, variable_name):
        """Set the current cell variable to visualize"""
        if variable_name in self.cell_variable_dict:
            self.current_cell_variable = variable_name
            
            # Re-visualize with the new variable
            if self.cell_visualizer and self.cell_data is not None:
                var_idx = self.cell_variable_dict[variable_name]
                self.cell_visualizer.visualize_cells_with_variable(
                    self.cell_data, 
                    variable_name=variable_name,
                    variable_idx=var_idx
                )
            
            return True
        return False
    
    def set_current_microenv_variable(self, variable_name):
        """Set the current microenvironment variable to visualize"""
        if variable_name in self.microenv_variable_dict:
            self.current_microenv_variable = variable_name
            
            # Re-visualize with the new variable
            if self.microenv_visualizer and self.microenv_data is not None:
                var_idx = self.microenv_variable_dict[variable_name]
                self.microenv_visualizer.visualize_microenvironment(
                    self.microenv_data,
                    variable_index=var_idx
                )
            
            return True
        return False
    
    def get_available_cell_variables(self):
        """Get list of available cell variables"""
        return self.cell_variables
    
    def get_available_microenv_variables(self):
        """Get list of available microenvironment variables"""
        return self.microenv_variables
    
    def set_wireframe_mode(self, enabled):
        """Enable/disable wireframe visualization mode"""
        if self.microenv_visualizer:
            self.microenv_visualizer.set_wireframe_mode(enabled)
            
            # Update visualization if microenv data exists
            if self.microenv_data is not None and self.current_microenv_variable:
                var_idx = self.microenv_variable_dict.get(self.current_microenv_variable, 0)
                self.microenv_visualizer.visualize_microenvironment(
                    self.microenv_data,
                    variable_index=var_idx
                )
    
    def set_cell_visibility(self, visible):
        """Set cell visualization visibility"""
        if self.cell_visualizer:
            self.cell_visualizer.set_visibility(visible)
    
    def set_microenv_visibility(self, visible):
        """Set microenvironment visualization visibility"""
        if self.microenv_visualizer:
            self.microenv_visualizer.set_visibility(visible)
    
    def set_cell_opacity(self, opacity_percent):
        """Set opacity for cell visualization (0-100)"""
        if self.cell_visualizer:
            self.cell_visualizer.set_opacity(opacity_percent)
    
    def set_microenv_opacity(self, opacity_percent):
        """Set opacity for microenvironment visualization (0-100)"""
        if self.microenv_visualizer:
            self.microenv_visualizer.set_opacity(opacity_percent)
            
    def load_frame_data(self, frame_dir):
        """Load all data from a frame directory"""
        if not os.path.exists(frame_dir):
            print(f"Frame directory does not exist: {frame_dir}")
            return False
        
        # Store frame data path
        self.frame_dir = frame_dir
        
        # Clear previous visualizations
        self.clear_all()
        
        # Check for XML files (cells)
        xml_file = os.path.join(frame_dir, "cells.xml")
        if os.path.exists(xml_file):
            # Logic to load and parse XML file (implement in subclass)
            pass
        
        # Check for MAT files (cells)
        mat_file = os.path.join(frame_dir, "cells.mat")
        if os.path.exists(mat_file):
            # Logic to load and parse MAT file (implement in subclass)
            pass
        
        # Check for microenvironment data
        microenv_file = os.path.join(frame_dir, "microenvironment.mat")
        if os.path.exists(microenv_file):
            # Logic to load and parse microenvironment file (implement in subclass)
            pass
        
        # Implement additional data loading in subclasses
        
        return True 