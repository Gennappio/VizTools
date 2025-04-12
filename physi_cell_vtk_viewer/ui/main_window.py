"""
Main window for PhysiCell VTK Viewer
"""

from PyQt5.QtWidgets import QMainWindow, QSplitter, QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt, QTimer
import numpy as np

from physi_cell_vtk_viewer.ui.control_panel import ControlPanel
from physi_cell_vtk_viewer.ui.vtk_panel import VTKPanel
from physi_cell_vtk_viewer.models.data_model import PhysiCellData
from physi_cell_vtk_viewer.visualization.cells_visualizer import CellsVisualizer
from physi_cell_vtk_viewer.visualization.microenvironment_visualizer import MicroenvironmentVisualizer
from physi_cell_vtk_viewer.visualization.slicing import SliceVisualizer

class MainWindow(QMainWindow):
    """Main window for PhysiCell VTK Viewer application"""
    
    def __init__(self, initial_dir=None, debug=False, cells_only=False):
        """Initialize the application window and UI components"""
        super().__init__()
        
        # Store debug setting
        self.debug = debug
        
        # Store cells_only setting
        self.cells_only = cells_only
        
        # Application state
        self.data_loaded = False
        self.initial_dir = initial_dir or ""
        
        # Initialize collected options
        self.collected_display_options = {}
        self.collected_slice_options = {}
        self.collected_frame = 0  # Store frame changes here
        self.collected_variables = None  # Store variable selection here
        
        # Set up data model
        self.data_model = PhysiCellData(debug=debug, cells_only=cells_only)
        
        # Set up the main window
        self.setWindowTitle("PhysiCell VTK Viewer" + (" (Cells Only)" if cells_only else ""))
        self.resize(1200, 800)
        
        # Create the central widget and main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create a splitter for resizable panels
        self.splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(self.splitter, 1)
        
        # Create control panel (left side)
        self.control_panel = ControlPanel()
        self.splitter.addWidget(self.control_panel)
        
        # Create VTK panel (right side)
        self.vtk_panel = VTKPanel()
        self.splitter.addWidget(self.vtk_panel)
        
        # Set initial splitter sizes (25% controls, 75% VTK view)
        self.splitter.setSizes([300, 900])
        
        # Create status bar at the bottom
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready. Please load a PhysiCell output directory.")
        
        # Create visualization components
        self.cells_visualizer = CellsVisualizer(self.vtk_panel.get_renderer())
        self.microenv_visualizer = MicroenvironmentVisualizer(self.vtk_panel.get_renderer())
        self.slice_visualizer = SliceVisualizer(self.vtk_panel.get_renderer())
        
        # Set up an animation timer
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.animation_step)
        self.animation_active = False
        
        # Connect signals
        self.connect_signals()
        
        # If initial directory provided, load it
        if initial_dir:
            self.load_directory(initial_dir)
    
    def connect_signals(self):
        """Connect signals from UI components"""
        # Connect control panel signals
        self.control_panel.directory_loaded.connect(self.load_directory)
        
        # Collect frame changes instead of loading them immediately
        self.control_panel.frame_changed.connect(self.collect_frame_change)
        
        # Connect the apply button signal instead of automatic updates
        self.control_panel.apply_changes.connect(self.apply_changes)
        
        # Connect variable selection signal
        self.control_panel.variables_selected.connect(self.collect_variable_selection)
        
        # Still connect these signals for collecting options but don't trigger updates
        self.control_panel.display_options_changed.connect(self.collect_display_options)
        self.control_panel.slice_options_changed.connect(self.collect_slice_options)
        
        self.control_panel.play_toggled.connect(self.toggle_animation)
        
        # If in cells_only mode, hide microenvironment controls
        if self.cells_only:
            # Hide microenvironment checkbox and controls
            self.control_panel.show_microenv_cb.setChecked(False)
            self.control_panel.show_microenv_cb.setEnabled(False)
            
            # Hide microenvironment opacity slider if it exists
            if hasattr(self.control_panel, 'microenv_opacity_slider'):
                self.control_panel.microenv_opacity_slider.setEnabled(False)
                # Look for a label near this slider
                for widget in self.control_panel.findChildren(QLabel):
                    if widget.text() == "Microenvironment Opacity:":
                        widget.setEnabled(False)
                        break
            
            # Hide slice controls
            self.control_panel.show_slice_cb.setChecked(False)
            self.control_panel.show_slice_cb.setEnabled(False)
            
            # Hide wireframe mode checkbox if it exists
            if hasattr(self.control_panel, 'microenv_wireframe_cb'):
                self.control_panel.microenv_wireframe_cb.setEnabled(False)
            
            # Hide any slice-related controls
            self.control_panel.hide_slice_controls()
    
    def load_directory(self, directory):
        """Load a PhysiCell output directory"""
        self.status_bar.showMessage(f"Loading directory: {directory}")
        
        # Set the directory in the data model
        max_frame = self.data_model.set_output_directory(directory)
        
        if max_frame >= 0:
            # Update status and controls
            self.control_panel.set_max_frame(max_frame)
            self.status_bar.showMessage(f"Loaded directory: {directory}, {max_frame+1} frames found")
            
            # Initialize collected frame to 0
            self.collected_frame = 0
            
            # Update variable dropdowns with first frame
            self.update_variable_dropdowns(0)
            
            # Load the first frame
            self.load_frame(0)
            self.data_loaded = True
        else:
            self.status_bar.showMessage(f"No PhysiCell output files found in {directory}")
            self.control_panel.update_info("No PhysiCell output files found in the selected directory.")
    
    def update_variable_dropdowns(self, frame_number):
        """Update the variable dropdowns with data from the specified frame"""
        directory = self.data_model.output_dir
        if not directory:
            return
        
        # Construct file paths for the frame
        frame_str = f"{frame_number:08d}"
        xml_file = f"{directory}/output{frame_str}.xml"
        cells_file = f"{directory}/output{frame_str}_cells.mat"
        
        # Update the dropdowns in the control panel
        self.control_panel.update_variable_dropdowns(xml_file, cells_file)
    
    def load_frame(self, frame_number):
        """Load and visualize data for a specific frame"""
        if not self.data_loaded and self.data_model.get_max_frame() < 0:
            return
        
        # Clear previous visualization
        self.clear_visualization()
        
        # Store current selections to restore them after loading
        saved_selections = {}
        try:
            # Save current selections if they exist
            if hasattr(self.control_panel, 'microenv_var_combo') and self.control_panel.microenv_var_combo.currentIndex() >= 0:
                saved_selections['microenv'] = self.control_panel.microenv_var_combo.currentIndex()
            if hasattr(self.control_panel, 'cell_var_combo') and self.control_panel.cell_var_combo.currentIndex() >= 0:
                saved_selections['cell'] = self.control_panel.cell_var_combo.currentIndex()
            if hasattr(self.control_panel, 'cell_color_combo') and self.control_panel.cell_color_combo.currentIndex() >= 0:
                saved_selections['cell_color'] = self.control_panel.cell_color_combo.currentIndex()
        except:
            # In case of any error, just continue without saved selections
            pass
        
        # Load the data
        data = self.data_model.load_frame(frame_number)
        
        if data:
            # Get current display options
            options = self.get_display_options()
            
            # Store data globally for all components
            self.current_frame_data = data
            
            # Process cell data regardless of visibility (store for later toggling)
            has_valid_cells = (
                'cells_mat' in data and 
                data['cells_mat'] is not None and
                'cells' in data['cells_mat'] and
                isinstance(data['cells_mat']['cells'], np.ndarray) and
                data['cells_mat']['cells'].size > 0 and
                data['cells_mat']['cells'].shape[1] > 0
            )
            
            if has_valid_cells:
                cells_data = data['cells_mat']['cells']
                
                # Process the data but set visibility based on options
                if cells_data.shape[0] >= 80 and cells_data.shape[1] == 1:
                    # Single cell
                    self.cells_visualizer.visualize_single_cell(cells_data, options['cell_opacity']/100.0)
                elif cells_data.shape[0] >= 80:
                    # Multiple cells
                    self.cells_visualizer.visualize_multiple_cells(cells_data, options['cell_opacity']/100.0)
                elif cells_data.shape[1] == 1:
                    # Scalar values
                    self.cells_visualizer.create_spiral_viz_from_values(cells_data.flatten(), options['cell_opacity']/100.0)
                
                # Set visibility based on option
                self.cells_visualizer.set_visibility(options['show_cells'])
            else:
                # Clear any existing cell visualization
                self.cells_visualizer.clear()
                if self.debug:
                    print("No valid cells data to visualize in this frame")
            
            # Process XML cells data regardless of visibility
            if data['cells_xml']:
                self.cells_visualizer.visualize_cells_from_xml(data['cells_xml'], options['cell_opacity']/100.0)
                # Set visibility based on option
                self.cells_visualizer.set_visibility(options['show_cells'])
            
            # Process microenvironment data regardless of visibility
            # Skip microenvironment visualization if cells_only is True
            if not self.cells_only and 'microenv' in data and data['microenv'] is not None and data['microenv'].size > 0:
                self.microenv_visualizer.visualize_microenvironment(
                    data['microenv'],
                    wireframe_mode=options['wireframe_mode'],
                    opacity=options['microenv_opacity']/100.0,
                    auto_range=options['auto_range'],
                    min_val=options['min_value'],
                    max_val=options['max_value'],
                    debug=self.debug
                )
                
                # Set visibility based on option
                self.microenv_visualizer.set_visibility(options['show_microenv'])
                
                # Update the control panel with the actual min/max values
                if options['auto_range']:
                    min_val, max_val = self.microenv_visualizer.get_data_range()
                    self.control_panel.update_min_max_ranges(min_val, max_val)
                
                # Update slice - now that all data is loaded
                self.update_slice_options(self.get_slice_options())
            elif self.cells_only:
                # In cells_only mode, hide microenvironment controls
                self.control_panel.show_microenv_cb.setChecked(False)
                self.control_panel.show_microenv_cb.setEnabled(False)
                self.control_panel.show_slice_cb.setChecked(False)
                self.control_panel.show_slice_cb.setEnabled(False)
                
                # Hide microenvironment variable selection if it exists
                if hasattr(self.control_panel, 'microenv_var_combo'):
                    # Find the label associated with the microenv_var_combo
                    for widget in self.control_panel.findChildren(QLabel):
                        if widget.text() == "Microenvironment Variable:":
                            widget.setVisible(False)
                            break
                    self.control_panel.microenv_var_combo.setVisible(False)
            
            # Reset camera to show all actors
            self.vtk_panel.reset_camera()
            
            # Update the frame info
            self.control_panel.set_frame(frame_number)
            self.control_panel.update_info(f"Frame: {frame_number}")
            
            # Update variable dropdowns when switching to a new frame
            if not hasattr(self, 'previous_frame_number') or self.previous_frame_number != frame_number:
                self.update_variable_dropdowns(frame_number)
                self.previous_frame_number = frame_number
                
                # Restore user selections if they exist
                try:
                    if 'microenv' in saved_selections and saved_selections['microenv'] < self.control_panel.microenv_var_combo.count():
                        self.control_panel.microenv_var_combo.setCurrentIndex(saved_selections['microenv'])
                    if 'cell' in saved_selections and saved_selections['cell'] < self.control_panel.cell_var_combo.count():
                        self.control_panel.cell_var_combo.setCurrentIndex(saved_selections['cell'])
                    if 'cell_color' in saved_selections and saved_selections['cell_color'] < self.control_panel.cell_color_combo.count():
                        self.control_panel.cell_color_combo.setCurrentIndex(saved_selections['cell_color'])
                except:
                    # In case of any error, just continue
                    pass
    
    def update_display_options(self, options):
        """Update the visualization based on display options"""
        if not self.data_loaded:
            return
        
        # Only update visibility and opacity, not the data itself
        
        # Update cell visibility
        self.cells_visualizer.set_visibility(options['show_cells'])
        
        # Update cell opacity
        self.cells_visualizer.set_opacity(options['cell_opacity'])
        
        # Update microenvironment visibility
        self.microenv_visualizer.set_visibility(options['show_microenv'])
        
        # Update microenvironment opacity
        self.microenv_visualizer.set_opacity(options['microenv_opacity'])
        
        # Update slice options
        slice_options = self.get_slice_options()
        if slice_options['show_slice']:
            # Only update slice visibility, not the data itself
            self.slice_visualizer.set_visibility(True)
        else:
            self.slice_visualizer.set_visibility(False)
        
        # Only reload the frame if we need to change the actual visualization type
        if 'wireframe_mode' in options or 'auto_range' in options:
            # These change the actual visualization structure, so we need to reload
            self.load_frame(self.data_model.get_current_frame())
            
    def update_slice_options(self, options):
        """Update the slice visualization based on slice options"""
        # Get the microenvironment data
        microenv_data = self.microenv_visualizer.get_microenv_data()
        microenv_vol_prop = self.microenv_visualizer.get_microenv_vol_prop()
        min_val, max_val = self.microenv_visualizer.get_data_range()
        
        # Get current display options
        display_options = self.get_display_options()
        
        if options['show_slice']:
            # Always update the actual slice when the slice options change
            self.slice_visualizer.update_slice(
                microenv_data, 
                microenv_vol_prop,
                options['position'],
                options['normal'],
                display_options['auto_range'],
                min_val,
                max_val
            )
            
            # Make slice visible
            self.slice_visualizer.set_visibility(True)
        else:
            # Hide but don't destroy the slice
            self.slice_visualizer.set_visibility(False)
    
    def toggle_animation(self, play):
        """Toggle animation playback"""
        if play:
            # Start animation if not already running
            if not self.animation_active:
                self.animation_timer.start(100)  # 10 frames per second
                self.animation_active = True
        else:
            # Stop animation if running
            if self.animation_active:
                self.animation_timer.stop()
                self.animation_active = False
    
    def animation_step(self):
        """Handle animation timer step - advance to next frame"""
        current_frame = self.data_model.get_current_frame()
        max_frame = self.data_model.get_max_frame()
        
        if current_frame < max_frame:
            # Go to next frame - directly load without using apply mechanism
            # We want animation to work immediately without waiting for Apply
            self.load_frame(current_frame + 1)
            
            # Update collected frame to keep it in sync
            self.collected_frame = current_frame + 1
            
            # Update the control panel to show the current frame
            self.control_panel.set_frame(current_frame + 1)
        else:
            # Reached the end, stop animation
            self.animation_timer.stop()
            self.animation_active = False
            self.control_panel.play_btn.setIcon(self.control_panel.style().standardIcon(Qt.SP_MediaPlay))
    
    def clear_visualization(self):
        """Clear all visualization actors"""
        self.cells_visualizer.clear()
        self.microenv_visualizer.clear()
        # Note: We don't clear the slice here as it's handled in update_slice_options
    
    def get_display_options(self):
        """Get current display options from the control panel"""
        options = {
            'show_cells': self.control_panel.show_mat_cb.isChecked(),
            'show_microenv': self.control_panel.show_microenv_cb.isChecked(),
            'wireframe_mode': self.control_panel.microenv_wireframe_cb.isChecked(),
            'auto_range': self.control_panel.auto_range_cb.isChecked(),
            'min_value': self.control_panel.min_range_input.value(),
            'max_value': self.control_panel.max_range_input.value(),
            'cell_opacity': self.control_panel.cell_opacity_slider.value(),
            'microenv_opacity': self.control_panel.microenv_opacity_slider.value(),
            'cell_as_spheres': self.control_panel.cell_as_spheres_cb.isChecked()
        }
        return options
    
    def get_slice_options(self):
        """Get current slice options from the control panel"""
        options = {
            'show_slice': self.control_panel.show_slice_cb.isChecked(),
            'position': (
                self.control_panel.slice_x.value(),
                self.control_panel.slice_y.value(),
                self.control_panel.slice_z.value()
            ),
            'normal': (
                self.control_panel.slice_i.value(),
                self.control_panel.slice_j.value(),
                self.control_panel.slice_k.value()
            )
        }
        return options
    
    def collect_frame_change(self, frame_number):
        """Collect frame changes without immediately loading the frame"""
        self.collected_frame = frame_number
        
        # Update the frame info without loading the frame
        self.control_panel.frame_info_label.setText(f"Frame: {frame_number} (not applied)")
    
    def collect_variable_selection(self, selection):
        """Collect variable selection without applying changes"""
        self.collected_variables = selection
        if self.debug:
            print(f"Variable selection: {selection}")
        # Update will happen when apply button is clicked
    
    def apply_changes(self):
        """Handle Apply button click - update all visualizations"""
        if not self.data_loaded:
            return
        
        # Store current selections to restore them after changes
        saved_selections = {}
        try:
            # Save current selections if they exist
            if hasattr(self.control_panel, 'microenv_var_combo') and self.control_panel.microenv_var_combo.currentIndex() >= 0:
                saved_selections['microenv'] = self.control_panel.microenv_var_combo.currentIndex()
            if hasattr(self.control_panel, 'cell_var_combo') and self.control_panel.cell_var_combo.currentIndex() >= 0:
                saved_selections['cell'] = self.control_panel.cell_var_combo.currentIndex()
            if hasattr(self.control_panel, 'cell_color_combo') and self.control_panel.cell_color_combo.currentIndex() >= 0:
                saved_selections['cell_color'] = self.control_panel.cell_color_combo.currentIndex()
        except:
            # In case of any error, just continue without saved selections
            pass
            
        # Check if we need to load a new frame
        current_frame = self.data_model.get_current_frame()
        if self.collected_frame != current_frame:
            # Load the new frame
            self.load_frame(self.collected_frame)
            return  # Loading a frame will handle all other updates
        
        # Get current options
        display_options = self.collected_display_options
        slice_options = self.collected_slice_options
        
        # Apply display options
        self.update_display_options(display_options)
        
        # Apply slice options
        self.update_slice_options(slice_options)
        
        # Apply variable selection if available
        if self.collected_variables:
            self.apply_variable_selection(self.collected_variables)
        
        # Restore user selections
        try:
            if 'microenv' in saved_selections and saved_selections['microenv'] < self.control_panel.microenv_var_combo.count():
                self.control_panel.microenv_var_combo.setCurrentIndex(saved_selections['microenv'])
            if 'cell' in saved_selections and saved_selections['cell'] < self.control_panel.cell_var_combo.count():
                self.control_panel.cell_var_combo.setCurrentIndex(saved_selections['cell'])
            if 'cell_color' in saved_selections and saved_selections['cell_color'] < self.control_panel.cell_color_combo.count():
                self.control_panel.cell_color_combo.setCurrentIndex(saved_selections['cell_color'])
        except:
            # In case of any error, just continue
            pass
        
        # Update the view
        self.vtk_panel.get_renderer().GetRenderWindow().Render()
        
        # Show confirmation in status bar
        self.status_bar.showMessage("Changes applied", 3000)
    
    def collect_display_options(self, options):
        """Collect display options without updating visualization"""
        self.collected_display_options = options
    
    def collect_slice_options(self, options):
        """Collect slice options without updating visualization"""
        self.collected_slice_options = options
    
    def apply_variable_selection(self, selection):
        """Apply variable selection to the visualization"""
        if not self.data_loaded:
            return
        
        # Check for microenvironment variable selection
        if not self.cells_only and 'microenv_var' in selection and selection['microenv_var']:
            microenv_var = selection['microenv_var']
            
            if self.debug:
                print(f"Applying microenvironment variable: {microenv_var['name']}")
            
            # Update the microenvironment visualization to show the selected variable
            if 'microenv' in self.current_frame_data:
                # Get the microenvironment data
                microenv_data = self.current_frame_data['microenv']
                
                # Extract variable data using the variable ID
                var_id = microenv_var['id']
                
                # Print debug info to help diagnose indexing
                if self.debug:
                    print(f"Using microenvironment variable ID: {var_id} for {microenv_var['name']}")
                    print(f"Data shape: {microenv_data.shape if hasattr(microenv_data, 'shape') else 'unknown'}")
                
                # Update wireframe mode based on variable selection
                wireframe_mode = selection.get('wireframe', False)
                
                # Get slice auto range and min/max values
                slice_auto_range = self.collected_display_options.get('slice_auto_range', True)
                slice_min_val = self.collected_display_options.get('slice_min_value', 0.0)
                slice_max_val = self.collected_display_options.get('slice_max_value', 1.0)
                
                # Update the microenvironment visualizer
                self.microenv_visualizer.visualize_microenvironment(
                    microenv_data,
                    wireframe_mode=wireframe_mode,
                    opacity=self.collected_display_options.get('microenv_opacity', 50)/100.0,
                    auto_range=self.collected_display_options.get('auto_range', True),
                    min_val=self.collected_display_options.get('min_value', 0.0),
                    max_val=self.collected_display_options.get('max_value', 1.0),
                    slice_auto_range=slice_auto_range,
                    slice_min_val=slice_min_val,
                    slice_max_val=slice_max_val,
                    debug=self.debug,
                    substrate_id=var_id  # Pass the specific substrate ID to visualize
                )
                
                # Update the control panel with the actual min/max values
                if self.collected_display_options.get('auto_range', True):
                    min_val, max_val = self.microenv_visualizer.get_data_range()
                    
                    # Get cell and slice data ranges if available
                    cell_min_val, cell_max_val = None, None
                    if hasattr(self.cells_visualizer, 'get_data_range'):
                        cell_min_val, cell_max_val = self.cells_visualizer.get_data_range()
                    
                    slice_min_val, slice_max_val = None, None
                    if hasattr(self.microenv_visualizer, 'get_slice_data_range'):
                        slice_min_val, slice_max_val = self.microenv_visualizer.get_slice_data_range()
                    
                    # Update all the ranges
                    self.control_panel.update_min_max_ranges(
                        min_val, max_val,
                        cell_min_val, cell_max_val,
                        slice_min_val, slice_max_val
                    )
                
                # Set visibility based on option
                self.microenv_visualizer.set_visibility(
                    self.collected_display_options.get('show_microenv', True)
                )
        
        # Check for cell variable selection
        if 'cell_var' in selection and selection['cell_var']:
            cell_var = selection['cell_var']
            
            if self.debug:
                print(f"Applying cell variable: {cell_var['name']}")
            
            # Update the cell visualization to color based on selected variable
            has_valid_cells = (
                'cells_mat' in self.current_frame_data and 
                self.current_frame_data['cells_mat'] is not None and
                'cells' in self.current_frame_data['cells_mat'] and
                isinstance(self.current_frame_data['cells_mat']['cells'], np.ndarray) and
                self.current_frame_data['cells_mat']['cells'].size > 0 and
                self.current_frame_data['cells_mat']['cells'].shape[1] > 0
            )
            
            if has_valid_cells:
                cells_data = self.current_frame_data['cells_mat']['cells']
                
                # Get coloring variable if specified
                color_var = selection.get('cell_color_var', None)
                
                # Determine visualization mode
                as_spheres = selection.get('cell_as_spheres', True)
                
                # Get cell color range settings
                cell_auto_range = self.collected_display_options.get('cell_auto_range', True)
                cell_min_val = self.collected_display_options.get('cell_min_value', 0.0)
                cell_max_val = self.collected_display_options.get('cell_max_value', 1.0)
                
                # Clear existing cell visualization
                self.cells_visualizer.clear()
                
                # Create new visualization based on selected variables
                if color_var:
                    # Get the variable indices for visualization and coloring
                    var_index = cell_var['index']  # Index of variable to visualize
                    color_var_index = color_var['index']  # Index of variable to use for coloring
                    
                    # Debug info - useful for checking indices
                    if self.debug:
                        print(f"Cells data shape: {cells_data.shape}")
                        print(f"Visualizing with var_index={var_index}, color_var_index={color_var_index}")
                        if var_index >= cells_data.shape[0]:
                            print(f"WARNING: var_index {var_index} is out of bounds for data shape {cells_data.shape}")
                        if color_var_index >= cells_data.shape[0]:
                            print(f"WARNING: color_var_index {color_var_index} is out of bounds for data shape {cells_data.shape}")
                        
                    # Clamp indices to valid range to prevent crashes
                    var_index = min(var_index, cells_data.shape[0] - 1)
                    color_var_index = min(color_var_index, cells_data.shape[0] - 1)
                    
                    self.cells_visualizer.visualize_cells_with_variable(
                        cells_data, 
                        var_index,  # Index of variable to visualize
                        color_var_index,  # Index of variable to use for coloring
                        opacity=self.collected_display_options.get('cell_opacity', 70)/100.0,
                        as_spheres=as_spheres,
                        auto_range=cell_auto_range,
                        min_val=cell_min_val,
                        max_val=cell_max_val
                    )
                else:
                    # Default visualization by cell type
                    if cells_data.shape[0] >= 80 and cells_data.shape[1] == 1:
                        # Single cell
                        self.cells_visualizer.visualize_single_cell(
                            cells_data,
                            self.collected_display_options.get('cell_opacity', 70)/100.0
                        )
                    elif cells_data.shape[0] >= 80:
                        # Multiple cells
                        self.cells_visualizer.visualize_multiple_cells(
                            cells_data,
                            self.collected_display_options.get('cell_opacity', 70)/100.0
                        )
                
                # Set visibility based on option
                self.cells_visualizer.set_visibility(
                    self.collected_display_options.get('show_cells', True)
                )
                
                # Update the cell range values in the control panel if auto range is enabled
                if cell_auto_range and hasattr(self.cells_visualizer, 'get_data_range'):
                    cell_min_val, cell_max_val = self.cells_visualizer.get_data_range()
                    
                    # Get other ranges
                    min_val, max_val = None, None
                    if hasattr(self.microenv_visualizer, 'get_data_range'):
                        min_val, max_val = self.microenv_visualizer.get_data_range()
                    
                    slice_min_val, slice_max_val = None, None
                    if hasattr(self.microenv_visualizer, 'get_slice_data_range'):
                        slice_min_val, slice_max_val = self.microenv_visualizer.get_slice_data_range()
                    
                    # Update all the ranges
                    self.control_panel.update_min_max_ranges(
                        min_val or 0.0, max_val or 1.0,
                        cell_min_val, cell_max_val,
                        slice_min_val, slice_max_val
                    )
            else:
                # No valid cells to visualize - clear existing cell visualization
                self.cells_visualizer.clear()
                if self.debug:
                    print("No valid cells data to visualize")
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Clean up VTK resources
        self.vtk_panel.cleanup()
        
        # Stop animation timer if running
        if self.animation_active:
            self.animation_timer.stop()
        
        # Accept the close event
        event.accept()
