#!/usr/bin/env python3
"""
vtk_qt_viewer.py - A Qt-based GUI for PhysiCell visualization with VTK

This application provides a user-friendly interface for visualizing PhysiCell output data:
- Qt-based GUI with intuitive controls
- VTK visualization of cell data from .mat files
- Navigation between frames with buttons and slider
- Display of key information and statistics

Usage:
  python vtk_qt_viewer.py
  python vtk_qt_viewer.py -f /path/to/output_folder
  python vtk_qt_viewer.py -f /path/to/output_folder -debug
"""

import os
import sys
import numpy as np
import scipy.io as sio
import vtk
from pathlib import Path
import argparse

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QSlider, QLabel, QFileDialog, QFrame, QSplitter,
    QGroupBox, QRadioButton, QCheckBox, QSpinBox, QStyle, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QTimer
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

try:
    # Try to import pyMCDS if available (for loading PhysiCell data)
    from pyMCDS_cells import pyMCDS_cells
except ImportError:
    pyMCDS_cells = None

# Global debug flag
DEBUG = False

class PhysiCellVTKQtViewer(QMainWindow):
    """Qt-based GUI application for PhysiCell VTK visualization"""
    
    def __init__(self, initial_dir=None, debug=False):
        """Initialize the application window and UI components"""
        super().__init__()
        
        # Store debug setting
        self.debug = debug
        
        # Application state
        self.output_dir = ""
        self.current_frame = 0
        self.max_frame = 0
        self.actors = []
        self.data_loaded = False
        self.initial_dir = initial_dir or os.path.expanduser("~")
        
        # Slice-related state
        self.slice_actor = None
        self.slice_mapper = None
        self.slice_plane = None
        self.slice_contour_actor = None
        self.slice_contour_mapper = None
        self.microenv_data = None
        self.microenv_vol_prop = None  # Store volume properties for consistent slice appearance
        
        # Set up the main window
        self.setWindowTitle("PhysiCell VTK Viewer")
        self.resize(1200, 800)
        
        # Create the central widget and main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create a splitter for resizable panels
        self.splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(self.splitter, 1)
        
        # Create control panel (left side)
        self.control_panel = self.create_control_panel()
        self.splitter.addWidget(self.control_panel)
        
        # Create VTK panel (right side)
        self.vtk_panel = self.create_vtk_panel()
        self.splitter.addWidget(self.vtk_panel)
        
        # Set initial splitter sizes (25% controls, 75% VTK view)
        self.splitter.setSizes([300, 900])
        
        # Create status bar at the bottom
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready. Please load a PhysiCell output directory.")
        
        # Set up VTK visualization
        self.setup_vtk_visualization()
        
        # Initialize with demo visualization
        self.add_demo_visualization()
    
    def create_control_panel(self):
        """Create the control panel with UI controls"""
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        
        # File loading controls
        file_group = QGroupBox("Data Loading")
        file_layout = QVBoxLayout(file_group)
        
        self.load_btn = QPushButton("Load Directory")
        self.load_btn.clicked.connect(self.load_directory)
        file_layout.addWidget(self.load_btn)
        
        self.directory_label = QLabel("No directory loaded")
        self.directory_label.setWordWrap(True)
        file_layout.addWidget(self.directory_label)
        
        control_layout.addWidget(file_group)
        
        # Frame navigation controls
        nav_group = QGroupBox("Frame Navigation")
        nav_layout = QVBoxLayout(nav_group)
        
        # Frame slider
        slider_layout = QHBoxLayout()
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.valueChanged.connect(self.slider_changed)
        slider_layout.addWidget(self.frame_slider)
        
        self.frame_spinbox = QSpinBox()
        self.frame_spinbox.setEnabled(False)
        self.frame_spinbox.setMinimum(0)
        self.frame_spinbox.setMaximum(0)
        self.frame_spinbox.valueChanged.connect(self.spinbox_changed)
        slider_layout.addWidget(self.frame_spinbox)
        
        nav_layout.addLayout(slider_layout)
        
        # Navigation buttons
        btn_layout = QHBoxLayout()
        
        self.prev_btn = QPushButton()
        self.prev_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaSkipBackward))
        self.prev_btn.setEnabled(False)
        self.prev_btn.clicked.connect(self.previous_frame)
        btn_layout.addWidget(self.prev_btn)
        
        self.play_btn = QPushButton()
        self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self.toggle_play)
        btn_layout.addWidget(self.play_btn)
        
        self.next_btn = QPushButton()
        self.next_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaSkipForward))
        self.next_btn.setEnabled(False)
        self.next_btn.clicked.connect(self.next_frame)
        btn_layout.addWidget(self.next_btn)
        
        nav_layout.addLayout(btn_layout)
        
        self.frame_info_label = QLabel("Frame: 0")
        nav_layout.addWidget(self.frame_info_label)
        
        control_layout.addWidget(nav_group)
        
        # Display options
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout(display_group)
        
        self.microenv_wireframe_cb = QCheckBox("Microenvironment Wireframe")
        self.microenv_wireframe_cb.setChecked(False)
        self.microenv_wireframe_cb.setEnabled(False)
        self.microenv_wireframe_cb.stateChanged.connect(self.update_visualization)
        display_layout.addWidget(self.microenv_wireframe_cb)
        
        self.show_mat_cb = QCheckBox("Show Cell Data")
        self.show_mat_cb.setChecked(True)
        self.show_mat_cb.setEnabled(False)
        self.show_mat_cb.stateChanged.connect(self.update_visualization)
        display_layout.addWidget(self.show_mat_cb)
        
        self.show_microenv_cb = QCheckBox("Show Microenvironment")
        self.show_microenv_cb.setChecked(True)
        self.show_microenv_cb.setEnabled(False)
        self.show_microenv_cb.stateChanged.connect(self.update_visualization)
        display_layout.addWidget(self.show_microenv_cb)
        
        # Add slice controls
        slice_group = QGroupBox("Slice Controls")
        slice_layout = QVBoxLayout(slice_group)
        
        # Slice position controls
        pos_group = QGroupBox("Slice Position")
        pos_layout = QVBoxLayout(pos_group)
        
        # X position
        x_layout = QHBoxLayout()
        x_layout.addWidget(QLabel("X:"))
        self.slice_x = QSpinBox()
        self.slice_x.setRange(-1000, 1000)
        self.slice_x.setValue(0)
        self.slice_x.valueChanged.connect(self.update_slice)
        x_layout.addWidget(self.slice_x)
        pos_layout.addLayout(x_layout)
        
        # Y position
        y_layout = QHBoxLayout()
        y_layout.addWidget(QLabel("Y:"))
        self.slice_y = QSpinBox()
        self.slice_y.setRange(-1000, 1000)
        self.slice_y.setValue(0)
        self.slice_y.valueChanged.connect(self.update_slice)
        y_layout.addWidget(self.slice_y)
        pos_layout.addLayout(y_layout)
        
        # Z position
        z_layout = QHBoxLayout()
        z_layout.addWidget(QLabel("Z:"))
        self.slice_z = QSpinBox()
        self.slice_z.setRange(-1000, 1000)
        self.slice_z.setValue(0)
        self.slice_z.valueChanged.connect(self.update_slice)
        z_layout.addWidget(self.slice_z)
        pos_layout.addLayout(z_layout)
        
        slice_layout.addWidget(pos_group)
        
        # Slice orientation controls
        orient_group = QGroupBox("Slice Orientation")
        orient_layout = QVBoxLayout(orient_group)
        
        # I orientation
        i_layout = QHBoxLayout()
        i_layout.addWidget(QLabel("I:"))
        self.slice_i = QDoubleSpinBox()
        self.slice_i.setRange(-1.0, 1.0)
        self.slice_i.setSingleStep(0.1)
        self.slice_i.setValue(0.0)
        self.slice_i.valueChanged.connect(self.update_slice)
        i_layout.addWidget(self.slice_i)
        orient_layout.addLayout(i_layout)
        
        # J orientation
        j_layout = QHBoxLayout()
        j_layout.addWidget(QLabel("J:"))
        self.slice_j = QDoubleSpinBox()
        self.slice_j.setRange(-1.0, 1.0)
        self.slice_j.setSingleStep(0.1)
        self.slice_j.setValue(0.0)
        self.slice_j.valueChanged.connect(self.update_slice)
        j_layout.addWidget(self.slice_j)
        orient_layout.addLayout(j_layout)
        
        # K orientation
        k_layout = QHBoxLayout()
        k_layout.addWidget(QLabel("K:"))
        self.slice_k = QDoubleSpinBox()
        self.slice_k.setRange(-1.0, 1.0)
        self.slice_k.setSingleStep(0.1)
        self.slice_k.setValue(1.0)
        self.slice_k.valueChanged.connect(self.update_slice)
        k_layout.addWidget(self.slice_k)
        orient_layout.addLayout(k_layout)
        
        slice_layout.addWidget(orient_group)
        
        # Show slice checkbox
        self.show_slice_cb = QCheckBox("Show Slice")
        self.show_slice_cb.setChecked(False)
        self.show_slice_cb.setEnabled(False)
        self.show_slice_cb.stateChanged.connect(self.update_slice)
        slice_layout.addWidget(self.show_slice_cb)
        
        display_layout.addWidget(slice_group)
        
        self.microenv_opacity_slider = QSlider(Qt.Horizontal)
        self.microenv_opacity_slider.setRange(0, 100)
        self.microenv_opacity_slider.setValue(50)
        self.microenv_opacity_slider.setEnabled(False)
        self.microenv_opacity_slider.valueChanged.connect(self.update_visualization)
        display_layout.addWidget(QLabel("Microenvironment Opacity:"))
        display_layout.addWidget(self.microenv_opacity_slider)
        
        self.cell_opacity_slider = QSlider(Qt.Horizontal)
        self.cell_opacity_slider.setRange(0, 100)
        self.cell_opacity_slider.setValue(70)
        self.cell_opacity_slider.setEnabled(False)
        self.cell_opacity_slider.valueChanged.connect(self.update_visualization)
        display_layout.addWidget(QLabel("Cell Opacity:"))
        display_layout.addWidget(self.cell_opacity_slider)
        
        control_layout.addWidget(display_group)
        
        # Add info panel at the bottom of control section
        info_group = QGroupBox("Information")
        info_layout = QVBoxLayout(info_group)
        
        self.info_label = QLabel("Load a directory to begin visualization.")
        self.info_label.setWordWrap(True)
        info_layout.addWidget(self.info_label)
        
        control_layout.addWidget(info_group)
        
        # Spacer at the bottom to push everything up
        control_layout.addStretch()
        
        return control_widget
    
    def create_vtk_panel(self):
        """Create the VTK visualization panel"""
        vtk_widget = QWidget()
        vtk_layout = QVBoxLayout(vtk_widget)
        
        # Create the VTK widget
        self.vtk_widget = QVTKRenderWindowInteractor()
        vtk_layout.addWidget(self.vtk_widget)
        
        return vtk_widget
    
    def setup_vtk_visualization(self):
        """Set up VTK rendering components"""
        # Get the render window
        self.render_window = self.vtk_widget.GetRenderWindow()
        
        # Enable anti-aliasing for smoother rendering
        self.render_window.SetMultiSamples(4)
        
        # Set rendering quality settings
        self.render_window.SetDesiredUpdateRate(30.0)  # Higher update rate during interaction
        
        # Create a renderer
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.1, 0.2, 0.3)  # Dark blue background
        
        # Enable two-sided lighting for better visuals
        self.renderer.SetTwoSidedLighting(True)
        
        # Set rendering quality
        self.renderer.SetUseDepthPeeling(1)  # Better transparency handling
        self.renderer.SetMaximumNumberOfPeels(8)
        self.renderer.SetOcclusionRatio(0.0)
        
        self.render_window.AddRenderer(self.renderer)
        
        # Set up the interactor
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        
        # Create a custom interactor style for faster zooming
        class FastZoomInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
            def __init__(self):
                super().__init__()
                self.zoom_factor = 0.2  # Default zoom factor (larger = faster zoom)
                
            def SetZoomFactor(self, factor):
                self.zoom_factor = factor
                
            def MouseWheelForwardEvent(self, obj, event):
                # Override default zoom to make it faster
                factor = self.zoom_factor * 4.0  # 4x faster than default
                self.GetCurrentRenderer().GetActiveCamera().Dolly(1.0 + factor)
                self.GetCurrentRenderer().ResetCameraClippingRange()
                self.GetInteractor().Render()
                
            def MouseWheelBackwardEvent(self, obj, event):
                # Override default zoom to make it faster
                factor = self.zoom_factor * 4.0  # 4x faster than default
                self.GetCurrentRenderer().GetActiveCamera().Dolly(1.0 - factor)
                self.GetCurrentRenderer().ResetCameraClippingRange()
                self.GetInteractor().Render()
        
        # Set the custom interactor style
        interactor_style = FastZoomInteractorStyle()
        interactor_style.SetZoomFactor(0.2)  # Set zoom factor (0.1 = slow, 0.5 = fast)
        interactor_style.SetMotionFactor(10.0)  # Faster motion
        
        self.interactor.SetInteractorStyle(interactor_style)
        
        # Configure the interactor for improved performance
        self.interactor.SetDesiredUpdateRate(30.0)  # Higher update rate during interaction
        self.interactor.SetStillUpdateRate(0.01)    # Lower update rate when not interacting
        
        # Add custom observers for interaction events
        self.interactor.AddObserver("StartInteractionEvent", self.on_interaction_start)
        self.interactor.AddObserver("EndInteractionEvent", self.on_interaction_end)
        
        # Add orientation axes
        self.add_orientation_axes()
        
        # Setup camera for better zooming
        camera = self.renderer.GetActiveCamera()
        camera.SetClippingRange(0.1, 10000)  # Wide clipping range
        
        # Initialize the interactor
        self.interactor.Initialize()
        
        # Set up an animation timer
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.animation_step)
        self.animation_active = False
        
        # Track actor properties for LOD during interaction
        self.actor_properties = {}  # Store original properties
    
    def add_orientation_axes(self):
        """Add orientation axes to the renderer"""
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(20, 20, 20)
        axes.GetXAxisCaptionActor2D().SetCaption("X")
        axes.GetYAxisCaptionActor2D().SetCaption("Y")
        axes.GetZAxisCaptionActor2D().SetCaption("Z")
        
        # Position the axes widget
        self.axes_widget = vtk.vtkOrientationMarkerWidget()
        self.axes_widget.SetOrientationMarker(axes)
        self.axes_widget.SetInteractor(self.interactor)
        self.axes_widget.SetViewport(0.0, 0.0, 0.2, 0.2)
        self.axes_widget.EnabledOn()
        self.axes_widget.InteractiveOff()
    
    def add_demo_visualization(self):
        """Add a simple demo visualization when no data is loaded"""
        # Create a simple sphere
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(0, 0, 0)
        sphere.SetRadius(50)
        sphere.SetPhiResolution(30)
        sphere.SetThetaResolution(30)
        
        # Create a mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())
        
        # Create an actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.8, 0.8, 0.8)  # Light gray
        
        # Add the actor to the renderer
        self.renderer.AddActor(actor)
        self.actors.append(actor)
        
        # Add text indicating demo mode
        text_actor = vtk.vtkTextActor()
        text_actor.SetInput("Demo Mode - Load a directory to begin")
        text_actor.GetTextProperty().SetFontSize(18)
        text_actor.GetTextProperty().SetColor(1, 1, 0)  # Yellow text
        text_actor.SetPosition(20, 30)
        self.renderer.AddActor2D(text_actor)
        self.actors.append(text_actor)
        
        # Reset the camera
        self.renderer.ResetCamera()
        self.render_window.Render()
    
    def clear_visualization(self):
        """Clear all actors from the renderer"""
        # Keep the slice actors if they exist and slice is enabled
        slice_actors = []
        if self.slice_actor and self.show_slice_cb.isChecked():
            slice_actors.append(self.slice_actor)
        if self.slice_contour_actor and self.show_slice_cb.isChecked():
            slice_actors.append(self.slice_contour_actor)
            
        # Remove all actors except slice actors
        for actor in self.actors:
            if actor not in slice_actors:
                self.renderer.RemoveActor(actor)
                
        # Update actor list to keep only slice actors
        if slice_actors:
            self.actors = slice_actors
        else:
            self.actors = []
    
    def load_directory(self):
        """Load a PhysiCell output directory"""
        directory = QFileDialog.getExistingDirectory(
            self, "Select PhysiCell Output Directory", 
            self.initial_dir,
            QFileDialog.ShowDirsOnly
        )
        
        if directory:
            self.output_dir = directory
            self.directory_label.setText(directory)
            self.status_bar.showMessage(f"Loading directory: {directory}")
            
            # Find the maximum frame number
            self.find_max_frame()
            
            if self.max_frame > 0:
                # Update controls
                self.frame_slider.setMaximum(self.max_frame)
                self.frame_slider.setValue(0)
                self.frame_spinbox.setMaximum(self.max_frame)
                self.frame_spinbox.setValue(0)
                
                # Enable controls
                self.frame_slider.setEnabled(True)
                self.frame_spinbox.setEnabled(True)
                self.prev_btn.setEnabled(True)
                self.next_btn.setEnabled(True)
                self.play_btn.setEnabled(True)
                self.microenv_wireframe_cb.setEnabled(True)
                self.show_mat_cb.setEnabled(True)
                self.show_microenv_cb.setEnabled(True)
                self.cell_opacity_slider.setEnabled(True)
                self.microenv_opacity_slider.setEnabled(True)
                self.show_slice_cb.setEnabled(True)  # Enable slice controls
                self.slice_x.setEnabled(True)
                self.slice_y.setEnabled(True)
                self.slice_z.setEnabled(True)
                self.slice_i.setEnabled(True)
                self.slice_j.setEnabled(True)
                self.slice_k.setEnabled(True)
                
                # Load the first frame
                self.current_frame = 0
                self.load_frame(self.current_frame)
                self.data_loaded = True
                
                self.status_bar.showMessage(f"Loaded directory: {directory}, {self.max_frame+1} frames found")
            else:
                self.status_bar.showMessage(f"No PhysiCell output files found in {directory}")
                self.info_label.setText("No PhysiCell output files found in the selected directory.")
    
    def find_max_frame(self):
        """Find the maximum frame number in the output directory"""
        max_frame = -1
        
        # Check for XML files
        for filename in os.listdir(self.output_dir):
            if filename.startswith("output") and filename.endswith(".xml"):
                try:
                    frame_num = int(filename[6:-4])
                    max_frame = max(max_frame, frame_num)
                except ValueError:
                    pass
                    
        # Check for .mat files (cells)
        for filename in os.listdir(self.output_dir):
            if filename.startswith("output") and filename.endswith("_cells.mat"):
                try:
                    frame_num = int(filename[6:-10])
                    max_frame = max(max_frame, frame_num)
                except ValueError:
                    pass
                    
        # Check for .mat files (microenvironment)
        for filename in os.listdir(self.output_dir):
            if filename.startswith("output") and filename.endswith("_microenvironment0.mat"):
                try:
                    frame_num = int(filename[6:-20])
                    max_frame = max(max_frame, frame_num)
                except ValueError:
                    pass
        
        # Check for SVG files
        for filename in os.listdir(self.output_dir):
            if filename.startswith("snapshot") and filename.endswith(".svg"):
                try:
                    frame_num = int(filename[8:-4])
                    max_frame = max(max_frame, frame_num)
                except ValueError:
                    pass
        
        self.max_frame = max_frame
        return max_frame
    
    def load_frame(self, frame_number):
        """Load and visualize the specified frame"""
        if not self.output_dir:
            return
            
        self.current_frame = frame_number
        self.frame_info_label.setText(f"Frame: {frame_number}")
        
        # Clear previous visualization
        self.clear_visualization()
        
        # Construct file paths
        xml_file = os.path.join(self.output_dir, f"output{frame_number:08d}.xml")
        cell_mat_file = os.path.join(self.output_dir, f"output{frame_number:08d}_cells.mat")
        microenv_mat_file = os.path.join(self.output_dir, f"output{frame_number:08d}_microenvironment0.mat")
        
        # Try to visualize the XML file if it exists
        if os.path.exists(xml_file):
            self.visualize_cells_from_xml(xml_file)
        
        # Try to visualize the cells .mat file if it exists and .mat data should be shown
        if os.path.exists(cell_mat_file) and self.show_mat_cb.isChecked():
            self.visualize_mat_file(cell_mat_file)
            
        # Try to visualize the microenvironment .mat file if it exists
        if os.path.exists(microenv_mat_file) and self.show_microenv_cb.isChecked():
            self.visualize_microenvironment(microenv_mat_file)
        
        # Reset the camera if this is the first time loading a frame
        if not self.data_loaded:
            self.renderer.ResetCamera()
            
        # Update the render window
        self.render_window.Render()
        
        # Update info label
        info_text = f"Frame: {frame_number}"
        if os.path.exists(xml_file):
            info_text += f"\nXML: {os.path.basename(xml_file)}"
        if os.path.exists(cell_mat_file):
            info_text += f"\nCells: {os.path.basename(cell_mat_file)}"
        if os.path.exists(microenv_mat_file):
            info_text += f"\nMicroenv: {os.path.basename(microenv_mat_file)}"
        self.info_label.setText(info_text)
    
    def visualize_cells_from_xml(self, xml_file):
        """Visualize cells from a PhysiCell XML file"""
        if pyMCDS_cells is None:
            self.info_label.setText("XML visualization requires pyMCDS_cells module")
            return False
            
        try:
            # Load cell data
            mcds = pyMCDS_cells(os.path.basename(xml_file), os.path.dirname(xml_file))
            
            # Get cell positions and types
            cell_df = mcds.get_cell_df()
            positions_x = cell_df['position_x'].values
            positions_y = cell_df['position_y'].values
            if 'position_z' in cell_df.columns:
                positions_z = cell_df['position_z'].values
            else:
                positions_z = np.zeros_like(positions_x)
            
            # Print cell bounds information from XML
            if self.debug:
                print("\n==== Cell Bounds from XML ====")
                print(f"X range: {positions_x.min():.6f} to {positions_x.max():.6f}")
                print(f"Y range: {positions_y.min():.6f} to {positions_y.max():.6f}")
                print(f"Z range: {positions_z.min():.6f} to {positions_z.max():.6f}")
                print(f"Number of cells: {len(positions_x)}")
            
            # Try to get microenvironment data bounds for comparison
            try:
                microenv_file = xml_file.replace(".xml", "_microenvironment0.mat")
                if os.path.exists(microenv_file):
                    microenv_contents = self.load_mat_file(microenv_file)
                    if 'multiscale_microenvironment' in microenv_contents:
                        microenv_data = microenv_contents['multiscale_microenvironment']
                        
                        # Extract coordinate bounds from microenvironment data
                        micro_x = microenv_data[0, :].flatten()
                        micro_y = microenv_data[1, :].flatten()
                        micro_z = microenv_data[2, :].flatten()
                        
                        # Print microenvironment bounds
                        if self.debug:
                            print("\n==== Microenvironment Bounds ====")
                            print(f"X range: {micro_x.min():.6f} to {micro_x.max():.6f}")
                            print(f"Y range: {micro_y.min():.6f} to {micro_y.max():.6f}")
                            print(f"Z range: {micro_z.min():.6f} to {micro_z.max():.6f}")
                            
                            # Calculate bounds differences
                            x_diff_min = positions_x.min() - micro_x.min()
                            x_diff_max = micro_x.max() - positions_x.max()
                            y_diff_min = positions_y.min() - micro_y.min()
                            y_diff_max = micro_y.max() - positions_y.max()
                            z_diff_min = positions_z.min() - micro_z.min()
                            z_diff_max = micro_z.max() - positions_z.max()
                            
                            print("\n==== Bounds Differences (positive means cells are inside) ====")
                            print(f"X min difference: {x_diff_min:.6f}")
                            print(f"X max difference: {x_diff_max:.6f}")
                            print(f"Y min difference: {y_diff_min:.6f}")
                            print(f"Y max difference: {y_diff_max:.6f}")
                            print(f"Z min difference: {z_diff_min:.6f}")
                            print(f"Z max difference: {z_diff_max:.6f}")
            except Exception as e:
                if self.debug:
                    print(f"Error comparing with microenvironment: {e}")
            
            # Calculate radii
            cell_vols = cell_df['total_volume'].values
            four_thirds_pi = 4.188790204786391
            cell_radii = np.divide(cell_vols, four_thirds_pi)
            cell_radii = np.power(cell_radii, 0.333333333333333333333333)
            
            # Get cell types
            cell_types = cell_df['cell_type'].values.astype(int)
            
            # Color mapping for different cell types
            cell_type_colors = {
                0: (180, 180, 180),  # Light gray for default
                1: (255, 0, 0),      # Red for type 1
                2: (0, 255, 0),      # Green for type 2
                3: (0, 0, 255),      # Blue for type 3
                4: (255, 255, 0),    # Yellow for type 4
                5: (255, 0, 255),    # Magenta for type 5
                6: (0, 255, 255),    # Cyan for type 6
            }
            
            # Create a direct polydata visualization instead of using glyphs
            num_cells = len(positions_x)
            all_points = vtk.vtkPoints()
            all_cells = vtk.vtkCellArray()
            
            # Create color array
            colors = vtk.vtkUnsignedCharArray()
            colors.SetNumberOfComponents(3)
            colors.SetName("Colors")
            
            # For each cell, create a colored sphere as a polydata
            for i in range(num_cells):
                # Get cell properties
                x, y, z = positions_x[i], positions_y[i], positions_z[i]
                radius = cell_radii[i]
                
                # Get color for this cell type
                ct = cell_types[i]
                color = cell_type_colors.get(ct, (180, 180, 180))
                
                # Create sphere for this cell
                sphere_source = vtk.vtkSphereSource()
                sphere_source.SetCenter(x, y, z)
                sphere_source.SetRadius(radius)  # Use actual radius without scaling down
                sphere_source.SetPhiResolution(16)
                sphere_source.SetThetaResolution(16)
                sphere_source.Update()
                
                # Get polydata from the sphere source
                sphere_polydata = sphere_source.GetOutput()
                
                # Get number of points in the current polydata
                num_points = all_points.GetNumberOfPoints()
                
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
            opacity = self.cell_opacity_slider.value() / 100.0
            actor.GetProperty().SetOpacity(opacity)
            
            # Add the actor to the renderer
            self.renderer.AddActor(actor)
            self.actors.append(actor)
            
            # Add cell count info
            time_mins = mcds.get_time()
            hrs = int(time_mins/60)
            days = int(hrs/24)
            time_str = f"{days}d, {hrs-days*24}h, {int(time_mins-hrs*60)}m"
            
            text_actor = vtk.vtkTextActor()
            text_actor.SetInput(f"Cells: {num_cells}, Time: {time_str}")
            text_actor.GetTextProperty().SetFontSize(16)
            text_actor.GetTextProperty().SetColor(1, 1, 0)
            text_actor.SetPosition(20, 60)
            self.renderer.AddActor2D(text_actor)
            self.actors.append(text_actor)
            
            return True
            
        except Exception as e:
            if self.debug:
                print(f"Error visualizing XML file: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_mat_file(self, file_path):
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
            if self.debug:
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
            if self.debug:
                print(f"Error loading .mat file: {e}")
            return None
    
    def create_spiral_viz_from_values(self, values):
        """Create a spiral visualization from an array of values"""
        if values is None or len(values) == 0:
            return []
        
        # Generate spiral coordinates
        n_points = len(values)
        t = np.linspace(0, 4*np.pi, n_points)
        
        # Create a spiral in 3D space
        radius = 50
        x = radius * np.cos(t)
        y = radius * np.sin(t)
        z = 10 * t
        
        # Scale values for visualization
        min_val = values.min()
        max_val = values.max()
        if min_val == max_val:
            normalized_values = np.ones_like(values)
        else:
            normalized_values = (values - min_val) / (max_val - min_val)
        
        # Create points for the polyline
        points = vtk.vtkPoints()
        for i in range(n_points):
            points.InsertNextPoint(x[i], y[i], z[i])
        
        # Create a polyline
        lines = vtk.vtkCellArray()
        polyline = vtk.vtkPolyLine()
        polyline.GetPointIds().SetNumberOfIds(n_points)
        for i in range(n_points):
            polyline.GetPointIds().SetId(i, i)
        lines.InsertNextCell(polyline)
        
        # Create polydata for the polyline
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetLines(lines)
        
        # Add scalar values to the polydata
        scalars = vtk.vtkFloatArray()
        scalars.SetNumberOfComponents(1)
        scalars.SetName("Values")
        for value in values:
            scalars.InsertNextValue(value)
        polydata.GetPointData().SetScalars(scalars)
        
        # Create a tube filter for better visualization
        tube_filter = vtk.vtkTubeFilter()
        tube_filter.SetInputData(polydata)
        tube_filter.SetRadius(3.0)  # Increased radius
        tube_filter.SetNumberOfSides(16)  # Higher resolution
        tube_filter.SetVaryRadiusToVaryRadiusByScalar()
        tube_filter.SetRadiusFactor(5.0)  # Scale by scalar values
        tube_filter.Update()
        
        # Create mapper for the tube
        tube_mapper = vtk.vtkPolyDataMapper()
        tube_mapper.SetInputConnection(tube_filter.GetOutputPort())
        tube_mapper.SetScalarRange(min_val, max_val)
        
        # Create actor for the tube
        tube_actor = vtk.vtkActor()
        tube_actor.SetMapper(tube_mapper)
        
        # Create a sphere source for the endpoints
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(5.0)
        sphere.SetPhiResolution(16)
        sphere.SetThetaResolution(16)
        sphere.Update()
        
        # Create a point glyph just for start/end markers
        start_point = vtk.vtkPoints()
        start_point.InsertNextPoint(x[0], y[0], z[0])
        end_point = vtk.vtkPoints()
        end_point.InsertNextPoint(x[-1], y[-1], z[-1])
        
        # Create polydata for the points
        start_polydata = vtk.vtkPolyData()
        start_polydata.SetPoints(start_point)
        end_polydata = vtk.vtkPolyData()
        end_polydata.SetPoints(end_point)
        
        # Create a mapper for start point
        start_mapper = vtk.vtkPolyDataMapper()
        start_mapper.SetInputData(sphere.GetOutput())
        start_actor = vtk.vtkActor()
        start_actor.SetMapper(start_mapper)
        start_actor.SetPosition(x[0], y[0], z[0])
        start_actor.GetProperty().SetColor(0, 0, 1)  # Blue for start
        
        # Create a mapper for end point
        end_mapper = vtk.vtkPolyDataMapper()
        end_mapper.SetInputData(sphere.GetOutput())
        end_actor = vtk.vtkActor()
        end_actor.SetMapper(end_mapper)
        end_actor.SetPosition(x[-1], y[-1], z[-1])
        end_actor.GetProperty().SetColor(1, 0, 0)  # Red for end
        
        # Add a color bar
        scalar_bar = vtk.vtkScalarBarActor()
        scalar_bar.SetLookupTable(tube_mapper.GetLookupTable())
        scalar_bar.SetTitle("Cell Values")
        scalar_bar.SetNumberOfLabels(5)
        scalar_bar.SetPosition(0.8, 0.1)
        scalar_bar.SetWidth(0.1)
        scalar_bar.SetHeight(0.8)
        
        self.renderer.AddActor2D(scalar_bar)
        self.actors.append(scalar_bar)
        
        return [tube_actor, start_actor, end_actor]
    
    def visualize_mat_file(self, file_path):
        """Visualize data from a .mat file"""
        try:
            mat_contents = self.load_mat_file(file_path)
            
            if mat_contents is None:
                return False
            
            # Check for cells data
            if 'cells' in mat_contents:
                cells_data = mat_contents['cells']
                
                # Print detailed information about the cells data structure
                if self.debug:
                    print("\n==== PhysiCell Cell Data Structure Analysis ====")
                    print(f"Shape of cells array: {cells_data.shape}")
                
                # Handle the case where cells_data is an array with 87 elements (single cell)
                if cells_data.shape[0] >= 80 and cells_data.shape[1] == 1:
                    # Process single cell format (87 properties x 1 cell)
                    if self.debug:
                        print("Detected PhysiCell single cell format (87 properties x 1 cell)")
                    
                    # Extract cell position (typically stored at indices 1, 2, 3)
                    x = float(cells_data[1, 0])
                    y = float(cells_data[2, 0])
                    z = float(cells_data[3, 0])
                    
                    # Extract cell radius (convert from volume)
                    volume = float(cells_data[4, 0])  # Assuming volume is at index 4
                    radius = (3.0 * volume / (4.0 * np.pi)) ** (1.0/3.0)
                    
                    # Cell type
                    cell_type = 1  # Default type if not specified
                    if cells_data.shape[0] > 5:
                        cell_type = int(cells_data[5, 0])  # Assuming type is at index 5
                    
                    if self.debug:
                        print(f"Cell position: ({x:.6f}, {y:.6f}, {z:.6f})")
                        print(f"Cell radius: {radius:.6f}")
                        print(f"Cell type: {cell_type}")
                    
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
                    colors = {
                        0: (0.7, 0.7, 0.7),  # Grey
                        1: (1.0, 0.0, 0.0),  # Red
                        2: (0.0, 1.0, 0.0),  # Green
                        3: (0.0, 0.0, 1.0),  # Blue
                        4: (1.0, 1.0, 0.0),  # Yellow
                        5: (1.0, 0.0, 1.0),  # Magenta
                        6: (0.0, 1.0, 1.0),  # Cyan
                    }
                    color = colors.get(cell_type, (0.7, 0.7, 0.7))  # Default to grey
                    actor.GetProperty().SetColor(color)
                    
                    # Set opacity
                    opacity = self.cell_opacity_slider.value() / 100.0
                    actor.GetProperty().SetOpacity(opacity)
                    
                    # Add the actor to the renderer
                    self.renderer.AddActor(actor)
                    self.actors.append(actor)
                    
                    # Add info text about the cell
                    text_actor = vtk.vtkTextActor()
                    text_actor.SetInput(f"Single cell at ({x:.2f}, {y:.2f}, {z:.2f})\nRadius: {radius:.2f}, Type: {cell_type}")
                    text_actor.GetTextProperty().SetFontSize(16)
                    text_actor.GetTextProperty().SetColor(1, 1, 0)  # Yellow text
                    text_actor.SetPosition(20, 30)
                    self.renderer.AddActor2D(text_actor)
                    self.actors.append(text_actor)
                    
                    return True
                
                # Handle multiple cells case (cells are in columns)
                elif cells_data.shape[0] >= 80:
                    # Process multiple cells format (87 properties x N cells)
                    num_cells = cells_data.shape[1]
                    if self.debug:
                        print(f"Detected PhysiCell multiple cells format (87 properties x {num_cells} cells)")
                    
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
                        radius = (3.0 * volume / (4.0 * np.pi)) ** (1.0/3.0)
                        
                        # Cell type
                        cell_type = 1  # Default type if not specified
                        if cells_data.shape[0] > 5:
                            cell_type = int(cells_data[5, cell_idx])  # Assuming type is at index 5
                        
                        if self.debug and cell_idx < 5:  # Show details for first 5 cells only
                            print(f"Cell {cell_idx} - Position: ({x:.6f}, {y:.6f}, {z:.6f}), Radius: {radius:.6f}, Type: {cell_type}")
                        
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
                        cell_type_colors = {
                            0: (180, 180, 180),  # Grey
                            1: (255, 0, 0),      # Red
                            2: (0, 255, 0),      # Green
                            3: (0, 0, 255),      # Blue
                            4: (255, 255, 0),    # Yellow
                            5: (255, 0, 255),    # Magenta
                            6: (0, 255, 255),    # Cyan
                        }
                        color = cell_type_colors.get(cell_type, (180, 180, 180))  # Default to grey
                        
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
                    opacity = self.cell_opacity_slider.value() / 100.0
                    actor.GetProperty().SetOpacity(opacity)
                    
                    # Add the actor to the renderer
                    self.renderer.AddActor(actor)
                    self.actors.append(actor)
                    
                    # Add info text about the cells
                    text_actor = vtk.vtkTextActor()
                    text_actor.SetInput(f"Multiple cells: {num_cells} total cells")
                    text_actor.GetTextProperty().SetFontSize(16)
                    text_actor.GetTextProperty().SetColor(1, 1, 0)  # Yellow text
                    text_actor.SetPosition(20, 30)
                    self.renderer.AddActor2D(text_actor)
                    self.actors.append(text_actor)
                    
                    return True
                
                # Handle scalar values case (single column, but not 87 elements format)
                elif cells_data.shape[1] == 1 and isinstance(cells_data[0, 0], (float, int, np.float64, np.int64)):
                    values = cells_data.flatten()
                    if self.debug:
                        print("Detected simple scalar values array, not PhysiCell cell format")
                    
                    # Create a spiral visualization for the values
                    spiral_actors = self.create_spiral_viz_from_values(values)
                    for actor in spiral_actors:
                        self.renderer.AddActor(actor)
                        self.actors.append(actor)
                    
                    # Add info text
                    text_actor = vtk.vtkTextActor()
                    text_actor.SetInput(f"Scalar values visualization\nNumber of values: {len(values)}")
                    text_actor.GetTextProperty().SetFontSize(16)
                    text_actor.GetTextProperty().SetColor(1, 1, 0)  # Yellow
                    text_actor.SetPosition(20, 30)
                    self.renderer.AddActor2D(text_actor)
                    self.actors.append(text_actor)
                    
                    return True
                
                # Handle other cases where cells data might have multiple columns
                else:
                    # This case is for more complex cell data structures
                    # For now, just show a message and return
                    if self.debug:
                        print(f"Complex cell data structure detected: {cells_data.shape}")
                    
                    # Add info text
                    text_actor = vtk.vtkTextActor()
                    text_actor.SetInput(f"Complex cell data structure\nShape: {cells_data.shape}")
                    text_actor.GetTextProperty().SetFontSize(16)
                    text_actor.GetTextProperty().SetColor(1, 1, 0)  # Yellow
                    text_actor.SetPosition(20, 30)
                    self.renderer.AddActor2D(text_actor)
                    self.actors.append(text_actor)
                    
                    return True
            
            # If we got here, no valid data was found to visualize
            return False
        
        except Exception as e:
            # Error handling for visualization issues
            if self.debug:
                print(f"Error visualizing MAT file: {e}")
            import traceback
            traceback.print_exc()
            
            # Add error message to the visualization
            text_actor = vtk.vtkTextActor()
            text_actor.SetInput(f"Error visualizing file:\n{str(e)}")
            text_actor.GetTextProperty().SetFontSize(16)
            text_actor.GetTextProperty().SetColor(1, 0, 0)  # Red text for errors
            text_actor.SetPosition(20, 30)
            self.renderer.AddActor2D(text_actor)
            self.actors.append(text_actor)
            
            return False
    
    def visualize_microenvironment(self, file_path):
        """Visualize microenvironment data from a .mat file"""
        try:
            mat_contents = self.load_mat_file(file_path)
            
            if mat_contents is None:
                return False
            
            # Debug output to help diagnose issues
            if self.debug:
                print(f"Microenvironment file keys: {list(mat_contents.keys())}")
            
            # Look for substrate data - PhysiCell format
            if 'multiscale_microenvironment' in mat_contents:
                microenv_data = mat_contents['multiscale_microenvironment']
                
                # Print structure information for debugging
                if self.debug:
                    print(f"Microenvironment data shape: {microenv_data.shape}")
                
                # The data is typically stored with substrates in rows and positions in columns
                # First several rows contain position info, then substrate values
                if microenv_data.shape[0] > 4:  # At least one substrate
                    # First 3 rows are x,y,z coordinates
                    x = microenv_data[0, :].flatten()  # x coordinates
                    y = microenv_data[1, :].flatten()  # y coordinates
                    z = microenv_data[2, :].flatten()  # z coordinates
                    
                    # Print coordinate information for debugging
                    if self.debug:
                        print(f"X range: {x.min():.6f} to {x.max():.6f}, shape: {x.shape}")
                        print(f"Y range: {y.min():.6f} to {y.max():.6f}, shape: {y.shape}")
                        print(f"Z range: {z.min():.6f} to {z.max():.6f}, shape: {z.shape}")
                    
                    # Number of substrates (chemical species)
                    substrate_count = microenv_data.shape[0] - 4
                    position_indices = microenv_data.shape[1]
                    
                    # Determine the grid dimensions
                    # In PhysiCell, the grid is typically structured with equally spaced points
                    unique_x = np.unique(x)
                    unique_y = np.unique(y)
                    unique_z = np.unique(z)
                    
                    nx = len(unique_x)
                    ny = len(unique_y)
                    nz = len(unique_z)
                    
                    if self.debug:
                        print(f"Grid dimensions: {nx} x {ny} x {nz}")
                    
                    # Create a structured grid for the visualization
                    if nz <= 1:  # 2D case - add a small z-dimension for visualization
                        nz = 2
                        unique_z = np.array([z[0]-0.5, z[0]+0.5]) if len(unique_z) > 0 else np.array([-0.5, 0.5])
                    
                    # Create VTK arrays for coordinates
                    x_vtk = vtk.vtkDoubleArray()
                    for val in unique_x:
                        x_vtk.InsertNextValue(val)
                        
                    y_vtk = vtk.vtkDoubleArray()
                    for val in unique_y:
                        y_vtk.InsertNextValue(val)
                        
                    z_vtk = vtk.vtkDoubleArray()
                    for val in unique_z:
                        z_vtk.InsertNextValue(val)
                    
                    # Create a rectilinear grid (which works well for regularly spaced data)
                    grid = vtk.vtkRectilinearGrid()
                    grid.SetDimensions(nx, ny, nz)
                    grid.SetXCoordinates(x_vtk)
                    grid.SetYCoordinates(y_vtk)
                    grid.SetZCoordinates(z_vtk)
                    
                    # Store the grid for slicing, independent of visualization
                    self.microenv_data = vtk.vtkRectilinearGrid()
                    self.microenv_data.DeepCopy(grid)
                    
                    # Process each substrate
                    for substrate_idx in range(substrate_count):
                        # Get the substrate data (row 4+idx in the microenvironment matrix)
                        substrate_data = microenv_data[4 + substrate_idx, :]
                        
                        # Print substrate range for debugging
                        if self.debug:
                            print(f"Substrate {substrate_idx} range: {substrate_data.min():.6f} to {substrate_data.max():.6f}")
                        
                        # Create the scalar array for this substrate
                        substrate_vtk = vtk.vtkDoubleArray()
                        substrate_vtk.SetName(f"Substrate_{substrate_idx}")
                        
                        # For a rectilinear grid, we need to map the unstructured data to the structured grid
                        # This requires reshaping/interpolating the values to fit the grid
                        
                        # Initialize a 3D array to hold the interpolated values
                        grid_values = np.zeros((nx, ny, nz))
                        
                        # Simple case: the number of points matches the grid size
                        if nx * ny == len(substrate_data) and nz <= 2:
                            # Reshape for 2D data mapped to a 3D visualization
                            grid_2d = substrate_data.reshape((ny, nx)).transpose()
                            
                            # Duplicate the 2D layer if we needed to create a fake z-dimension
                            for k in range(nz):
                                grid_values[:, :, k] = grid_2d
                                
                        else:
                            # More complex case: we need to map points to the 3D grid
                            # Use nearest neighbor mapping
                            
                            # Create KD-tree for fast nearest-neighbor lookup
                            from scipy.spatial import cKDTree
                            points = np.vstack((x, y, z)).T
                            tree = cKDTree(points)
                            
                            # Query points on the structured grid
                            all_points = []
                            for i, xi in enumerate(unique_x):
                                for j, yi in enumerate(unique_y):
                                    for k, zi in enumerate(unique_z):
                                        all_points.append([xi, yi, zi])
                            
                            all_points = np.array(all_points)
                            
                            # Find nearest neighbors
                            distances, indices = tree.query(all_points, k=1)
                            
                            # Map values to grid
                            counter = 0
                            for i in range(nx):
                                for j in range(ny):
                                    for k in range(nz):
                                        if counter < len(indices):
                                            idx = indices[counter]
                                            if idx < len(substrate_data):
                                                grid_values[i, j, k] = substrate_data[idx]
                                        counter += 1
                        
                        # Add the scalar values to the grid in VTK order (k, j, i)
                        for k in range(nz):
                            for j in range(ny):
                                for i in range(nx):
                                    substrate_vtk.InsertNextValue(grid_values[i, j, k])
                        
                        grid.GetPointData().SetScalars(substrate_vtk)
                        
                        # Also add to the stored microenv_data grid for slicing
                        self.microenv_data.GetPointData().SetScalars(substrate_vtk.NewInstance())
                        self.microenv_data.GetPointData().GetScalars().DeepCopy(substrate_vtk)
                        
                        # Create a volume mapper
                        mapper = vtk.vtkSmartVolumeMapper()
                        mapper.SetInputData(grid)
                        
                        # Create a color transfer function
                        ctf = vtk.vtkColorTransferFunction()
                        min_val = substrate_data.min()
                        max_val = substrate_data.max()
                        range_val = max_val - min_val
                        
                        if range_val > 0:
                            # Create color points that work well for visualizing concentrations
                            ctf.AddRGBPoint(min_val, 0.0, 0.0, 1.0)  # Blue for min
                            ctf.AddRGBPoint(min_val + 0.25 * range_val, 0.0, 1.0, 1.0)  # Cyan
                            ctf.AddRGBPoint(min_val + 0.5 * range_val, 0.0, 1.0, 0.0)   # Green
                            ctf.AddRGBPoint(min_val + 0.75 * range_val, 1.0, 1.0, 0.0)  # Yellow
                            ctf.AddRGBPoint(max_val, 1.0, 0.0, 0.0)  # Red for max
                        else:
                            # Handle case where all values are the same
                            ctf.AddRGBPoint(min_val, 0.0, 0.0, 1.0)
                        
                        # Create an opacity transfer function
                        otf = vtk.vtkPiecewiseFunction()
                        
                        # Set opacity based on value and user slider
                        opacity_factor = self.microenv_opacity_slider.value() / 100.0
                        
                        # For values close to minimum, set lower opacity
                        otf.AddPoint(min_val, 0.0)
                        
                        # Higher opacity for higher values, scaled by the user slider
                        if range_val > 0:
                            otf.AddPoint(min_val + 0.25 * range_val, 0.1 * opacity_factor)
                            otf.AddPoint(min_val + 0.5 * range_val, 0.3 * opacity_factor)
                            otf.AddPoint(min_val + 0.75 * range_val, 0.6 * opacity_factor)
                            otf.AddPoint(max_val, 0.8 * opacity_factor)
                        else:
                            otf.AddPoint(min_val, 0.5 * opacity_factor)
                        
                        # Create volume properties
                        volume_property = vtk.vtkVolumeProperty()
                        volume_property.SetColor(ctf)
                        volume_property.SetScalarOpacity(otf)
                        volume_property.ShadeOn()
                        volume_property.SetInterpolationTypeToLinear()
                        
                        # Store the volume property for slice visualization consistency
                        self.microenv_vol_prop = vtk.vtkVolumeProperty()
                        self.microenv_vol_prop.DeepCopy(volume_property)
                        
                        # Set wireframe mode based on checkbox
                        if self.microenv_wireframe_cb.isChecked():
                            # Use edges to create wireframe effect
                            outline_filter = vtk.vtkOutlineFilter()
                            outline_filter.SetInputData(grid)
                            outline_mapper = vtk.vtkPolyDataMapper()
                            outline_mapper.SetInputConnection(outline_filter.GetOutputPort())
                            outline_actor = vtk.vtkActor()
                            outline_actor.SetMapper(outline_mapper)
                            outline_actor.GetProperty().SetColor(1, 1, 1)  # White wireframe
                            self.renderer.AddActor(outline_actor)
                            self.actors.append(outline_actor)
                            
                            # Also add visible grid lines
                            grid_filter = vtk.vtkRectilinearGridOutlineFilter()
                            grid_filter.SetInputData(grid)
                            grid_mapper = vtk.vtkPolyDataMapper()
                            grid_mapper.SetInputConnection(grid_filter.GetOutputPort())
                            grid_actor = vtk.vtkActor()
                            grid_actor.SetMapper(grid_mapper)
                            grid_actor.GetProperty().SetColor(0.7, 0.7, 0.7)  # Light grey grid
                            self.renderer.AddActor(grid_actor)
                            self.actors.append(grid_actor)
                            
                            # Make volume more transparent in wireframe mode
                            for i in range(otf.GetSize()):
                                val = otf.GetDataPointer()[i*2]
                                opacity = otf.GetValue(val) * 0.3  # Reduce opacity
                                otf.AddPoint(val, opacity)
                        
                        # Create the volume
                        volume = vtk.vtkVolume()
                        volume.SetMapper(mapper)
                        volume.SetProperty(volume_property)
                        
                        # Only add the volume to the renderer if show_microenv_cb is checked
                        if self.show_microenv_cb.isChecked():
                            self.renderer.AddVolume(volume)
                        self.actors.append(volume)
                        
                        # Add color bar for this substrate
                        scalar_bar = vtk.vtkScalarBarActor()
                        scalar_bar.SetLookupTable(ctf)
                        scalar_bar.SetTitle(f"Substrate {substrate_idx}")
                        scalar_bar.SetNumberOfLabels(5)
                        
                        # Position the scalar bar based on the substrate index
                        x_pos = 0.05 + (substrate_idx * 0.15)
                        scalar_bar.SetPosition(x_pos, 0.05)
                        scalar_bar.SetWidth(0.1)
                        scalar_bar.SetHeight(0.3)
                        
                        self.renderer.AddActor2D(scalar_bar)
                        self.actors.append(scalar_bar)
                    
                    # Add info text about the microenvironment
                    text_actor = vtk.vtkTextActor()
                    text_actor.SetInput(f"Microenvironment: {substrate_count} substrates")
                    text_actor.GetTextProperty().SetFontSize(16)
                    text_actor.GetTextProperty().SetColor(1, 1, 0)
                    text_actor.SetPosition(20, 90)
                    self.renderer.AddActor2D(text_actor)
                    self.actors.append(text_actor)
                    
                    # Update slice if it's enabled
                    if self.show_slice_cb.isChecked():
                        self.update_slice()
                    
                    return True
                    
            # If we get here, we couldn't visualize the microenvironment
            if self.debug:
                print("No valid microenvironment data found in the file.")
            return False
            
        except Exception as e:
            if self.debug:
                print(f"Error visualizing microenvironment file: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def update_visualization(self):
        """Update the visualization based on current settings"""
        if self.data_loaded:
            # Remember if we had a slice visible
            slice_was_visible = False
            if self.slice_actor and self.slice_actor.GetVisibility():
                slice_was_visible = True
                
            # Clear previous actors
            self.clear_visualization()
            
            # Reload the current frame with updated settings
            self.load_frame(self.current_frame)
            
            # Update the slice if it was visible
            if slice_was_visible and self.show_slice_cb.isChecked():
                self.update_slice()
    
    def update_slice(self):
        """Update the slice visualization based on current settings"""
        # Clean up existing slice actors
        if self.slice_actor:
            self.renderer.RemoveActor(self.slice_actor)
            self.slice_actor = None
        
        if self.slice_contour_actor:
            self.renderer.RemoveActor(self.slice_contour_actor)
            self.slice_contour_actor = None
            
        # If slice checkbox is unchecked or no data, don't show slice
        if not self.data_loaded or not self.show_slice_cb.isChecked() or not self.microenv_data:
            return
            
        # Create or update the slice plane
        if not self.slice_plane:
            self.slice_plane = vtk.vtkPlane()
            
        # Set plane origin and normal
        self.slice_plane.SetOrigin(
            self.slice_x.value(),
            self.slice_y.value(),
            self.slice_z.value()
        )
        
        # Normalize the orientation vector
        i = self.slice_i.value()
        j = self.slice_j.value()
        k = self.slice_k.value()
        length = (i*i + j*j + k*k)**0.5
        if length > 0:
            i /= length
            j /= length
            k /= length
        self.slice_plane.SetNormal(i, j, k)
        
        # Create the cutter to slice the microenvironment data
        cutter = vtk.vtkCutter()
        cutter.SetCutFunction(self.slice_plane)
        cutter.SetInputData(self.microenv_data)  # Use stored microenv data
        cutter.Update()
        
        # Get the slice output
        slice_output = cutter.GetOutput()
        
        # Create mapper for the slice
        self.slice_mapper = vtk.vtkPolyDataMapper()
        self.slice_mapper.SetInputConnection(cutter.GetOutputPort())
        self.slice_mapper.ScalarVisibilityOn()  # Show the scalars
        
        # Use the color transfer function from the microenvironment if available
        if self.microenv_vol_prop and self.microenv_vol_prop.GetRGBTransferFunction():
            ctf = self.microenv_vol_prop.GetRGBTransferFunction()
            self.slice_mapper.SetLookupTable(ctf)
            
            # Get scalar range from the microenvironment data
            if self.microenv_data.GetPointData() and self.microenv_data.GetPointData().GetScalars():
                scalar_range = self.microenv_data.GetPointData().GetScalars().GetRange()
                self.slice_mapper.SetScalarRange(scalar_range)
        
        # Create actor for the slice
        self.slice_actor = vtk.vtkActor()
        self.slice_actor.SetMapper(self.slice_mapper)
        
        # Make the slice more visible
        self.slice_actor.GetProperty().SetLineWidth(1)
        self.slice_actor.GetProperty().SetPointSize(4)
        self.slice_actor.GetProperty().SetOpacity(1.0)  # Fully opaque
        
        # Add the slice actor to the renderer
        self.renderer.AddActor(self.slice_actor)
        
        # Create a triangulated surface for better visualization
        warp = vtk.vtkWarpScalar()
        warp.SetInputConnection(cutter.GetOutputPort())
        warp.SetScaleFactor(0)  # No actual warping, just for triangulation
        
        triangulate = vtk.vtkTriangleFilter()
        triangulate.SetInputConnection(warp.GetOutputPort())
        
        # Create mapper for the triangulated slice
        self.slice_contour_mapper = vtk.vtkPolyDataMapper()
        self.slice_contour_mapper.SetInputConnection(triangulate.GetOutputPort())
        self.slice_contour_mapper.ScalarVisibilityOn()
        
        # Use the same color mapping as the original slice
        if self.microenv_vol_prop and self.microenv_vol_prop.GetRGBTransferFunction():
            self.slice_contour_mapper.SetLookupTable(self.microenv_vol_prop.GetRGBTransferFunction())
            if self.microenv_data.GetPointData() and self.microenv_data.GetPointData().GetScalars():
                self.slice_contour_mapper.SetScalarRange(self.microenv_data.GetPointData().GetScalars().GetRange())
        
        # Create an actor for the triangulated slice
        self.slice_contour_actor = vtk.vtkActor()
        self.slice_contour_actor.SetMapper(self.slice_contour_mapper)
        self.slice_contour_actor.GetProperty().SetOpacity(0.7)  # Slightly transparent
        
        # Add the triangulated slice actor to the renderer
        self.renderer.AddActor(self.slice_contour_actor)
        
        # Render the scene
        self.render_window.Render()
    
    def slider_changed(self, value):
        """Handle changes to the frame slider"""
        if self.frame_spinbox.value() != value:
            self.frame_spinbox.setValue(value)
            self.load_frame(value)
    
    def spinbox_changed(self, value):
        """Handle changes to the frame spinbox"""
        if self.frame_slider.value() != value:
            self.frame_slider.setValue(value)
            self.load_frame(value)
    
    def previous_frame(self):
        """Go to the previous frame"""
        if self.current_frame > 0:
            self.frame_slider.setValue(self.current_frame - 1)
    
    def next_frame(self):
        """Go to the next frame"""
        if self.current_frame < self.max_frame:
            self.frame_slider.setValue(self.current_frame + 1)
    
    def toggle_play(self):
        """Toggle playback of frames"""
        if self.animation_active:
            self.animation_timer.stop()
            self.animation_active = False
            self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        else:
            self.animation_timer.start(200)  # 200ms between frames
            self.animation_active = True
            self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
    
    def animation_step(self):
        """Advance one frame in the animation"""
        if self.current_frame < self.max_frame:
            self.frame_slider.setValue(self.current_frame + 1)
        else:
            # Loop back to the start
            self.frame_slider.setValue(0)
    
    def on_interaction_start(self, obj, event):
        """Reduce rendering quality during interaction for better performance"""
        # Store original resolution of spheres to restore later
        for actor in self.actors:
            if isinstance(actor, vtk.vtkActor) and actor.GetMapper() and actor.GetMapper().GetInput():
                # For sphere sources, reduce resolution during interaction
                if hasattr(actor, 'original_resolution'):
                    continue  # Already stored
                
                mapper = actor.GetMapper()
                if hasattr(mapper, 'GetInputAlgorithm'):
                    alg = mapper.GetInputAlgorithm()
                    if alg and isinstance(alg, vtk.vtkSphereSource):
                        # Store original resolution
                        actor.original_resolution = (alg.GetPhiResolution(), alg.GetThetaResolution())
                        # Reduce resolution for faster rendering
                        alg.SetPhiResolution(6)
                        alg.SetThetaResolution(6)
                        alg.Update()
        
        # Reduce sample distance for volume rendering during interaction
        for actor in self.actors:
            if isinstance(actor, vtk.vtkVolume) and actor.GetMapper():
                mapper = actor.GetMapper()
                if not hasattr(actor, 'original_sample_distance'):
                    if hasattr(mapper, 'GetSampleDistance'):
                        actor.original_sample_distance = mapper.GetSampleDistance()
                        # Double the sample distance during interaction (less quality, faster rendering)
                        mapper.SetSampleDistance(actor.original_sample_distance * 2.0)
        
        # Force render window to use faster algorithms during interaction
        self.render_window.SetDesiredUpdateRate(30.0)
    
    def on_interaction_end(self, obj, event):
        """Restore full rendering quality after interaction is complete"""
        # Restore original sphere resolution
        for actor in self.actors:
            if isinstance(actor, vtk.vtkActor) and actor.GetMapper() and hasattr(actor, 'original_resolution'):
                mapper = actor.GetMapper()
                if hasattr(mapper, 'GetInputAlgorithm'):
                    alg = mapper.GetInputAlgorithm()
                    if alg and isinstance(alg, vtk.vtkSphereSource):
                        # Restore original resolution
                        phi, theta = actor.original_resolution
                        alg.SetPhiResolution(phi)
                        alg.SetThetaResolution(theta)
                        alg.Update()
        
        # Restore original sample distance for volume rendering
        for actor in self.actors:
            if isinstance(actor, vtk.vtkVolume) and actor.GetMapper() and hasattr(actor, 'original_sample_distance'):
                mapper = actor.GetMapper()
                if hasattr(mapper, 'SetSampleDistance'):
                    mapper.SetSampleDistance(actor.original_sample_distance)
        
        # Return to high quality rendering for still images
        self.render_window.SetDesiredUpdateRate(0.0001)
        
        # Force a high-quality render
        self.render_window.Render()
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Clean up VTK objects
        self.animation_timer.stop()
        self.vtk_widget.GetRenderWindow().Finalize()
        event.accept()

def main():
    """Main entry point for the application"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='PhysiCell VTK Viewer')
    parser.add_argument('-f', '--folder', 
                       help='Initial folder for loading PhysiCell output',
                       default=None)
    parser.add_argument('-debug', '--debug', 
                       help='Enable debug output',
                       action='store_true')
    args = parser.parse_args()
    
    # Set global debug flag
    global DEBUG
    DEBUG = args.debug
    
    if DEBUG:
        print("Debug mode enabled")
        if pyMCDS_cells is None:
            print("Warning: pyMCDS_cells module not found. XML visualization will be limited.")
    
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    # Pass the initial directory and debug flag to the viewer
    window = PhysiCellVTKQtViewer(initial_dir=args.folder, debug=args.debug)
    window.show()
    
    # If a folder was specified and exists, automatically load it
    if args.folder and os.path.isdir(args.folder):
        window.output_dir = args.folder
        window.directory_label.setText(args.folder)
        window.find_max_frame()
        if window.max_frame >= 0:
            window.frame_slider.setMaximum(window.max_frame)
            window.frame_spinbox.setMaximum(window.max_frame)
            window.frame_slider.setEnabled(True)
            window.frame_spinbox.setEnabled(True)
            window.prev_btn.setEnabled(True)
            window.next_btn.setEnabled(True)
            window.play_btn.setEnabled(True)
            window.microenv_wireframe_cb.setEnabled(True)
            window.show_mat_cb.setEnabled(True)
            window.show_microenv_cb.setEnabled(True)
            window.cell_opacity_slider.setEnabled(True)
            window.microenv_opacity_slider.setEnabled(True)
            window.show_slice_cb.setEnabled(True)
            window.slice_x.setEnabled(True)
            window.slice_y.setEnabled(True)
            window.slice_z.setEnabled(True)
            window.slice_i.setEnabled(True)
            window.slice_j.setEnabled(True)
            window.slice_k.setEnabled(True)
            window.current_frame = 0
            window.load_frame(window.current_frame)
            window.data_loaded = True
            window.status_bar.showMessage(f"Loaded directory: {args.folder}, {window.max_frame+1} frames found")
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
