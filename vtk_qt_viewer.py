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
    QGroupBox, QRadioButton, QCheckBox, QSpinBox, QStyle
)
from PyQt5.QtCore import Qt, QTimer
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

try:
    # Try to import pyMCDS if available (for loading PhysiCell data)
    from pyMCDS_cells import pyMCDS_cells
except ImportError:
    print("Warning: pyMCDS_cells module not found. XML visualization will be limited.")
    pyMCDS_cells = None

class PhysiCellVTKQtViewer(QMainWindow):
    """Qt-based GUI application for PhysiCell VTK visualization"""
    
    def __init__(self, initial_dir=None):
        """Initialize the application window and UI components"""
        super().__init__()
        
        # Application state
        self.output_dir = ""
        self.current_frame = 0
        self.max_frame = 0
        self.actors = []
        self.data_loaded = False
        self.initial_dir = initial_dir or os.path.expanduser("~")
        
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
        
        self.show_cells_cb = QCheckBox("Show Cells")
        self.show_cells_cb.setChecked(True)
        self.show_cells_cb.setEnabled(False)
        self.show_cells_cb.stateChanged.connect(self.update_visualization)
        display_layout.addWidget(self.show_cells_cb)
        
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
        
        # Create a renderer
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.1, 0.2, 0.3)  # Dark blue background
        self.render_window.AddRenderer(self.renderer)
        
        # Set up the interactor
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        
        # Set interactor style for smooth interaction
        interactor_style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(interactor_style)
        
        # Add orientation axes
        self.add_orientation_axes()
        
        # Initialize the interactor
        self.interactor.Initialize()
        
        # Set up an animation timer
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.animation_step)
        self.animation_active = False
    
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
        for actor in self.actors:
            self.renderer.RemoveActor(actor)
        self.actors = []
    
    def load_directory(self):
        """Open a file dialog to select a PhysiCell output directory"""
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
                self.show_cells_cb.setEnabled(True)
                self.show_mat_cb.setEnabled(True)
                self.show_microenv_cb.setEnabled(True)
                self.cell_opacity_slider.setEnabled(True)
                self.microenv_opacity_slider.setEnabled(True)
                
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
        
        # Try to visualize the XML file if it exists and cells should be shown
        if os.path.exists(xml_file) and self.show_cells_cb.isChecked():
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
            
            # Calculate radii
            cell_vols = cell_df['total_volume'].values
            four_thirds_pi = 4.188790204786391
            cell_radii = np.divide(cell_vols, four_thirds_pi)
            cell_radii = np.power(cell_radii, 0.333333333333333333333333)
            
            # Get cell types
            cell_types = cell_df['cell_type'].values.astype(int)
            
            # Create points for cell centers
            points = vtk.vtkPoints()
            num_cells = len(positions_x)
            
            # Create arrays for colors and sizes
            colors = vtk.vtkUnsignedCharArray()
            colors.SetNumberOfComponents(3)
            colors.SetName("Colors")
            
            sizes = vtk.vtkFloatArray()
            sizes.SetNumberOfComponents(1)
            sizes.SetName("Sizes")
            
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
            
            # Add points and data
            for i in range(num_cells):
                points.InsertNextPoint(positions_x[i], positions_y[i], positions_z[i])
                
                # Set color based on cell type
                ct = cell_types[i]
                color = cell_type_colors.get(ct, (180, 180, 180))
                colors.InsertNextTuple3(*color)
                sizes.InsertNextValue(cell_radii[i])
            
            # Create a polydata object
            polydata = vtk.vtkPolyData()
            polydata.SetPoints(points)
            
            # Add the color and size data to the points
            polydata.GetPointData().SetScalars(colors)
            polydata.GetPointData().AddArray(sizes)
            
            # Create the sphere source for glyphing
            sphere = vtk.vtkSphereSource()
            sphere.SetPhiResolution(8)
            sphere.SetThetaResolution(8)
            sphere.SetRadius(1.0)
            
            # Create the glyph
            glyph = vtk.vtkGlyph3D()
            glyph.SetSourceConnection(sphere.GetOutputPort())
            glyph.SetInputData(polydata)
            glyph.SetScaleModeToScaleByScalar()
            glyph.SetScaleFactor(1.0)
            glyph.SetColorModeToColorByScalar()
            glyph.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "Sizes")
            glyph.SetInputArrayToProcess(1, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "Colors")
            
            # Create mapper and actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(glyph.GetOutputPort())
            
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
            
            return mat_contents
        
        except Exception as e:
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
        
        # Create points for visualization
        points = vtk.vtkPoints()
        for i in range(n_points):
            points.InsertNextPoint(x[i], y[i], z[i])
        
        # Create a polydata for points
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        
        # Add scalar values
        scalars = vtk.vtkFloatArray()
        scalars.SetNumberOfComponents(1)
        scalars.SetName("Values")
        for value in values:
            scalars.InsertNextValue(value)
        polydata.GetPointData().SetScalars(scalars)
        
        # Create spheres for each point
        sphere = vtk.vtkSphereSource()
        sphere.SetPhiResolution(8)
        sphere.SetThetaResolution(8)
        sphere.SetRadius(1.0)
        
        glyph = vtk.vtkGlyph3D()
        glyph.SetInputData(polydata)
        glyph.SetSourceConnection(sphere.GetOutputPort())
        glyph.SetScaleModeToScaleByScalar()
        glyph.SetScaleFactor(3.0)
        glyph.Update()
        
        # Create a tube connecting the points
        points_for_line = vtk.vtkPoints()
        for i in range(n_points):
            points_for_line.InsertNextPoint(x[i], y[i], z[i])
        
        lines = vtk.vtkCellArray()
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(n_points)
        for i in range(n_points):
            line.GetPointIds().SetId(i, i)
        lines.InsertNextCell(line)
        
        line_polydata = vtk.vtkPolyData()
        line_polydata.SetPoints(points_for_line)
        line_polydata.SetLines(lines)
        
        # Add scalar values to the line
        line_scalars = vtk.vtkFloatArray()
        line_scalars.SetNumberOfComponents(1)
        line_scalars.SetName("LineValues")
        for value in values:
            line_scalars.InsertNextValue(value)
        line_polydata.GetPointData().SetScalars(line_scalars)
        
        # Create a tube
        tube_filter = vtk.vtkTubeFilter()
        tube_filter.SetInputData(line_polydata)
        tube_filter.SetRadius(2.0)
        tube_filter.SetNumberOfSides(8)
        tube_filter.SetVaryRadiusToVaryRadiusByScalar()
        tube_filter.Update()
        
        # Create mappers
        points_mapper = vtk.vtkPolyDataMapper()
        points_mapper.SetInputConnection(glyph.GetOutputPort())
        points_mapper.SetScalarRange(min_val, max_val)
        
        tube_mapper = vtk.vtkPolyDataMapper()
        tube_mapper.SetInputConnection(tube_filter.GetOutputPort())
        tube_mapper.SetScalarRange(min_val, max_val)
        
        # Create actors
        points_actor = vtk.vtkActor()
        points_actor.SetMapper(points_mapper)
        
        tube_actor = vtk.vtkActor()
        tube_actor.SetMapper(tube_mapper)
        
        # Add a color bar
        scalar_bar = vtk.vtkScalarBarActor()
        scalar_bar.SetLookupTable(points_mapper.GetLookupTable())
        scalar_bar.SetTitle("Cell Values")
        scalar_bar.SetNumberOfLabels(5)
        scalar_bar.SetPosition(0.8, 0.1)
        scalar_bar.SetWidth(0.1)
        scalar_bar.SetHeight(0.8)
        
        self.renderer.AddActor2D(scalar_bar)
        self.actors.append(scalar_bar)
        
        return [points_actor, tube_actor]
    
    def visualize_mat_file(self, file_path):
        """Visualize data from a .mat file"""
        try:
            mat_contents = self.load_mat_file(file_path)
            
            if mat_contents is None:
                return False
                
            # Check for cells data
            if 'cells' in mat_contents:
                cells_data = mat_contents['cells']
                
                # Handle scalar values case (most common for PhysiCell outputs)
                if cells_data.shape[1] == 1 and isinstance(cells_data[0, 0], (float, int, np.float64, np.int64)):
                    values = cells_data.flatten()
                    
                    # Create a spiral visualization
                    actors = self.create_spiral_viz_from_values(values)
                    
                    # Add actors to the renderer
                    for actor in actors:
                        self.renderer.AddActor(actor)
                        self.actors.append(actor)
                    
                    # Add info text
                    min_val = values.min()
                    max_val = values.max()
                    text_actor = vtk.vtkTextActor()
                    text_actor.SetInput(f"Cell data: {len(values)} values, range [{min_val:.2f}, {max_val:.2f}]")
                    text_actor.GetTextProperty().SetFontSize(16)
                    text_actor.GetTextProperty().SetColor(1, 1, 0)
                    text_actor.SetPosition(20, 30)
                    self.renderer.AddActor2D(text_actor)
                    self.actors.append(text_actor)
                    
                    return True
                elif cells_data.shape[1] >= 4:  # Matrix with at least 4 columns (x,y,z,value)
                    x_data = cells_data[:, 0]
                    y_data = cells_data[:, 1]
                    z_data = cells_data[:, 2]
                    scalar_data = cells_data[:, 3]
                    
                    # Create points for visualization
                    points = vtk.vtkPoints()
                    for i in range(len(x_data)):
                        points.InsertNextPoint(x_data[i], y_data[i], z_data[i])
                    
                    # Create a polydata
                    polydata = vtk.vtkPolyData()
                    polydata.SetPoints(points)
                    
                    # Add scalar values
                    scalars = vtk.vtkFloatArray()
                    scalars.SetNumberOfComponents(1)
                    scalars.SetName("Values")
                    for value in scalar_data:
                        scalars.InsertNextValue(value)
                    polydata.GetPointData().SetScalars(scalars)
                    
                    # Create a delaunay 3D filter
                    delaunay = vtk.vtkDelaunay3D()
                    delaunay.SetInputData(polydata)
                    delaunay.SetTolerance(0.01)
                    delaunay.Update()
                    
                    # Create a mapper
                    mapper = vtk.vtkDataSetMapper()
                    mapper.SetInputConnection(delaunay.GetOutputPort())
                    mapper.SetScalarRange(scalar_data.min(), scalar_data.max())
                    
                    # Create an actor
                    actor = vtk.vtkActor()
                    actor.SetMapper(mapper)
                    actor.GetProperty().SetOpacity(0.7)
                    
                    # Add the actor
                    self.renderer.AddActor(actor)
                    self.actors.append(actor)
                    
                    # Add a color bar
                    scalar_bar = vtk.vtkScalarBarActor()
                    scalar_bar.SetLookupTable(mapper.GetLookupTable())
                    scalar_bar.SetTitle("Values")
                    scalar_bar.SetPosition(0.8, 0.1)
                    scalar_bar.SetWidth(0.1)
                    scalar_bar.SetHeight(0.8)
                    self.renderer.AddActor2D(scalar_bar)
                    self.actors.append(scalar_bar)
                    
                    return True
            
            # If we get here, we couldn't visualize the .mat file
            return False
            
        except Exception as e:
            print(f"Error visualizing .mat file: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def visualize_microenvironment(self, file_path):
        """Visualize microenvironment data from a .mat file"""
        try:
            mat_contents = self.load_mat_file(file_path)
            
            if mat_contents is None:
                return False
            
            # Debug output to help diagnose issues
            print(f"Microenvironment file keys: {list(mat_contents.keys())}")
            
            # Look for substrate data - PhysiCell format
            if 'multiscale_microenvironment' in mat_contents:
                microenv_data = mat_contents['multiscale_microenvironment']
                
                # Print structure information for debugging
                print(f"Microenvironment data shape: {microenv_data.shape}")
                
                # The data is typically stored with substrates in rows and positions in columns
                # First several rows contain position info, then substrate values
                if microenv_data.shape[0] > 4:  # At least one substrate
                    # First 3 rows are x,y,z coordinates
                    x = microenv_data[0, :].flatten()  # x coordinates
                    y = microenv_data[1, :].flatten()  # y coordinates
                    z = microenv_data[2, :].flatten()  # z coordinates
                    
                    # Print coordinate information for debugging
                    print(f"X range: {x.min()} to {x.max()}, shape: {x.shape}")
                    print(f"Y range: {y.min()} to {y.max()}, shape: {y.shape}")
                    print(f"Z range: {z.min()} to {z.max()}, shape: {z.shape}")
                    
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
                    
                    # Process each substrate
                    for substrate_idx in range(substrate_count):
                        # Get the substrate data (row 4+idx in the microenvironment matrix)
                        substrate_data = microenv_data[4 + substrate_idx, :]
                        
                        # Print substrate range for debugging
                        print(f"Substrate {substrate_idx} range: {substrate_data.min()} to {substrate_data.max()}")
                        
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
                        
                        # Create the volume
                        volume = vtk.vtkVolume()
                        volume.SetMapper(mapper)
                        volume.SetProperty(volume_property)
                        
                        # Add the volume to the renderer
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
                    
                    return True
                    
            # If we get here, we couldn't visualize the microenvironment
            print("No valid microenvironment data found in the file.")
            return False
            
        except Exception as e:
            print(f"Error visualizing microenvironment file: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def update_visualization(self):
        """Update the visualization based on current settings"""
        if self.data_loaded:
            self.load_frame(self.current_frame)
    
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
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    # Pass the initial directory to the viewer
    window = PhysiCellVTKQtViewer(initial_dir=args.folder)
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
            window.show_cells_cb.setEnabled(True)
            window.show_mat_cb.setEnabled(True)
            window.show_microenv_cb.setEnabled(True)
            window.cell_opacity_slider.setEnabled(True)
            window.microenv_opacity_slider.setEnabled(True)
            window.current_frame = 0
            window.load_frame(window.current_frame)
            window.data_loaded = True
            window.status_bar.showMessage(f"Loaded directory: {args.folder}, {window.max_frame+1} frames found")
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
