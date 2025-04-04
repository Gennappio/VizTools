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
    
    def __init__(self):
        """Initialize the application window and UI components"""
        super().__init__()
        
        # Application state
        self.output_dir = ""
        self.current_frame = 0
        self.max_frame = 0
        self.actors = []
        self.data_loaded = False
        
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
        
        self.show_mat_cb = QCheckBox("Show .mat Data")
        self.show_mat_cb.setChecked(True)
        self.show_mat_cb.setEnabled(False)
        self.show_mat_cb.stateChanged.connect(self.update_visualization)
        display_layout.addWidget(self.show_mat_cb)
        
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
            os.path.expanduser("~"),
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
                self.cell_opacity_slider.setEnabled(True)
                
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
                    
        # Check for .mat files
        for filename in os.listdir(self.output_dir):
            if filename.startswith("output") and filename.endswith("_cells.mat"):
                try:
                    frame_num = int(filename[6:-10])
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
        mat_file = os.path.join(self.output_dir, f"output{frame_number:08d}_cells.mat")
        
        # Try to visualize the XML file if it exists and cells should be shown
        if os.path.exists(xml_file) and self.show_cells_cb.isChecked():
            self.visualize_cells_from_xml(xml_file)
        
        # Try to visualize the .mat file if it exists and .mat data should be shown
        if os.path.exists(mat_file) and self.show_mat_cb.isChecked():
            self.visualize_mat_file(mat_file)
        
        # Reset the camera if this is the first time loading a frame
        if not self.data_loaded:
            self.renderer.ResetCamera()
            
        # Update the render window
        self.render_window.Render()
        
        # Update info label
        info_text = f"Frame: {frame_number}"
        if os.path.exists(xml_file):
            info_text += f"\nXML: {os.path.basename(xml_file)}"
        if os.path.exists(mat_file):
            info_text += f"\nMAT: {os.path.basename(mat_file)}"
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
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    window = PhysiCellVTKQtViewer()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
