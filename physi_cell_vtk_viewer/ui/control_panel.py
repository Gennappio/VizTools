"""
Control panel for PhysiCell viewer
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel, 
    QGroupBox, QCheckBox, QSpinBox, QDoubleSpinBox, QStyle, QFileDialog,
    QScrollArea, QSizePolicy, QFrame, QComboBox
)
from PyQt5.QtCore import Qt, pyqtSignal

from physi_cell_vtk_viewer.utils.variable_reader import (
    get_microenv_variables, get_cell_variables
)

class ControlPanel(QWidget):
    """Control panel for PhysiCell VTK visualization"""
    
    # Define signals for user interactions
    directory_loaded = pyqtSignal(str)
    frame_changed = pyqtSignal(int)
    display_options_changed = pyqtSignal(dict)
    slice_options_changed = pyqtSignal(dict)
    apply_changes = pyqtSignal()
    play_toggled = pyqtSignal(bool)
    variables_selected = pyqtSignal(dict)  # Signal for variable selection
    
    def __init__(self, parent=None):
        """Initialize the control panel"""
        super().__init__(parent)
        
        # Create scrollable panel
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # Create widget for scroll area
        self.scroll_widget = QWidget()
        self.layout = QVBoxLayout(self.scroll_widget)
        self.scroll_area.setWidget(self.scroll_widget)
        
        # Set up layout for the control panel
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.scroll_area)
        
        # Create controls
        self.create_file_group()
        self.create_nav_group()
        self.create_display_group()
        self.create_slice_group()
        self.create_apply_button()
        self.create_info_group()
        
        # Add a spacer to push everything up
        self.layout.addStretch()
    
    def create_file_group(self):
        """Create the file loading controls"""
        file_group = QGroupBox("Data Loading")
        file_layout = QVBoxLayout(file_group)
        
        self.load_btn = QPushButton("Load Directory")
        self.load_btn.clicked.connect(self.load_directory)
        file_layout.addWidget(self.load_btn)
        
        self.directory_label = QLabel("No directory loaded")
        self.directory_label.setWordWrap(True)
        file_layout.addWidget(self.directory_label)
        
        self.layout.addWidget(file_group)
    
    def create_nav_group(self):
        """Create frame navigation controls"""
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
        
        self.layout.addWidget(nav_group)
    
    def create_display_group(self):
        """Create display options controls"""
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout(display_group)
        
        # Variable selection group
        var_group = QGroupBox("Variable Selection")
        var_layout = QVBoxLayout(var_group)
        
        # Microenvironment variable selection
        var_layout.addWidget(QLabel("Microenvironment Variable:"))
        self.microenv_var_combo = QComboBox()
        self.microenv_var_combo.setEnabled(False)
        self.microenv_var_combo.currentIndexChanged.connect(self.on_microenv_var_changed)
        var_layout.addWidget(self.microenv_var_combo)
        
        # Cell variable selection
        var_layout.addWidget(QLabel("Cell Variable:"))
        self.cell_var_combo = QComboBox()
        self.cell_var_combo.setEnabled(False)
        self.cell_var_combo.currentIndexChanged.connect(self.on_cell_var_changed)
        var_layout.addWidget(self.cell_var_combo)
        
        # Cell coloring variable selection
        var_layout.addWidget(QLabel("Color Cells By:"))
        self.cell_color_combo = QComboBox()
        self.cell_color_combo.setEnabled(False)
        self.cell_color_combo.currentIndexChanged.connect(self.on_cell_color_var_changed)
        var_layout.addWidget(self.cell_color_combo)
        
        display_layout.addWidget(var_group)
        
        # Add a separator line
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        display_layout.addWidget(line)
        
        # Display mode options
        self.cell_as_spheres_cb = QCheckBox("Display Cells as Spheres")
        self.cell_as_spheres_cb.setChecked(True)
        self.cell_as_spheres_cb.setEnabled(False)
        self.cell_as_spheres_cb.stateChanged.connect(self.on_display_option_changed)
        display_layout.addWidget(self.cell_as_spheres_cb)
        
        self.microenv_wireframe_cb = QCheckBox("Microenvironment Wireframe")
        self.microenv_wireframe_cb.setChecked(False)
        self.microenv_wireframe_cb.setEnabled(False)
        self.microenv_wireframe_cb.stateChanged.connect(self.on_display_option_changed)
        display_layout.addWidget(self.microenv_wireframe_cb)
        
        self.show_mat_cb = QCheckBox("Show Cell Data")
        self.show_mat_cb.setChecked(True)
        self.show_mat_cb.setEnabled(False)
        self.show_mat_cb.stateChanged.connect(self.on_display_option_changed)
        display_layout.addWidget(self.show_mat_cb)
        
        self.show_microenv_cb = QCheckBox("Show Microenvironment")
        self.show_microenv_cb.setChecked(True)
        self.show_microenv_cb.setEnabled(False)
        self.show_microenv_cb.stateChanged.connect(self.on_display_option_changed)
        display_layout.addWidget(self.show_microenv_cb)
        
        # Microenvironment color range controls (Volume Rendering)
        color_range_group = QGroupBox("Microenvironment Volume Color Range")
        color_range_layout = QVBoxLayout(color_range_group)
        
        # Auto range checkbox
        self.auto_range_cb = QCheckBox("Auto Range")
        self.auto_range_cb.setChecked(True)
        self.auto_range_cb.setEnabled(False)
        self.auto_range_cb.stateChanged.connect(self.toggle_color_range_inputs)
        self.auto_range_cb.stateChanged.connect(self.on_display_option_changed)
        color_range_layout.addWidget(self.auto_range_cb)
        
        # Min/Max value inputs
        range_layout = QHBoxLayout()
        
        # Min value input
        min_layout = QVBoxLayout()
        min_layout.addWidget(QLabel("Min:"))
        self.min_range_input = QDoubleSpinBox()
        self.min_range_input.setRange(-1000000, 1000000)
        self.min_range_input.setDecimals(6)
        self.min_range_input.setValue(0.0)
        self.min_range_input.setEnabled(False)
        self.min_range_input.valueChanged.connect(self.on_display_option_changed)
        min_layout.addWidget(self.min_range_input)
        range_layout.addLayout(min_layout)
        
        # Max value input
        max_layout = QVBoxLayout()
        max_layout.addWidget(QLabel("Max:"))
        self.max_range_input = QDoubleSpinBox()
        self.max_range_input.setRange(-1000000, 1000000)
        self.max_range_input.setDecimals(6)
        self.max_range_input.setValue(1.0)
        self.max_range_input.setEnabled(False)
        self.max_range_input.valueChanged.connect(self.on_display_option_changed)
        max_layout.addWidget(self.max_range_input)
        range_layout.addLayout(max_layout)
        
        color_range_layout.addLayout(range_layout)
        display_layout.addWidget(color_range_group)
        
        # Cell variable color range controls
        cell_color_range_group = QGroupBox("Cell Variable Color Range")
        cell_color_range_layout = QVBoxLayout(cell_color_range_group)
        
        # Auto range checkbox for cells
        self.cell_auto_range_cb = QCheckBox("Auto Range")
        self.cell_auto_range_cb.setChecked(True)
        self.cell_auto_range_cb.setEnabled(False)
        self.cell_auto_range_cb.stateChanged.connect(self.toggle_cell_color_range_inputs)
        self.cell_auto_range_cb.stateChanged.connect(self.on_display_option_changed)
        cell_color_range_layout.addWidget(self.cell_auto_range_cb)
        
        # Min/Max value inputs for cells
        cell_range_layout = QHBoxLayout()
        
        # Min value input for cells
        cell_min_layout = QVBoxLayout()
        cell_min_layout.addWidget(QLabel("Min:"))
        self.cell_min_range_input = QDoubleSpinBox()
        self.cell_min_range_input.setRange(-1000000, 1000000)
        self.cell_min_range_input.setDecimals(6)
        self.cell_min_range_input.setValue(0.0)
        self.cell_min_range_input.setEnabled(False)
        self.cell_min_range_input.valueChanged.connect(self.on_display_option_changed)
        cell_min_layout.addWidget(self.cell_min_range_input)
        cell_range_layout.addLayout(cell_min_layout)
        
        # Max value input for cells
        cell_max_layout = QVBoxLayout()
        cell_max_layout.addWidget(QLabel("Max:"))
        self.cell_max_range_input = QDoubleSpinBox()
        self.cell_max_range_input.setRange(-1000000, 1000000)
        self.cell_max_range_input.setDecimals(6)
        self.cell_max_range_input.setValue(1.0)
        self.cell_max_range_input.setEnabled(False)
        self.cell_max_range_input.valueChanged.connect(self.on_display_option_changed)
        cell_max_layout.addWidget(self.cell_max_range_input)
        cell_range_layout.addLayout(cell_max_layout)
        
        cell_color_range_layout.addLayout(cell_range_layout)
        display_layout.addWidget(cell_color_range_group)
        
        # Microenvironment slice color range controls
        slice_color_range_group = QGroupBox("Slice Color Range")
        slice_color_range_layout = QVBoxLayout(slice_color_range_group)
        
        # Auto range checkbox for slice
        self.slice_auto_range_cb = QCheckBox("Auto Range")
        self.slice_auto_range_cb.setChecked(True)
        self.slice_auto_range_cb.setEnabled(False)
        self.slice_auto_range_cb.stateChanged.connect(self.toggle_slice_color_range_inputs)
        self.slice_auto_range_cb.stateChanged.connect(self.on_display_option_changed)
        slice_color_range_layout.addWidget(self.slice_auto_range_cb)
        
        # Min/Max value inputs for slice
        slice_range_layout = QHBoxLayout()
        
        # Min value input for slice
        slice_min_layout = QVBoxLayout()
        slice_min_layout.addWidget(QLabel("Min:"))
        self.slice_min_range_input = QDoubleSpinBox()
        self.slice_min_range_input.setRange(-1000000, 1000000)
        self.slice_min_range_input.setDecimals(6)
        self.slice_min_range_input.setValue(0.0)
        self.slice_min_range_input.setEnabled(False)
        self.slice_min_range_input.valueChanged.connect(self.on_display_option_changed)
        slice_min_layout.addWidget(self.slice_min_range_input)
        slice_range_layout.addLayout(slice_min_layout)
        
        # Max value input for slice
        slice_max_layout = QVBoxLayout()
        slice_max_layout.addWidget(QLabel("Max:"))
        self.slice_max_range_input = QDoubleSpinBox()
        self.slice_max_range_input.setRange(-1000000, 1000000)
        self.slice_max_range_input.setDecimals(6)
        self.slice_max_range_input.setValue(1.0)
        self.slice_max_range_input.setEnabled(False)
        self.slice_max_range_input.valueChanged.connect(self.on_display_option_changed)
        slice_max_layout.addWidget(self.slice_max_range_input)
        slice_range_layout.addLayout(slice_max_layout)
        
        slice_color_range_layout.addLayout(slice_range_layout)
        display_layout.addWidget(slice_color_range_group)
        
        # Opacity sliders
        self.microenv_opacity_slider = QSlider(Qt.Horizontal)
        self.microenv_opacity_slider.setRange(0, 100)
        self.microenv_opacity_slider.setValue(50)
        self.microenv_opacity_slider.setEnabled(False)
        self.microenv_opacity_slider.valueChanged.connect(self.on_display_option_changed)
        display_layout.addWidget(QLabel("Microenvironment Opacity:"))
        display_layout.addWidget(self.microenv_opacity_slider)
        
        self.cell_opacity_slider = QSlider(Qt.Horizontal)
        self.cell_opacity_slider.setRange(0, 100)
        self.cell_opacity_slider.setValue(70)
        self.cell_opacity_slider.setEnabled(False)
        self.cell_opacity_slider.valueChanged.connect(self.on_display_option_changed)
        display_layout.addWidget(QLabel("Cell Opacity:"))
        display_layout.addWidget(self.cell_opacity_slider)
        
        self.layout.addWidget(display_group)
    
    def create_slice_group(self):
        """Create slice controls"""
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
        self.slice_x.valueChanged.connect(self.on_slice_option_changed)
        x_layout.addWidget(self.slice_x)
        pos_layout.addLayout(x_layout)
        
        # Y position
        y_layout = QHBoxLayout()
        y_layout.addWidget(QLabel("Y:"))
        self.slice_y = QSpinBox()
        self.slice_y.setRange(-1000, 1000)
        self.slice_y.setValue(0)
        self.slice_y.valueChanged.connect(self.on_slice_option_changed)
        y_layout.addWidget(self.slice_y)
        pos_layout.addLayout(y_layout)
        
        # Z position
        z_layout = QHBoxLayout()
        z_layout.addWidget(QLabel("Z:"))
        self.slice_z = QSpinBox()
        self.slice_z.setRange(-1000, 1000)
        self.slice_z.setValue(0)
        self.slice_z.valueChanged.connect(self.on_slice_option_changed)
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
        self.slice_i.valueChanged.connect(self.on_slice_option_changed)
        i_layout.addWidget(self.slice_i)
        orient_layout.addLayout(i_layout)
        
        # J orientation
        j_layout = QHBoxLayout()
        j_layout.addWidget(QLabel("J:"))
        self.slice_j = QDoubleSpinBox()
        self.slice_j.setRange(-1.0, 1.0)
        self.slice_j.setSingleStep(0.1)
        self.slice_j.setValue(0.0)
        self.slice_j.valueChanged.connect(self.on_slice_option_changed)
        j_layout.addWidget(self.slice_j)
        orient_layout.addLayout(j_layout)
        
        # K orientation
        k_layout = QHBoxLayout()
        k_layout.addWidget(QLabel("K:"))
        self.slice_k = QDoubleSpinBox()
        self.slice_k.setRange(-1.0, 1.0)
        self.slice_k.setSingleStep(0.1)
        self.slice_k.setValue(1.0)
        self.slice_k.valueChanged.connect(self.on_slice_option_changed)
        k_layout.addWidget(self.slice_k)
        orient_layout.addLayout(k_layout)
        
        slice_layout.addWidget(orient_group)
        
        # Add Contour Value input
        contour_group = QGroupBox("Slice Contour")
        contour_layout = QVBoxLayout(contour_group)
        
        # Contour value input
        contour_value_layout = QHBoxLayout()
        contour_value_layout.addWidget(QLabel("Contour Value:"))
        self.slice_contour_value = QDoubleSpinBox()
        self.slice_contour_value.setRange(-1000000, 1000000)
        self.slice_contour_value.setDecimals(6)
        self.slice_contour_value.setValue(0.5)  # Default contour value
        self.slice_contour_value.setEnabled(False)
        self.slice_contour_value.valueChanged.connect(self.on_slice_option_changed)
        contour_value_layout.addWidget(self.slice_contour_value)
        contour_layout.addLayout(contour_value_layout)
        
        # Show contour checkbox
        self.show_contour_cb = QCheckBox("Show Contour Line")
        self.show_contour_cb.setChecked(False)
        self.show_contour_cb.setEnabled(False)
        self.show_contour_cb.stateChanged.connect(self.on_slice_option_changed)
        contour_layout.addWidget(self.show_contour_cb)
        
        slice_layout.addWidget(contour_group)
        
        # Show slice checkbox
        self.show_slice_cb = QCheckBox("Show Slice")
        self.show_slice_cb.setChecked(False)
        self.show_slice_cb.setEnabled(False)
        self.show_slice_cb.stateChanged.connect(self.on_slice_option_changed)
        slice_layout.addWidget(self.show_slice_cb)
        
        self.layout.addWidget(slice_group)
    
    def create_apply_button(self):
        """Create Apply button"""
        self.apply_btn = QPushButton("Apply Changes")
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self.apply_changes_handler)
        self.apply_btn.setStyleSheet("font-weight: bold; background-color: #4CAF50; color: white;")
        self.apply_btn.setMinimumHeight(40)
        self.layout.addWidget(self.apply_btn)
    
    def create_info_group(self):
        """Create info panel"""
        info_group = QGroupBox("Information")
        info_layout = QVBoxLayout(info_group)
        
        self.info_label = QLabel("Load a directory to begin visualization.")
        self.info_label.setWordWrap(True)
        info_layout.addWidget(self.info_label)
        
        self.layout.addWidget(info_group)
    
    def apply_changes_handler(self):
        """Handle Apply button click - emit signals for display and slice options"""
        # Emit display options
        self.display_options_changed_handler()
        
        # Emit slice options
        self.slice_options_changed_handler()
        
        # Reset styling
        self.frame_info_label.setStyleSheet("")
        self.apply_btn.setStyleSheet("font-weight: bold; background-color: #4CAF50; color: white;")
        
        # Emit apply signal
        self.apply_changes.emit()
    
    def load_directory(self):
        """Handle load directory button click"""
        directory = QFileDialog.getExistingDirectory(
            self, "Select PhysiCell Output Directory", 
            "~",
            QFileDialog.ShowDirsOnly
        )
        
        if directory:
            self.directory_label.setText(directory)
            self.directory_loaded.emit(directory)
    
    def slider_changed(self, value):
        """Handle frame slider value change"""
        # Update spinbox without triggering its own signal
        self.frame_spinbox.blockSignals(True)
        self.frame_spinbox.setValue(value)
        self.frame_spinbox.blockSignals(False)
        
        # Update frame info label - indicate pending change
        self.frame_info_label.setText(f"Frame: {value} (pending)")
        self.frame_info_label.setStyleSheet("color: #FF5722; font-weight: bold;")
        
        # Highlight apply button to indicate pending changes
        self.apply_btn.setStyleSheet("font-weight: bold; background-color: #FF5722; color: white;")
        
        # Emit signal
        self.frame_changed.emit(value)
    
    def spinbox_changed(self, value):
        """Handle frame spinbox value change"""
        # Update slider without triggering its own signal
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(value)
        self.frame_slider.blockSignals(False)
        
        # Update frame info label - indicate pending change
        self.frame_info_label.setText(f"Frame: {value} (pending)")
        self.frame_info_label.setStyleSheet("color: #FF5722; font-weight: bold;")
        
        # Highlight apply button to indicate pending changes
        self.apply_btn.setStyleSheet("font-weight: bold; background-color: #FF5722; color: white;")
        
        # Emit signal
        self.frame_changed.emit(value)
    
    def previous_frame(self):
        """Handle previous frame button click"""
        current = self.frame_slider.value()
        if current > 0:
            self.frame_slider.setValue(current - 1)
    
    def next_frame(self):
        """Handle next frame button click"""
        current = self.frame_slider.value()
        if current < self.frame_slider.maximum():
            self.frame_slider.setValue(current + 1)
    
    def toggle_play(self):
        """Handle play/pause button click"""
        # Toggle the play state and update the icon
        if self.play_btn.icon().cacheKey() == self.style().standardIcon(QStyle.SP_MediaPlay).cacheKey():
            # Currently showing play, switch to pause
            self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
            self.play_toggled.emit(True)
        else:
            # Currently showing pause, switch to play
            self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.play_toggled.emit(False)
    
    def toggle_color_range_inputs(self):
        """Toggle the min/max input fields based on auto range checkbox"""
        auto_color_range = self.auto_range_cb.isChecked()
        
        # Only enable min/max input fields if auto range is off
        enabled = not auto_color_range
        
        # Set input field enabled state
        self.min_range_input.setEnabled(enabled)
        self.max_range_input.setEnabled(enabled)
    
    def toggle_cell_color_range_inputs(self):
        """Toggle the cell min/max input fields based on auto range checkbox"""
        auto_color_range = self.cell_auto_range_cb.isChecked()
        
        # Only enable min/max input fields if auto range is off
        enabled = not auto_color_range
        
        # Set input field enabled state
        self.cell_min_range_input.setEnabled(enabled)
        self.cell_max_range_input.setEnabled(enabled)
    
    def toggle_slice_color_range_inputs(self):
        """Toggle the slice min/max input fields based on auto range checkbox"""
        auto_color_range = self.slice_auto_range_cb.isChecked()
        
        # Only enable min/max input fields if auto range is off
        enabled = not auto_color_range
        
        # Set input field enabled state
        self.slice_min_range_input.setEnabled(enabled)
        self.slice_max_range_input.setEnabled(enabled)
    
    def display_options_changed_handler(self):
        """Handle any display option change"""
        # Collect all display options into a dictionary
        options = {
            'show_cells': self.show_mat_cb.isChecked(),
            'show_microenv': self.show_microenv_cb.isChecked(),
            'wireframe_mode': self.microenv_wireframe_cb.isChecked(),
            'auto_range': self.auto_range_cb.isChecked(),
            'min_value': self.min_range_input.value(),
            'max_value': self.max_range_input.value(),
            'cell_auto_range': self.cell_auto_range_cb.isChecked(),
            'cell_min_value': self.cell_min_range_input.value(),
            'cell_max_value': self.cell_max_range_input.value(),
            'slice_auto_range': self.slice_auto_range_cb.isChecked(),
            'slice_min_value': self.slice_min_range_input.value(),
            'slice_max_value': self.slice_max_range_input.value(),
            'cell_opacity': self.cell_opacity_slider.value(),
            'microenv_opacity': self.microenv_opacity_slider.value(),
            'cell_as_spheres': self.cell_as_spheres_cb.isChecked()
        }
        
        # Emit signal with all options
        self.display_options_changed.emit(options)
    
    def slice_options_changed_handler(self):
        """Handle any slice option change"""
        # Collect all slice options into a dictionary
        options = {
            'show_slice': self.show_slice_cb.isChecked(),
            'position': (self.slice_x.value(), self.slice_y.value(), self.slice_z.value()),
            'normal': (self.slice_i.value(), self.slice_j.value(), self.slice_k.value()),
            'show_contour': self.show_contour_cb.isChecked(),
            'contour_value': self.slice_contour_value.value()
        }
        
        # Emit signal with all options
        self.slice_options_changed.emit(options)
    
    def set_max_frame(self, max_frame):
        """Set the maximum frame number and enable navigation controls"""
        self.frame_slider.setMaximum(max_frame)
        self.frame_spinbox.setMaximum(max_frame)
        
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
        self.auto_range_cb.setEnabled(True)
        self.cell_auto_range_cb.setEnabled(True)
        self.slice_auto_range_cb.setEnabled(True)
        self.show_slice_cb.setEnabled(True)
        self.slice_x.setEnabled(True)
        self.slice_y.setEnabled(True)
        self.slice_z.setEnabled(True)
        self.slice_i.setEnabled(True)
        self.slice_j.setEnabled(True)
        self.slice_k.setEnabled(True)
        self.slice_contour_value.setEnabled(True)
        self.show_contour_cb.setEnabled(True)
        self.apply_btn.setEnabled(True)
        
        # Enable variable selection dropdowns
        self.microenv_var_combo.setEnabled(True)
        self.cell_var_combo.setEnabled(True)
        self.cell_color_combo.setEnabled(True)
        self.cell_as_spheres_cb.setEnabled(True)
    
    def set_frame(self, frame):
        """Set the current frame number"""
        # Block signals to prevent triggering frame change events
        self.frame_slider.blockSignals(True)
        self.frame_spinbox.blockSignals(True)
        
        self.frame_slider.setValue(frame)
        self.frame_spinbox.setValue(frame)
        
        # Unblock signals
        self.frame_slider.blockSignals(False)
        self.frame_spinbox.blockSignals(False)
        
        # Update frame info label - applied state
        self.frame_info_label.setText(f"Frame: {frame}")
        self.frame_info_label.setStyleSheet("")  # Reset styling
    
    def update_info(self, text):
        """Update the info label text"""
        self.info_label.setText(text)
    
    def update_min_max_ranges(self, min_val, max_val, cell_min_val=None, cell_max_val=None, slice_min_val=None, slice_max_val=None):
        """Update the min/max range inputs with current data values"""
        # Block signals to avoid triggering visualization update
        self.min_range_input.blockSignals(True)
        self.max_range_input.blockSignals(True)
        self.cell_min_range_input.blockSignals(True)
        self.cell_max_range_input.blockSignals(True)
        self.slice_min_range_input.blockSignals(True)
        self.slice_max_range_input.blockSignals(True)
        
        # Update microenvironment values
        self.min_range_input.setValue(min_val)
        self.max_range_input.setValue(max_val)
        
        # Update cell values if provided
        if cell_min_val is not None and cell_max_val is not None:
            self.cell_min_range_input.setValue(cell_min_val)
            self.cell_max_range_input.setValue(cell_max_val)
        
        # Update slice values if provided
        if slice_min_val is not None and slice_max_val is not None:
            self.slice_min_range_input.setValue(slice_min_val)
            self.slice_max_range_input.setValue(slice_max_val)
        
        # Unblock signals
        self.min_range_input.blockSignals(False)
        self.max_range_input.blockSignals(False)
        self.cell_min_range_input.blockSignals(False)
        self.cell_max_range_input.blockSignals(False)
        self.slice_min_range_input.blockSignals(False)
        self.slice_max_range_input.blockSignals(False)
    
    def update_variable_dropdowns(self, xml_file, cells_file):
        """Update the dropdown menus with variables from the current files"""
        # Clear previous items
        self.microenv_var_combo.clear()
        self.cell_var_combo.clear()
        self.cell_color_combo.clear()
        
        # Get variables from files
        microenv_vars = get_microenv_variables(xml_file)
        cell_vars = get_cell_variables(xml_file, cells_file)
        
        # Print debug info about the variables
        print(f"Dropdown update: Found {len(cell_vars)} cell variables")
        print(f"Variables names: {[var['name'] for var in cell_vars]}")
        
        # Populate microenvironment variable dropdown
        for var in microenv_vars:
            var_name = var['name']
            var_units = var.get('units', 'dimensionless')
            self.microenv_var_combo.addItem(f"{var_name} ({var_units})", var)
        
        # Populate cell variable dropdown
        for var in cell_vars:
            var_name = var['name']
            var_desc = var.get('description', '')
            
            # Ensure the variable name is properly formatted for display
            display_name = f"{var_name}"
            if var_desc:
                display_name = f"{var_name} - {var_desc}"
                
            self.cell_var_combo.addItem(display_name, var)
        
        # Populate cell coloring dropdown
        # Add "Default cell type colors" option
        self.cell_color_combo.addItem("Default Cell Type Colors", None)
        
        # Add all cell variables
        for var in cell_vars:
            var_name = var['name']
            var_desc = var.get('description', '')
            
            # Ensure the variable name is properly formatted for display
            display_name = f"{var_name}"
            if var_desc:
                display_name = f"{var_name} - {var_desc}"
                
            self.cell_color_combo.addItem(display_name, var)
       
        # Emit variable selection signal to update visualization
        self.emit_variable_selection()
    
    def on_microenv_var_changed(self, index):
        """Handle microenvironment variable selection change"""
        if index >= 0:
            # Get selected variable
            var = self.microenv_var_combo.itemData(index)
            
            # Create and emit variable selection
            self.emit_variable_selection()
            
            # Highlight apply button to indicate pending changes
            self.apply_btn.setStyleSheet("font-weight: bold; background-color: #FF5722; color: white;")
    
    def on_cell_var_changed(self, index):
        """Handle cell variable selection change"""
        if index >= 0:
            # Get selected variable
            var = self.cell_var_combo.itemData(index)
            
            # Create and emit variable selection
            self.emit_variable_selection()
            
            # Highlight apply button to indicate pending changes
            self.apply_btn.setStyleSheet("font-weight: bold; background-color: #FF5722; color: white;")
    
    def on_cell_color_var_changed(self, index):
        """Handle cell color variable selection change"""
        # Create and emit variable selection
        self.emit_variable_selection()
        
        # Highlight apply button to indicate pending changes
        self.apply_btn.setStyleSheet("font-weight: bold; background-color: #FF5722; color: white;")
    
    def on_display_option_changed(self):
        """Handle any display option change"""
        # Highlight apply button to indicate pending changes
        self.apply_btn.setStyleSheet("font-weight: bold; background-color: #FF5722; color: white;")
    
    def on_slice_option_changed(self):
        """Handle any slice option change"""
        # Highlight apply button to indicate pending changes
        self.apply_btn.setStyleSheet("font-weight: bold; background-color: #FF5722; color: white;")
    
    def emit_variable_selection(self):
        """Create and emit the current variable selection"""
        # Get selected microenvironment variable
        microenv_idx = self.microenv_var_combo.currentIndex()
        microenv_var = self.microenv_var_combo.itemData(microenv_idx) if microenv_idx >= 0 else None
        
        # Get selected cell variable
        cell_idx = self.cell_var_combo.currentIndex()
        cell_var = self.cell_var_combo.itemData(cell_idx) if cell_idx >= 0 else None
        
        # Get selected cell color variable
        color_idx = self.cell_color_combo.currentIndex()
        color_var = self.cell_color_combo.itemData(color_idx) if color_idx > 0 else None
        
        # Create selection dictionary
        selection = {
            'microenv_var': microenv_var,
            'cell_var': cell_var,
            'cell_color_var': color_var,
            'wireframe': self.microenv_wireframe_cb.isChecked(),
            'cell_as_spheres': self.cell_as_spheres_cb.isChecked()
        }
        
        # Emit signal with selection
        self.variables_selected.emit(selection)

    def hide_slice_controls(self):
        """Hide all slice-related controls when in cells-only mode"""
        # Find all the slice controls
        slice_controls = [
            self.slice_x, self.slice_y, self.slice_z,
            self.slice_i, self.slice_j, self.slice_k,
            self.slice_auto_range_cb,
            self.slice_min_range_input, self.slice_max_range_input
        ]
        
        # Hide the slice group box by finding the parent group box
        for widget in self.scroll_widget.findChildren(QGroupBox):
            if widget.title() == "Slice Controls":
                widget.setVisible(False)
            elif widget.title() == "Slice Color Range":
                widget.setVisible(False)
        
        # Disable all slice controls
        for control in slice_controls:
            control.setEnabled(False)
