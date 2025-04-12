"""
Dialog for selecting PhysiCell variables to visualize
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QListWidget, QListWidgetItem, QTabWidget,
    QWidget, QCheckBox, QSplitter, QGroupBox
)
from PyQt5.QtCore import Qt, pyqtSignal

from physi_cell_vtk_viewer.utils.variable_reader import (
    get_microenv_variables, get_cell_variables
)


class VariableSelectorDialog(QDialog):
    """Dialog for selecting variables from PhysiCell output"""
    
    # Signal emitted when variable selection changes
    variable_selected = pyqtSignal(dict)
    
    def __init__(self, xml_file, cells_file=None, parent=None):
        """Initialize the dialog"""
        super().__init__(parent)
        
        self.xml_file = xml_file
        self.cells_file = cells_file
        
        # Get variables from files
        self.microenv_vars = get_microenv_variables(xml_file)
        self.cell_vars = get_cell_variables(xml_file, cells_file)
        
        # Currently selected variables
        self.selected_microenv_var = None
        self.selected_cell_var = None
        self.selected_cell_color_var = None
        
        # Set up UI
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the dialog UI"""
        self.setWindowTitle("Select Variables")
        self.resize(600, 400)
        
        # Main layout
        layout = QVBoxLayout(self)
        
        # Create tabs
        self.tab_widget = QTabWidget()
        self.microenv_tab = QWidget()
        self.cells_tab = QWidget()
        
        self.setup_microenv_tab()
        self.setup_cells_tab()
        
        self.tab_widget.addTab(self.microenv_tab, "Microenvironment")
        self.tab_widget.addTab(self.cells_tab, "Cells")
        
        layout.addWidget(self.tab_widget)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply_selection)
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        
        button_layout.addStretch()
        button_layout.addWidget(self.apply_button)
        button_layout.addWidget(self.close_button)
        
        layout.addLayout(button_layout)
    
    def setup_microenv_tab(self):
        """Set up the microenvironment tab"""
        layout = QVBoxLayout(self.microenv_tab)
        
        # Title and description
        layout.addWidget(QLabel("Select Microenvironment Variable:"))
        layout.addWidget(QLabel("Choose a substrate to visualize in the 3D view."))
        
        # Variable selection combo box
        self.microenv_combo = QComboBox()
        for var in self.microenv_vars:
            self.microenv_combo.addItem(
                f"{var['name']} ({var['units']})", 
                userData=var
            )
        
        self.microenv_combo.currentIndexChanged.connect(self.on_microenv_selection_changed)
        layout.addWidget(self.microenv_combo)
        
        # Display options
        options_group = QGroupBox("Display Options")
        options_layout = QVBoxLayout(options_group)
        
        self.wireframe_cb = QCheckBox("Show as Wireframe")
        self.wireframe_cb.setChecked(False)
        options_layout.addWidget(self.wireframe_cb)
        
        layout.addWidget(options_group)
        layout.addStretch()
    
    def setup_cells_tab(self):
        """Set up the cells tab"""
        layout = QVBoxLayout(self.cells_tab)
        
        # Title and description
        layout.addWidget(QLabel("Select Cell Variables:"))
        layout.addWidget(QLabel("Choose cell variables to visualize."))
        
        # Split into two columns
        splitter = QSplitter(Qt.Horizontal)
        
        # Left side - variable selection
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        left_layout.addWidget(QLabel("Variable to Visualize:"))
        self.cell_var_combo = QComboBox()
        for var in self.cell_vars:
            self.cell_var_combo.addItem(
                f"{var['name']} - {var['description']}", 
                userData=var
            )
        
        self.cell_var_combo.currentIndexChanged.connect(self.on_cell_selection_changed)
        left_layout.addWidget(self.cell_var_combo)
        
        # Right side - color mapping
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        right_layout.addWidget(QLabel("Color Cells By:"))
        self.cell_color_combo = QComboBox()
        
        # Add None option
        self.cell_color_combo.addItem("Default Cell Type Colors", userData=None)
        
        # Add all variables as options for coloring
        for var in self.cell_vars:
            self.cell_color_combo.addItem(
                f"{var['name']} - {var['description']}", 
                userData=var
            )
        
        self.cell_color_combo.currentIndexChanged.connect(self.on_cell_color_selection_changed)
        right_layout.addWidget(self.cell_color_combo)
        
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        
        layout.addWidget(splitter)
        
        # Display options
        options_group = QGroupBox("Display Options")
        options_layout = QVBoxLayout(options_group)
        
        self.cell_as_spheres_cb = QCheckBox("Display as Spheres")
        self.cell_as_spheres_cb.setChecked(True)
        options_layout.addWidget(self.cell_as_spheres_cb)
        
        layout.addWidget(options_group)
        layout.addStretch()
    
    def on_microenv_selection_changed(self, index):
        """Handle microenvironment variable selection change"""
        if index >= 0 and index < len(self.microenv_vars):
            self.selected_microenv_var = self.microenv_vars[index]
    
    def on_cell_selection_changed(self, index):
        """Handle cell variable selection change"""
        if index >= 0 and index < len(self.cell_vars):
            self.selected_cell_var = self.cell_vars[index]
    
    def on_cell_color_selection_changed(self, index):
        """Handle cell color variable selection change"""
        if index == 0:  # "Default Cell Type Colors"
            self.selected_cell_color_var = None
        elif index > 0 and index <= len(self.cell_vars):
            self.selected_cell_color_var = self.cell_vars[index - 1]
    
    def apply_selection(self):
        """Apply the selected variables"""
        # Create selection dictionary
        selection = {
            'microenv_var': self.selected_microenv_var,
            'cell_var': self.selected_cell_var,
            'cell_color_var': self.selected_cell_color_var,
            'wireframe': self.wireframe_cb.isChecked(),
            'cell_as_spheres': self.cell_as_spheres_cb.isChecked()
        }
        
        # Emit signal with selection
        self.variable_selected.emit(selection)
        
        # Close the dialog
        self.accept() 