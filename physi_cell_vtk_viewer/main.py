"""
Main entry point for PhysiCell VTK Viewer
"""

import sys
import os
import argparse
from PyQt5.QtWidgets import QApplication

from physi_cell_vtk_viewer.ui.main_window import MainWindow

def main():
    """Entry point for the PhysiCell VTK Viewer application"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='PhysiCell VTK Viewer')
    parser.add_argument('--directory', '-d', type=str, help='PhysiCell output directory to load')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()
    
    # Create application
    app = QApplication(sys.argv)
    
    # Create main window
    initial_dir = args.directory if args.directory and os.path.isdir(args.directory) else None
    window = MainWindow(initial_dir=initial_dir, debug=args.debug)
    window.show()
    
    # Run the application
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
