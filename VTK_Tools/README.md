# PhysiCell VTK Tools

Utilities for visualizing PhysiCell simulation data using VTK.

## PhysiCell Slicer

A Python script to display 2D slices of PhysiCell microenvironment data using VTK. The script reads PhysiCell output files and visualizes a slice of the microenvironment at a specified position and orientation.

### Dependencies

- Python 3.x
- NumPy
- SciPy
- VTK

You can install the dependencies with:

```bash
pip install numpy scipy vtk
```

### Usage

```bash
python physicell_slicer.py --filename <filename> [options]
```

#### Parameters

- `--filename`: Root name of PhysiCell output files (without extension)
  - e.g., `output0000060` will load `output0000060.xml`, `output0000060_cells.mat`, etc.
  - Supports both relative and absolute file paths
  - Example with absolute path: `--filename /path/to/data/output0000060`
- `--log`: Enable logging to understand what is going on
  - Logs are written to both the console and a file named `output.log`
- `--range min,max`: User-defined range of values to display in the colormap
  - If not provided, auto-range based on min/max values of the scalars will be used
- `--position x,y,z`: Position of the slice
  - If not provided, slice will be at the center of the domain
- `--normal i,j,k`: Orientation of the slice
  - Default: `0,0,1` (XY plane)
- `--substrate name_or_index`: Name or index of the substrate to display
  - Can be specified by name (e.g., "oxygen") or by index (e.g., "4")
  - If not provided, the first substrate will be used
  - Substrate names are read from the PhysiCell XML configuration file
- `--contour value`: Draw a contour polyline at the specified value
  - Shows a black polyline where the substrate equals the specified value
  - Useful for highlighting specific concentration boundaries

### Examples

Visualize a slice at the default position (center) with the default orientation (XY plane):
```bash
python physicell_slicer.py --filename output0000060
```

Visualize a slice with logging enabled:
```bash
python physicell_slicer.py --filename output0000060 --log
```

Visualize a slice at a specific position with a specific orientation:
```bash
python physicell_slicer.py --filename output0000060 --position 100,100,50 --normal 0,1,0
```

Use a custom colormap range:
```bash
python physicell_slicer.py --filename output0000060 --range 0,1
```

Display a specific substrate by name:
```bash
python physicell_slicer.py --filename output0000060 --substrate "oxygen"
```

Display a specific substrate by index:
```bash
python physicell_slicer.py --filename output0000060 --substrate 4
```

Draw a contour line at a specific value:
```bash
python physicell_slicer.py --filename output0000060 --contour 0.02
```

Combine multiple options:
```bash
python physicell_slicer.py --filename output0000060 --position 100,100,50 --normal 0,1,0 --range 0,2 --substrate "oxygen" --contour 0.5 --log
```

Using an absolute path:
```bash
python physicell_slicer.py --filename /Users/username/Documents/PhysiCell/output/output0000060
```

### Interaction

Once the visualization is displayed, you can:
- Rotate the view by dragging with the left mouse button
- Pan by dragging with the middle mouse button
- Zoom in/out using the mouse wheel or dragging with the right mouse button

### Notes

- The script shows a colormap legend on the right side of the window
- A wireframe outline of the complete domain is displayed
- Orientation axes are shown in the bottom-left corner
- Slice information (position, normal, substrate, value range, contour) is displayed on the top-left corner
- With the `--log` option, a complete substrate list with indices, names, and value ranges is written to `output.log` 