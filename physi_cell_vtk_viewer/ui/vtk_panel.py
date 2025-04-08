"""
VTK rendering panel for PhysiCell viewer
"""

import vtk
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from physi_cell_vtk_viewer.utils.vtk_utils import add_orientation_axes

class VTKPanel(QWidget):
    """VTK rendering panel for PhysiCell visualization"""
    
    def __init__(self, parent=None):
        """Initialize the VTK panel"""
        super().__init__(parent)
        
        # Set up the VTK panel layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Create the VTK widget
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        self.layout.addWidget(self.vtk_widget)
        
        # Initialize VTK rendering components
        self.setup_vtk_visualization()
        
        # Add a demo visualization initially
        self.add_demo_visualization()
    
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
        self.axes_widget = add_orientation_axes(self.renderer)
        self.axes_widget.SetInteractor(self.interactor)
        self.axes_widget.EnabledOn()
        
        # Setup camera for better zooming
        camera = self.renderer.GetActiveCamera()
        camera.SetClippingRange(0.1, 10000)  # Wide clipping range
        
        # Initialize the interactor
        self.interactor.Initialize()
        
        # Track actor properties for LOD during interaction
        self.actor_properties = {}  # Store original properties
    
    def add_demo_visualization(self):
        """Add a simple demo visualization with instructions"""
        # Add instructional text
        text_actor = vtk.vtkTextActor()
        text_actor.SetInput("PhysiCell VTK Viewer\n\nLoad a directory to begin visualization.")
        text_actor.GetTextProperty().SetFontSize(24)
        text_actor.GetTextProperty().SetColor(1, 1, 1)  # White text
        text_actor.SetPosition(20, 20)
        self.renderer.AddActor2D(text_actor)
        
        # Create axes to show coordinate system
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(5, 5, 5)
        axes.SetShaftTypeToLine()
        
        # Position the axes in the center
        transform = vtk.vtkTransform()
        transform.Translate(0, 0, 0)
        axes.SetUserTransform(transform)
        
        self.renderer.AddActor(axes)
        
        # Reset the camera to show the demo
        self.renderer.ResetCamera()
    
    def on_interaction_start(self, obj, event):
        """Handle the start of user interaction - reduce quality for better performance"""
        # Store original properties of actors
        self.actor_properties = {}
        
        # Set a higher desired update rate during interaction
        self.render_window.SetDesiredUpdateRate(30.0)
        
        # Reduce the resolution of any sphere sources
        for actor in self.renderer.GetActors():
            mapper = actor.GetMapper()
            if mapper and hasattr(mapper, 'GetInput'):
                input_data = mapper.GetInput()
                if input_data and hasattr(input_data, 'GetSource'):
                    source = input_data.GetSource()
                    if source and isinstance(source, vtk.vtkSphereSource):
                        # Store original resolution
                        self.actor_properties[actor] = {
                            'phi': source.GetPhiResolution(),
                            'theta': source.GetThetaResolution()
                        }
                        # Reduce resolution during interaction
                        source.SetPhiResolution(8)
                        source.SetThetaResolution(8)
                        source.Update()
                        mapper.Update()
        
        # For volume rendering, increase sample distance during interaction
        for volume in self.renderer.GetVolumes():
            mapper = volume.GetMapper()
            if mapper and isinstance(mapper, vtk.vtkSmartVolumeMapper):
                if not volume in self.actor_properties:
                    self.actor_properties[volume] = {}
                    
                # Increase the sample distance during interaction for better performance
                original_distance = mapper.GetSampleDistance()
                self.actor_properties[volume]['sample_distance'] = original_distance
                mapper.SetSampleDistance(original_distance * 2.0)
    
    def on_interaction_end(self, obj, event):
        """Handle the end of user interaction - restore quality"""
        # Restore the original update rate
        self.render_window.SetDesiredUpdateRate(10.0)
        
        # Restore original properties of actors
        for actor, props in self.actor_properties.items():
            # For sphere sources, restore resolution
            if isinstance(actor, vtk.vtkActor):
                mapper = actor.GetMapper()
                if mapper and hasattr(mapper, 'GetInput'):
                    input_data = mapper.GetInput()
                    if input_data and hasattr(input_data, 'GetSource'):
                        source = input_data.GetSource()
                        if source and isinstance(source, vtk.vtkSphereSource) and 'phi' in props and 'theta' in props:
                            source.SetPhiResolution(props['phi'])
                            source.SetThetaResolution(props['theta'])
                            source.Update()
                            mapper.Update()
            
            # For volume mappers, restore sample distance
            elif isinstance(actor, vtk.vtkVolume):
                mapper = actor.GetMapper()
                if mapper and isinstance(mapper, vtk.vtkSmartVolumeMapper) and 'sample_distance' in props:
                    mapper.SetSampleDistance(props['sample_distance'])
        
        # Clear the stored properties
        self.actor_properties = {}
        
        # Force rendering with restored quality
        self.render_window.Render()
    
    def get_renderer(self):
        """Get the VTK renderer"""
        return self.renderer
    
    def get_interactor(self):
        """Get the VTK render window interactor"""
        return self.interactor
    
    def reset_camera(self):
        """Reset the camera to show all actors"""
        self.renderer.ResetCamera()
        self.render_window.Render()
    
    def cleanup(self):
        """Clean up VTK resources before destruction"""
        self.interactor.RemoveAllObservers()
        self.vtk_widget.GetRenderWindow().Finalize()
        self.interactor.TerminateApp()
