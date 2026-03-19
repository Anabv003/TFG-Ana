import asyncio


import omni.kit.app
import omni.ext
from isaacsim.gui.components.ui_utils import ui
from omni.isaac.ui.element_wrappers import DropDown
from omni.isaac.core.utils.stage import get_current_stage
import carb.events
import omni.timeline
import omni.physx

from isaacsim.core.prims import Articulation
from isaacsim.core.prims import RigidPrim
from navsim_utils.extensions_utils import ExtensionUtils
import numpy as np




class AT_Comando(omni.ext.IExt):
    def on_startup(self, ext_id):
        self.init_vars()
        self.joints()
        self.build_ui()

    def on_shutdown(self):
        #self.stop_update()

        self.on_physics_step_sub = None
        self.on_stop_sub = None
        self.plots_update_sub = None

    def joints(self):
        self.articulations = Articulation(
            prim_paths_expr=["/root"]
        )
        self.articulations.initialize()

        self.joint_names =list(self.articulations.joint_names)
        self.joint_name_to_index = {name: i for i, name in enumerate(self.joint_names)}
        self.rotor_joint_names = [name for name in self.joint_names if name.startswith("JRotor")]
        self.palas_joint_names = [name for name in self.joint_names if name.startswith("JPalas")]

        self.rotor_angles = {name : 0.0 for name in self.rotor_joint_names}
        self.palas_vel = {name : 0.0 for name in self.palas_joint_names}


    def on_physics_step(self, step_size:int):
        self.current_time += step_size

    def on_stop(self, event):
        self.current_time = 0
        # self.start_stop_tool_button.model.set_value(False)
        # self.start_stop_update()
        #self.stop_update()

    def init_vars(self):
        self.event_stream = omni.kit.app.get_app_interface().get_message_bus_event_stream()
        self.operator_uav_event = carb.events.type_from_string("NavSim.OperatorUAV")
        
        self.ext_utils = ExtensionUtils()
        self.current_time = 0
        self.stop_update_plot = True

        self.articulations = None
        self.joint_names = []
        self.joint_name_to_index = {}

        self.rotor_joint_names = []
        self.palas_joint_names = []

        self.rotor_angles = {}
        self.palas_vel = {}

        self.rotor_sliders = {}
        self.palas_sliders = {}

        self.current_joint_positions = None
        
        self.physx_interface = omni.physx.get_physx_interface()

        self.timeline = omni.timeline.get_timeline_interface()
        timeline_stream = self.timeline.get_timeline_event_stream()
        self.on_stop_sub = timeline_stream.create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.STOP), 
            self.on_stop
        )
#Interfaz
    def build_ui(self):
        # The ui.RasterPolicy.NEVER is to always update plots line drawing
        self.window = ui.Window(
            "AT - Manual Control", 
            width=600, 
            height=600, 
            raster_policy=ui.RasterPolicy.NEVER
        )

        with self.window.frame:
            with ui.ScrollingFrame():
                with ui.VStack(spacing=10, height=0):
                    ui.Spacer(height=10)

                    #Poner cada rotor un ángulo diferente
                    self.rotors_angle = ui.CollapsableFrame(
                        title="Rotors Angles", 
                        collapsed=False
                    )

                    with self.rotors_angle:
                        with ui.VStack(style={"margin": 1}, height=0, spacing=5):
                            for rotor_name in self.rotor_angles.keys():
                                with ui.HStack(alignment=ui.Alignment.RIGHT):
                                    ui.Label(rotor_name)

                                    slider = ui.FloatSlider(
                                        min=0, 
                                        max=90,
                                        step=1,
                                        precision=1,
                                        style={
                                            "background_color": ui.color(0.13),
                                            "secondary_color": ui.color(0.3),
                                            "draw_mode": ui.SliderDrawMode.FILLED
                                        }
                                    )

                                    slider.model.set_value(0.0)
                                    slider.model.add_value_changed_fn(
                                        lambda model,
                                        name=rotor_name: self.on_rotor_change(model, name)
                                    )

                                    self.rotor_sliders[rotor_name] = slider


                            ui.Spacer(height=15)

    def on_rotor_change(self, model, rotor_name):
        value = model.get_value_as_float()
        # Guardar valor
        self.rotor_angles[rotor_name] = value
        # Aplicar al joint
        self.apply_rotor_angle(rotor_name, value)

    def apply_rotor_angle(self, rotor_name, angle_deg):
        idx = self.articulations.get_joint_index(rotor_name)

        angle_rad = np.deg2rad(angle_deg)

        self.articulations.set_joint_positions(
            positions=np.array([angle_rad]),
            joint_indices=np.array([idx])
        )