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

        self.joints()
        self.rotor_angles = {"JRotorAD":0.0, "JRotorMD":0.0, "JRotorFD":0.0, "JRotorAI":0.0, "JRotorMI":0.0,"JRotorFI":0.0}
        self.rotor_sliders ={}
        self.rotor_indices = {"JRotorAD":0, "JRotorMD":1, "JRotorFD":2, "JRotorAI":3, "JRotorMI":4,"JRotorFI":5}
        
        self.palas_vel = {"JPalasAD":0.0, "JPalasMD":0.0, "JPalasFD":0.0, "JPalasAI":0.0, "JPalasMI":0.0,"JPalasFI":0.0}
        self.palas_sliders = {}
        self.palas_indices = {"JPalasAD":0, "JPalasMD":1, "JPalasFD":2, "JPalasAI":3, "JPalasMI":4,"JPalasFI":0.0}
        self.physx_interface = omni.physx.get_physx_interface()
        # self.on_physics_step_sub = self.physx_interface.subscribe_physics_step_events(
        #     self.on_physics_step
        # )

        self.timeline = omni.timeline.get_timeline_interface()
        timeline_stream = self.timeline.get_timeline_event_stream()
        self.on_stop_sub = timeline_stream.create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.STOP), 
            self.on_stop
        )

        # # Plot data
        # self.x_lv_plot_data = [0.0, 0.0]
        # self.y_lv_plot_data = [0.0, 0.0]
        # self.z_lv_plot_data = [0.0, 0.0]
        # self.z_av_plot_data = [0.0, 0.0]

        # # Plots appereance
        # self.plots_appearance = 1
        # self.build_plots = False
        # update_event_stream = omni.kit.app.get_app_interface().get_update_event_stream()
        # self.plots_update_sub = update_event_stream.create_subscription_to_pop(
        #     self.build_plots_container, 
        #     name="Plots_building"
        # )

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

                            # On/Off init rotors
                            with ui.HStack(alignment=ui.Alignment.RIGHT):
                                ui.Label("Start with rotors on")
                                self.start_rotors_on_off_checkbox = ui.CheckBox(width=0)

                            ui.Spacer(height=15)

    def on_rotor_change(self, model, rotor_name):
        value = model.get_value_as_float()
        # Guardar valor
        self.rotor_angles[rotor_name] = value
        # Aplicar al joint
        self.apply_rotor_angle(rotor_name, value)

    def apply_rotor_angle(self, rotor_name, angle_deg):
        idx = self.rotor_indices[rotor_name]
        # Convertir a radianes
        angle_rad = np.deg2rad(angle_deg)
        # Obtener posiciones actuales
        joint_positions = self.articulations.get_joint_positions()
        # Modificar solo ese rotor
        joint_positions[idx] = angle_rad
        # Aplicar
        self.articulations.set_joint_positions(joint_positions)

