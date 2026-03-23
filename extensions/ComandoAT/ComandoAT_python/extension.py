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

    # def on_shutdown(self):
    #     pass

    def joints(self):

        self.articulations = Articulation(
            prim_paths_expr=["/root"]
        )
        self.articulations.initialize()

        self.joint_names =list(self.articulations.joint_names)
        self.joint_name_to_index = {name: i for i, name in enumerate(self.joint_names)}
        self.rotor_joint_names = [name for name in self.joint_names if name.startswith("JRotor")]
        self.blade_joint_names = [name for name in self.joint_names if name.startswith("JPalas")]

        self.rotor_angles = {name : 0.0 for name in self.rotor_joint_names}
        self.blade_speed = {name : 0.0 for name in self.blade_joint_names}

    # def on_timeline_play(self, event):
    #     if not self.is_simulation_running:
    #         self.joints()
    #         self.is_simulation_running = True
    
    # def on_timeline_stop(self, event):
        if self.is_simulation_running:
            self.articulations = None 
            self.is_simulation_running = False

    def init_vars(self):

        self.is_simulation_running = False

        self.articulations = None
        self.joint_names = []
        self.joint_name_to_index = {}

        self.rotor_joint_names = []
        self.blade_joint_names = []

        self.rotor_angles = {}
        self.blade_speed = {}

        self.rotor_sliders = {}
        self.blase_sliders = {}

        self.current_joint_positions = None

        #  # Timeline callbacks
        # self.timeline = omni.timeline.get_timeline_interface()
        # timeline_stream = self.timeline.get_timeline_event_stream()

        # self.on_stop_sub = timeline_stream.create_subscription_to_pop_by_type(
        #     int(omni.timeline.TimelineEventType.STOP), 
        #     self.on_timeline_stop
        # )

        # self.on_play_sub = timeline_stream.create_subscription_to_pop_by_type(
        #     int(omni.timeline.TimelineEventType.PLAY), 
        #     self.on_timeline_play
        # )

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

                    #Poner cada pala a una velocidad diferente
                    self.blade_speeds = ui.CollapsableFrame(
                        title="Blade Speed", 
                        collapsed=False
                    )

                    with self.blade_speeds:
                        with ui.VStack(style={"margin": 1}, height=0, spacing=5):
                            for blade_name in self.blade_speed.keys():
                                with ui.HStack(alignment=ui.Alignment.RIGHT):
                                    ui.Label(blade_name)

                                    slider = ui.FloatSlider(
                                        min=-120, 
                                        max=120,
                                        step=2,
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
                                        name=blade_name: self.on_blade_change(model, name)
                                    )

                                    self.blase_sliders[blade_name] = slider


    def on_rotor_change(self, model, rotor_name):
        value = model.get_value_as_float()
        # Guardar valor
        self.rotor_angles[rotor_name] = value
        # Aplicar al joint
        self.apply_rotor_angle(rotor_name, value)

    # def on_rotor_change(self, model, rotor_name):
    #     raw = model.get_value_as_float()

    #     # invertir
    #     value = 90 - raw

    #     self.rotor_angles[rotor_name] = value
    #     self.apply_rotor_angle(rotor_name, value)

    def apply_rotor_angle(self, rotor_name, angle_deg):
        idx = self.articulations.get_joint_index(rotor_name)

        angle_rad = np.deg2rad(angle_deg)

        # self.articulations.set_joint_positions(
        #     positions=np.array([angle_rad]),
        #     joint_indices=np.array([idx])
        # )
        self.articulations.set_joint_position_targets(
            positions=np.array([angle_rad]),
            joint_indices=np.array([idx])
        )

    def on_blade_change(self, model, blade_name):
        value = model.get_value_as_float()
        # Guardar valor
        self.blade_speed[blade_name] = value
        # Aplicar al joint
        self.apply_blade_vel(blade_name, value)

    def apply_blade_vel(self, blade_name, vel):
        idx = self.articulations.get_joint_index(blade_name)

        #angle_rad = np.deg2rad(angle_deg)

        self.articulations.set_joint_velocity_targets(
            velocities =np.array([vel]),
            joint_indices=np.array([idx])
        )
        # self.articulations.set_joint_position_targets(
        #     positions=np.array([1.57,1.57,1.57,1.57,1.57,1.57]),
        #     joint_indices=range(6, 12)
        # )


