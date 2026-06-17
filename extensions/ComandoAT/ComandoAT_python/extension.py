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
        self._physx_sub = None
        self.on_play_sub = None
        self.on_stop_sub = None

    def on_physics_step(self, dt):
        if self.is_simulation_running:
            self.rigid_prim.apply_forces(
                forces=self.forces_to_apply,
                is_global=True
            )

    def init_rigid_prim(self):
        blade_paths = [
        "/AeroTaxi/BladeNW",
        "/AeroTaxi/BladeNE",
        "/AeroTaxi/BladeW",
        "/AeroTaxi/BladeE",
        "/AeroTaxi/BladeSW",
        "/AeroTaxi/BladeSE",
       ]

        self.rigid_prim = RigidPrim(
            prim_paths_expr=blade_paths
        )
        self.rigid_prim.initialize()

        for i, prim in enumerate(self.rigid_prim.prims):
            name = prim.GetName()
            self.blade_prim_to_index[name] = i
            self.blade_prims_order_names.append(name)
            self.blade_prims_order_index.append(i)
        self.force_to_apply = np.zeros((self.rigid_prim.count, 3), dtype=np.float32)
        # self.force_to_apply[:, 2] = self.hover_force / self.rigid_prim.count
        self.gravity_prim_force = self.hover_force / self.rigid_prim.count

    def init_articulation_root(self):

        self.articulations = Articulation(
            prim_paths_expr=["/AeroTaxi"]
        )
        self.articulations.initialize()

    def on_timeline_play(self, event):
        if not self.is_simulation_running:
            self.init_articulation_root()
            self.init_rigid_prim()
            self.is_simulation_running = True
            #self.set_ninety()
    
    def on_timeline_stop(self, event):
        if self.is_simulation_running:
            self.articulations = None 
            self.rigid_prim = None
            self.is_simulation_running = False

            self.forces_to_apply = np.zeros((6, 3))

    def init_vars(self):

        self.is_simulation_running = False

        self.articulations = None
        self.joint_names = []
        self.joint_name_to_index = {}

        self.rotor_joint_names = ["JRotorNW","JRotorNE","JRotorW","JRotorE","JRotorSW","JRotorSE"]
        self.blade_joint_names = ["JBladeNW","JBladeNE","JBladeW","JBladeE","JBladeSW","JBladeSE"]

        self.rotor_angles = {}
        self.blade_speed = {}

        self.blade_prim_name = ["BladeNW","BladeNE","BladeW","BladeE","BladeSW","BladeSE"]
        self.blade_prim_to_index = {}
        self.blade_prims_order_names = []
        self.blade_prims_order_index = []

        self.forces_to_apply = np.zeros((6, 3), dtype=np.float32)
        self.torques_to_apply = np.zeros((6, 3), dtype=np.float32)
        self._physx_sub = None
        self.rotor_sliders = {}
        self.blade_sliders = {}
        self.force_fields = {}
        

        self.mass = 2002
        self.hover_force = 9.81 * self.mass
        self.gravity_prim_force = 0
        self.noventa = False
        self.current_joint_positions = None


        self.wait_after_ninety = 2.5
        self.wait_timer = 0.0
        self.waiting_for_rotors = False
        self.noventa = False
        


        self.apply_forces_enabled = False
        # Phyxs callback
        self._physx_sub = omni.physx.acquire_physx_interface().subscribe_physics_step_events(
            self.on_physics_step
        )

        # Timeline callbacks
        self.timeline = omni.timeline.get_timeline_interface()
        timeline_stream = self.timeline.get_timeline_event_stream()

        self.on_stop_sub = timeline_stream.create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.STOP), 
            self.on_timeline_stop
        )

        self.on_play_sub = timeline_stream.create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.PLAY), 
            self.on_timeline_play
        )

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
                            for name in self.rotor_joint_names:
                                with ui.HStack(alignment=ui.Alignment.RIGHT):
                                    ui.Label(name)
                                    
                                    slider = ui.FloatSlider(
                                        min=0, 
                                        max=100,
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
                                        name=name: self.on_rotor_change(model, name)
                                    )

                                    self.rotor_sliders[name] = slider
                            button_90 = ui.Button(clicked_fn = self.set_ninety, text = "Set 90 degrees")
                            button_0 = ui.Button(clicked_fn = self.set_zero, text = "Set 0 degrees")

                    #Poner cada pala a una velocidad diferente
                    self.blade_speeds = ui.CollapsableFrame(
                        title="Blade Speed", 
                        collapsed=False
                    )
                    with self.blade_speeds:
                        with ui.VStack(style={"margin": 1}, height=0, spacing=5):
                            for blade_name in self.blade_joint_names:
                                with ui.HStack(alignment=ui.Alignment.RIGHT):
                                    ui.Label(blade_name)
                                    
                                    slider = ui.FloatSlider(
                                        min=0, 
                                        max=500,
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

                                    self.blade_sliders[blade_name] = slider
                            with ui.HStack():
                                self.velocity = ui.FloatField()
                                button_mandarcosi = ui.Button(clicked_fn = self.set_velocity, text = "Set velocity")
                    self.blade_forces = ui.CollapsableFrame(
                        title="Blade Force", 
                        collapsed=False
                    )
                    with self.blade_forces:
                        with ui.VStack(style={"margin": 1}, height=0, spacing=5):
                            for blade_name in self.blade_prim_name:
                                with ui.HStack(alignment=ui.Alignment.RIGHT):
                                    ui.Label(blade_name)
                                    
                                    self.force_fields[blade_name] = ui.FloatField(
                                        style={
                                            "background_color": ui.color(0.13),
                                            "secondary_color": ui.color(0.3)
                                        }
                                    
                                    )
                                    self.force_fields[blade_name].model.set_value(3273.27)
                            
                            ui.Button(text="Set Upward forces", clicked_fn=self.set_upward_forces)
                            
                            force_send_button = ui.Button(clicked_fn = self.send_forces, text = "Send forces")

    # Rotors Angles
    def on_rotor_change(self, model, rotor_name):
        if self.articulations is None:
            return
        value = model.get_value_as_float() * (-1)
        # Aplicar al joint
        self.apply_rotor_angle(rotor_name, value)

    def apply_rotor_angle(self, rotor_name, angle_deg):
        
        idx = self.articulations.get_joint_index(rotor_name)

        angle_rad = np.deg2rad(angle_deg)

        self.articulations.set_joint_position_targets(
            positions=np.array([angle_rad]),
            joint_indices=np.array([idx])
        )

    def set_ninety(self):
        if self.is_simulation_running:
            positions = np.array([
                np.deg2rad(-90),
                np.deg2rad(-90),
                np.deg2rad(-90),
                np.deg2rad(-90),
                np.deg2rad(-90),
                np.deg2rad(-90),
            ])
            indices = np.array([self.articulations.get_joint_index(name) for name in self.rotor_joint_names])

            self.articulations.set_joint_position_targets(
                positions=positions,
                joint_indices=indices
            )
    
    def set_zero(self):
        if self.is_simulation_running:
            positions = np.array([
                np.deg2rad(0),
                np.deg2rad(0),
                np.deg2rad(0),
                np.deg2rad(0),
                np.deg2rad(0),
                np.deg2rad(0),
            ])
            indices = np.array([self.articulations.get_joint_index(name) for name in self.rotor_joint_names])

            self.articulations.set_joint_position_targets(
                positions=positions,
                joint_indices=indices
            )

    def set_velocity(self):
        vel_value = self.velocity.model.get_value_as_float()
        velocity = np.zeros((6))
        velocity[:] = vel_value
        # ["JBladeNW","JBladeNE","JBladeW","JBladeE","JBladeSW","JBladeSE"]
        velocity[[1,2,4]] *= -1
        
        for blade_name, vel in zip(self.blade_joint_names, velocity):
            self.apply_blade_vel(blade_name, vel)
        
        
    # Blade Speeds
    def on_blade_change(self, model, blade_name):
        if self.articulations is None:
            return
        
        if blade_name in ["JBladeSE","JBladeNW","JBladeE"]:
            value = model.get_value_as_float() 
        else: 
            value = model.get_value_as_float() * (-1)
        # Guardar valor
        self.blade_speed[blade_name] = value
        # Aplicar al joint
        self.apply_blade_vel(blade_name, value)

    def apply_blade_vel(self, blade_name, vel):
        idx = self.articulations.get_joint_index(blade_name)

        self.articulations.set_joint_velocity_targets(
            velocities =np.array([vel]),
            joint_indices=np.array([idx])
        )

    # Blade Forces
    def set_upward_forces(self):
        for name in self.blade_prim_name:
            self.force_fields[name].model.set_value(3273.27 * 1.1)
    
    def send_forces(self):
        for blade_name in self.blade_prim_name:
            index = self.blade_prim_to_index[blade_name]
            self.forces_to_apply[index, 2] = self.force_fields[blade_name].model.get_value_as_float()
            
        self.forces_to_apply[:, 2] -= self.gravity_prim_force
        print(self.forces_to_apply)
        
                
            