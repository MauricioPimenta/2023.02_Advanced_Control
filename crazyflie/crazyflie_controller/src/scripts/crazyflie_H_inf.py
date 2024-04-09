#!/usr/bin/env python3
import rospy

from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Empty
from std_srvs.srv import Empty
from geometry_msgs.msg import Vector3

import rosbag

import numpy as np
import random
import tf.transformations as tf_trans

from sensor_msgs.msg import Joy

from aurora_py.controllers import Hinf_control 

class CrazyflieHinfNode:
    def __init__(self):
        rospy.init_node('crazyflie_Hinf_node')

        ############## Crazyflie Model ############
        self.m = 0.035  # mass of the drone in kg
        self.g = 9.81  # gravity
        # kvx = 0.03
        # kvy = 0.04
        kvx = 0.0
        kvy = 0.0
        kvz = 0

        self.ang_max = 15
        self.circle_flag = False
        self.psi_ref_max = 100
        self.desired_z_height = 1
        self.ang_max_rad = np.deg2rad(self.ang_max)
        self.dont_takeoff = rospy.get_param('~dont_takeoff', True)
        self.print_kalman = rospy.get_param('~print_kalman', False)
        self.theta_trim = rospy.get_param('~theta_trim', 0.0)
        self.phi_trim = rospy.get_param('~phi_trim', 0.0)
        print(f'phi trim = {self.phi_trim}')
        self.print_controller = rospy.get_param('~print_controller', False)
        self.bag = rosbag.Bag('/home/miguel/catkin_ws/src/crazyflie/crazyflie_controller/src/data/h_inf_bag.bag', 'w')

        # Flags
        self.emergency_button_pressed = False
        self.is_landing = False
        self.goto_center = False
        self.compute_controller_flag = False

        # Is true Flags
        self.compute_controller_is_true = False
        self.emergency_flag_is_true = False
        self.pose_is_true = False
        self.joy_is_true = False

        # Message to stop the drone
        self.stop_msg = Twist()
        self.stop_msg.linear.x = 0
        self.stop_msg.linear.y = 0
        self.stop_msg.linear.z = 0
        self.stop_msg.angular.z = 0

        self.namespace = rospy.get_namespace()

        # Subscriber for pose
        self.pose_subscriber = rospy.Subscriber(f"/vrpn_client_node{self.namespace}pose", 
                                                PoseStamped, 
                                                self.pose_callback)
        
        # Subscriber for the joystick
        self.joystick_cmd = rospy.Subscriber("/joy", 
                                            Joy, 
                                            self.joy_callback)
        
        self.control_publish = rospy.Publisher(f"cmd_vel", 
                                                Twist,
                                                queue_size=10)
        
        self.trajectory_service = rospy.Service('start_trajectory', Empty, self.handle_trajectory)

        self.hz = 1/30
        self.rate = rospy.Rate(1/self.hz)

        self.wait_rate = rospy.Rate(2)
        self.timer = rospy.Timer(rospy.Duration(1/4), self.prints)
        self.bag_timer = rospy.Timer(rospy.Duration(self.hz), self.save_bag_timer)
        self.is_bag_open = True
        rospy.on_shutdown(self.shutdown_hook)
        rospy.loginfo("Crazyflie Hinf Node started")

    def shutdown_hook(self):
        self.is_bag_open = False
        self.bag.close()

    def get_euler_from_quaternion(self, quaternion):
        return np.array(tf_trans.euler_from_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w]))
    
    def emergency_landing(self, message):
        rospy.logerr(message)
        self.is_landing = True
        for _ in range(30):
            self.control_publish.publish(self.stop_msg)
        rospy.signal_shutdown('Drone Landed.')

    def joy_callback(self, data):
        if not self.emergency_button_pressed:
            self.joy_is_true = True
            self.joy = data

            if data.buttons[2]:
                self.goto_center = True
            if data.buttons[0]:
                self.emergency_button_pressed = True
                self.emergency_landing('Emergency landing')

    def pose_callback(self, msg):
        if not self.pose_is_true:
            self.pose_is_true = True
        x, y, z = msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
        roll, pitch, yaw = self.get_euler_from_quaternion(msg.pose.orientation)

        self.pose = np.array([x, y, z, roll, pitch, yaw])
        self.pose_time = msg.header.stamp

    def conversions_crazyflie(self, theta, phi, thrust):
        # Normalizing angles and conversion to degrees
        u_theta = theta/self.ang_max_rad
        u_phi   = phi/self.ang_max_rad

        theta_deg = np.sign(u_theta)*self.ang_max if np.abs(u_theta) > 1.0 else u_theta*self.ang_max
        phi_deg   = np.sign(u_phi)*self.ang_max   if np.abs(u_phi)   > 1.0 else u_phi*self.ang_max

        # transforming thrust
        a_max = 15
        conversao_digital = 60000/a_max

        digital_thrust = (thrust/self.m + self.g) * conversao_digital
        digital_thrust = min(max(digital_thrust,11000),60000)
        
        # self.real_thrust = (thrust/self.m + self.g) * self.m

        return np.array([theta_deg, phi_deg, digital_thrust])
    
    def undo_conversions_crazyflie(self, theta, phi, thrust):
        # Normalizing angles and conversion to degrees
        theta_rad = np.deg2rad(theta)
        phi_rad   = np.deg2rad(phi)

        # transforming thrust
        a_max = 15
        conversao_digital = 60000/a_max

        thrust = (thrust*self.m - self.g) / conversao_digital
        
        # self.real_thrust = (thrust/self.m + self.g) * self.m

        return theta_rad, phi_rad, thrust
    
    def compute_psi(self, psi_dot_ref, psi_desired):
        psi_til = psi_desired - self.pose[5]
        if np.abs(psi_til) > np.pi:
            if psi_til > 0:
                psi_til = psi_til - 2*np.pi
            else:
                psi_til = psi_til + 2*np.pi

        psi_ref = psi_dot_ref + 3*psi_til
        psi_ref = min(max(psi_ref,-1),1)
        return psi_ref*self.psi_ref_max
    
    def compute_thrust(self, z, z_dot, z_ref):
        a_max = 15
        g = 9.81
        conversao_digital = 60000/a_max

        zref = 2*(-z_dot) + 4*(z_ref - z)
        thrust = (zref + g)*conversao_digital
        thrust = min(max(thrust,11000),60000)
        return thrust
    
    def compute_controller(self):
        counter = 0
        counter_loop = 0
        pose_old = None
        time_old = None

        while not rospy.is_shutdown():

            if not self.compute_controller_flag:
                self.rate.sleep()
                continue

            if not self.pose_is_true:
                counter += 1
                if counter == 4:
                    rospy.loginfo("It was'nt possible to find the pose of the crazyflie!")
                    counter = 0
                self.wait_rate.sleep()
                continue

            if self.is_landing:
                self.rate.sleep()
                continue
            
            pose = self.pose
            pose_time = self.pose_time

            if counter_loop == 0:
                takeoff_time = rospy.Time.now()
                initial_state = np.array([pose[0], pose[1], pose[2], 0, 0, 0])
                cf_Hinf = Hinf_control(initial_state, self.hz)
                self.desired_takeoff_pose = [self.pose[0], self.pose[1], self.pose[2]+self.desired_z_height, 0, 0, 0]
                counter_loop +=1

            # Compute velocity
            if pose_old is not None:
                dt = (pose_time - time_old).to_sec()
                if dt == 0:
                    self.rate.sleep()
                    continue
            
                vel = (np.array([pose[0], pose[1], pose[2]]) 
                        - np.array([pose_old[0], pose_old[1], pose_old[2]])) / dt
            else:
                vel = np.array([0.0, 0.0, 0.0])
            
            pose_old = pose
            time_old = pose_time

            # Get the current timestamp
            current_time = rospy.Time.now()

            # Calculate elapsed time
            elapsed_time = (current_time - takeoff_time).to_sec()

            angular_velocity = np.sqrt(0.5)
            x_radius = 1
            y_radius = 1

            if elapsed_time < 5:
                psi_desired = 0
                psi_dot_ref = 0
                
                desired = self.desired_takeoff_pose
            
            else:
                # Circle
                x = x_radius * np.cos(angular_velocity * elapsed_time)
                y = y_radius * np.sin(angular_velocity * elapsed_time)
                z = self.desired_z_height

                vx = -x_radius * angular_velocity * np.sin(angular_velocity * elapsed_time)
                vy = y_radius * angular_velocity * np.cos(angular_velocity * elapsed_time)
                vz = 0

                # # Line
                # x = x_radius * np.cos(angular_velocity * elapsed_time)
                # y = 0.95
                # z = self.desired_z_height

                # vx = -x_radius * angular_velocity * np.sin(angular_velocity * elapsed_time)
                # vy = 0
                # vz = 0

                # # Positoining
                # x = 1
                # y = 1
                # z = self.desired_z_height

                # vx = 0
                # vy = 0
                # vz = 0

                desired = [x, y, z, vx, vy, vz]
                psi_desired = np.arctan2(vy, vx)*0
                psi_dot_ref = -angular_velocity*0

            if self.goto_center:
                desired = [0, 0, self.desired_z_height, 0, 0, 0]

                # # Center of the line
                # desired = [0, 0.95, self.desired_z_height, 0, 0, 0]
                
                psi_desired = 0
                psi_dot_ref = 0
            
            self.desired = desired

            pose_Hinf = np.array([pose[0], pose[1], pose[2], vel[0], vel[1], vel[2]])
            self.pose_Hinf = pose_Hinf

            theta_Hinf, phi_Hinf, thrust_Hinf = cf_Hinf.compute_u(pose_Hinf, desired)

            #### to save
            theta_save, phi_save, thrust_save = self.conversions_crazyflie(theta_Hinf, phi_Hinf, thrust_Hinf)
            self.control_input = self.undo_conversions_crazyflie(theta_save, phi_save, thrust_save)

            rotation_matrix_theta_phi = np.array([[np.cos(pose[5]) , np.sin(pose[5])], 
                                                  [-np.sin(pose[5]), np.cos(pose[5])]])

            theta_phi = rotation_matrix_theta_phi @ np.array([theta_Hinf, phi_Hinf])

            theta_ref, phi_ref, thrust_ref = self.conversions_crazyflie(theta_phi[0], theta_phi[1], thrust_Hinf)

            psi_ref = self.compute_psi(psi_dot_ref, psi_desired)
            thrust_ref = self.compute_thrust(pose[2], vel[2], self.desired_z_height)

            self.theta_ref = -theta_ref + self.theta_trim 
            self.phi_ref = -phi_ref + self.phi_trim 
            self.psi_ref = psi_ref
            self.thrust_ref = thrust_ref

            self.compute_controller_is_true = True

    def prints(self, req):
        if not self.compute_controller_is_true:
            return

        if self.print_kalman:
            print(f'Measure : {np.round(self.pose_Hinf, 3)}')

        if self.print_controller:
            print(f'Hinf_controller : {[float(np.round(val, 3)) for val in [self.theta_ref, self.phi_ref, self.psi_ref, self.thrust_ref/60000]]}')

    def save_bag_timer(self, req):
        if not self.compute_controller_is_true:
            return

        if not self.is_bag_open:
            return
           
        pose_Hinf_msg = Vector3()
        pose_Hinf_msg.x, pose_Hinf_msg.y, pose_Hinf_msg.z = self.pose_Hinf[:3] 
        self.bag.write('position_Optitrack', pose_Hinf_msg)
        vel_Hinf_msg = Vector3()
        vel_Hinf_msg.x, vel_Hinf_msg.y, vel_Hinf_msg.z = self.pose_Hinf[3:] 
        self.bag.write('vel_Optitrack', vel_Hinf_msg)

        ### Desired position
        desired_position_msg = Vector3()
        desired_position_msg.x, desired_position_msg.y, desired_position_msg.z = self.desired[:3]
        self.bag.write('desired_position', desired_position_msg)
        ### Desired velocity
        desired_vel_msg = Vector3()
        desired_vel_msg.x, desired_vel_msg.y, desired_vel_msg.z = self.desired[3:]
        self.bag.write('desired_vel', desired_vel_msg)

        ### Control input ref
        control_input_msg = Vector3()
        control_input_msg.x, control_input_msg.y, control_input_msg.z = self.control_input
        self.bag.write('control_input', control_input_msg)

    def handle_trajectory(self, req):
        self.compute_controller_flag = True

        while not rospy.is_shutdown():
            if self.is_landing:
                self.rate.sleep()
                continue

            if not self.compute_controller_is_true:
                self.rate.sleep()
                continue

            theta_ref = self.theta_ref
            phi_ref = self.phi_ref
            psi_ref = self.psi_ref
            thrust_ref = self.thrust_ref

            ### Joystick
            if self.joy_is_true:
                if np.linalg.norm([self.joy.axes[4], self.joy.axes[3]]) > .1:
                    theta_ref = self.joy.axes[4] * self.ang_max
                    phi_ref = self.joy.axes[3] * self.ang_max

                if abs((self.joy.axes[2] - self.joy.axes[5])*0.5) > .1:
                    psi_ref = -((self.joy.axes[2] - self.joy.axes[5])*0.5)*100

                if abs(self.joy.axes[1]) > .1:
                    thrust_ref = (.5 + self.joy.axes[1]*.5)*60000
                    thrust_ref = min(max(thrust_ref,30000),60000)
            
            pub_controller = Twist()
            pub_controller.linear.x = theta_ref
            pub_controller.linear.y = phi_ref
            pub_controller.linear.z = thrust_ref
            pub_controller.angular.z = psi_ref
            if not self.dont_takeoff:
                self.control_publish.publish(pub_controller)
            self.rate.sleep()

if __name__ == '__main__':
    node = CrazyflieHinfNode()
    try:
        node.compute_controller()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(e)
        rospy.signal_shutdown("Parameter not set")