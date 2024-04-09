#!/usr/bin/env python3
import rospy

from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Empty, Bool
from std_srvs.srv import Empty

import numpy as np
import tf.transformations as tf_trans

import logging
from sensor_msgs.msg import Joy

import cflib
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.utils import uri_helper

from aurora_py.H_inf_control import Hinf_control
from crazyflie.crazyflie_controller.src.scripts.aurora_py.controllers import CrazyflieController, CrazyflieLQR
from aurora_py.kalman_filter import KalmanFilter

class MyNode:
    def __init__(self):
        rospy.init_node('crazyflie_node')

        # Initializing crazyflie
        cflib.crtp.init_drivers()
        self._cf = Crazyflie()

        # Crazyflie connection callbacks
        self._cf.connected.add_callback(self._connected)
        self._cf.disconnected.add_callback(self._disconnected)
        self._cf.connection_failed.add_callback(self._connection_failed)
        self._cf.connection_lost.add_callback(self._connection_lost)        

        uri_ = rospy.get_param('~uri', None)
        
        if not rospy.has_param('~uri'):
            raise Exception(f"Required parameter 'uri' not set")

        uri = uri_helper.uri_from_env(default=uri_)
        logging.basicConfig(level=logging.ERROR)

        # Opening crazyflie link
        self._cf.open_link(uri)

        # Setting Crazyflie parameters
        self._cf.param.set_value("stabilizer.controller", 2) 
        self._cf.param.set_value("stabilizer.estimator", 3)
        self._cf.param.set_value("ctrlMel.ki_m_z", 0)
        self._cf.param.set_value("ctrlMel.i_range_m_z", 0)
        
        self.cf_controller = CrazyflieController()
        
        # Flags
        self.emergency_flag_is_true = False
        self.emergency_button_pressed = False
        self.joy_is_true = False
        self.goto_center = False

        self.drone_is_landed = True # Flag to know if the drone is landed
        self.pose_is_true = None
        self.twist_is_true = False
        self.is_landing = False

        self.test_motors_flag = False

        self.zd = 0.75
        self.g = 9.81
        self.amax = 15
        self.max_angle = 15
        
        self.namespace = rospy.get_namespace()

        self.pose_subscriber = rospy.Subscriber(f"/vrpn_client_node{self.namespace}pose", 
                                                PoseStamped, 
                                                self.pose_callback)
        
        self.pose_subscriber = rospy.Subscriber(f"cmd_vel", 
                                                Twist, 
                                                self.publish_twist)
        
        self.pose_subscriber = rospy.Subscriber(f"cmd_vel_motors", 
                                                Twist, 
                                                self.cmd_vel_motors)
        
        # Subscriber for the joystick
        self.joystick_cmd = rospy.Subscriber("/joy", 
                                            Joy, 
                                            self.joy_callback)
        
        # Subscriber for the emergency flag
        self.emergency_subscriber = rospy.Subscriber("/emergency_flag", 
                                                     Bool, 
                                                     self.emergency_callback)
        
        self.kalman_pub = rospy.Publisher('/kalman_pub',
                                            Twist,
                                            queue_size=10)
        
        self.pose_pub = rospy.Publisher('/pose_pub',
                                            Twist,
                                            queue_size=10)
        
        self.takeoff_service = rospy.Service('takeoff', Empty, self.handle_takeoff)

        self.land_service = rospy.Service('land', Empty, self.handle_land)

        self.hz = 1/30
        self.rate = rospy.Rate(1/self.hz)
        
        rospy.loginfo("Crazyflie Node started")
    
    def cmd_vel_motors(self, msg):
        # self._cf.param.set_value("stabilizer.controller", msg.linear.x) # 2
        # self._cf.param.set_value("stabilizer.estimator", msg.linear.y) # 3

        self._cf.param.set_value("motorPowerSet.m1", int(msg.linear.x))
        self._cf.param.set_value("motorPowerSet.m2", int(msg.linear.y))
        self._cf.param.set_value("motorPowerSet.m3", int(msg.linear.z))
        self._cf.param.set_value("motorPowerSet.m4", int(msg.angular.x))
        self._cf.param.set_value("motorPowerSet.enable", 0)

        # self._cf.param.set_value("stabilizer.stop", msg.angular.z)
        # self._cf.param.set_value("usec.reset", msg.angular.z)
        
        # roll = msg.linear.y
        # pitch = msg.linear.x
        # thrust = msg.linear.z
        # yawrate = msg.angular.z
        # self._cf.commander.send_setpoint(-roll, pitch, -yawrate, int(thrust))
    
    def get_euler_from_quaternion(self, quaternion):
        return np.array(tf_trans.euler_from_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w]))

    def _connected(self, link_uri):
        """ This callback is called form the Crazyflie API when a Crazyflie
        has been connected and the TOCs have been downloaded."""
        rospy.loginfo(f'Crazyflie {link_uri} connected!')

        self._lg_vel = LogConfig(name='velocity in the drone frame', period_in_ms=1000/65)
        self._lg_vel.add_variable('kalman.statePX', 'float')
        self._lg_vel.add_variable('kalman.statePY', 'float')
        self._lg_vel.add_variable('kalman.statePZ', 'float')

        # Adding the configuration cannot be done until a Crazyflie is
        # connected, since we need to check that the variables we
        # would like to log are in the TOC.
        try:
            self._cf.log.add_config(self._lg_vel)
            # This callback will receive the data
            self._lg_vel.data_received_cb.add_callback(self._vel_log_data)
            # This callback will be called on errors
            self._lg_vel.error_cb.add_callback(self._vel_log_error)
            # Start the logging
            self._lg_vel.start()
        except KeyError as e:
            print('Could not start log configuration,'
                  '{} not found in TOC'.format(str(e)))
        except AttributeError:
            print('Could not add Stabilizer log config, bad configuration.')
        # Start a separate thread to do the motor test.
        # Do not hijack the calling thread!
        # Thread(target=self._ramp_motors).start()

    def _connection_failed(self, link_uri, msg):
        """Callback when connection initial connection fails (i.e no Crazyflie
        at the specified address)"""
        rospy.loginfo('Connection to %s failed: %s' % (link_uri, msg))

    def _connection_lost(self, link_uri, msg):
        """Callback when disconnected after a connection has been made (i.e
        Crazyflie moves out of range)"""
        rospy.loginfo('Connection to %s lost: %s' % (link_uri, msg))

    def _disconnected(self, link_uri):
        """Callback when the Crazyflie is disconnected (called in all cases)"""
        rospy.loginfo('Disconnected from %s' % link_uri)

    def _vel_log_error(self, logconf, msg):
        """Callback from the log API when an error occurs"""
        print('Error when logging %s: %s' % (logconf.name, msg))

    def _vel_log_data(self, timestamp, data, logconf):
        """Callback from a the log API when data arrives"""
        # print(f'[{timestamp}][{logconf.name}]: ', end='')
        for name, value in data.items():
            # print(f'{name}: {value:3.3f} ', end='')
            pass
        # print()

    # Emergency callback to handle emergency landing situations
    def emergency_callback(self, data):
        self.emergency_flag_is_true = True
        if data.data:
            self.emergency_button_pressed = True
            self.emergency_landing('Emergency landing')
    
    def emergency_landing(self, message):
        rospy.logerr(message)
        self.is_landing = True
        for _ in range(30):
            self._cf.commander.send_setpoint(0, 0, 0, 0)
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

            self.last_joy_update = rospy.get_time()

    def pose_callback(self, msg):
        if not self.pose_is_true:
            rospy.loginfo("Expose is on!")
            self.pose_is_true = True
        x, y, z = msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
        roll, pitch, yaw = self.get_euler_from_quaternion(msg.pose.orientation)

        self.pose = np.array([x, y, z, roll, pitch, yaw])
        self.pose_time = msg.header.stamp
        # self._cf.extpos.send_extpose(x, y, z, msg.pose.orientation.x, msg.pose.orientation.y,  msg.pose.orientation.z,  msg.pose.orientation.w)

    def publish_twist(self, msg):
        if not self.twist_is_true:
            self.twist_is_true = True

            # Unlock startup thrust protection
            self._cf.commander.send_setpoint(0, 0, 0, 0)

        if self.is_landing:
            return
        
        roll = msg.linear.y
        pitch = msg.linear.x
        thrust = msg.linear.z
        yawrate = msg.angular.z
        self._cf.commander.send_setpoint(-roll, pitch, -yawrate, int(thrust))

    def handle_takeoff(self, req):
        rospy.loginfo("Takeoff requested!")
        pose_old = None
        time_old = None
        takeoff_time = rospy.Time.now()

        cf_LQR = CrazyflieLQR()

        self.initial_pose = self.pose
        psi_ref_max = 100

        if not self.pose_is_true:
            rospy.loginfo("It was'nt possible to find the pose of the crazyflie!")
            return []

        if not self.drone_is_landed:
            rospy.loginfo("The crazyflie is not landed!")
            return []

        # Unlock startup thrust protection
        self._cf.commander.send_setpoint(0, 0, 0, 0)
        pose = self.pose
        rotation_matrix_theta_phi = np.array([[np.cos(pose[5]) , np.sin(pose[5])], 
                                                  [-np.sin(pose[5]), np.cos(pose[5])]])
        
        position = rotation_matrix_theta_phi @ np.array([pose[0], pose[1]])
        pose_hinf = [position[0], position[1], pose[2], 0, 0, 0]

        hinf_control = Hinf_control(initial_state=pose_hinf, Ts=self.hz)
        kalman_filter = KalmanFilter(initial_state=pose_hinf)

        # Takeoff loop
        while not rospy.is_shutdown():
            if self.is_landing:
                self.rate.sleep()
                continue
            
            pose = self.pose
            pose_time = self.pose_time
            # Compute velocity
            if pose_old is not None:
                dt = (pose_time - time_old).to_sec()
                if dt == 0:
                    self.rate.sleep()
                    continue
            
                vel = (np.array([pose[0], pose[1], pose[2]]) 
                        - np.array([pose_old[0], pose_old[1], pose_old[2]])) / dt
                ang_vel = (np.array([pose[3], pose[4], pose[5]]) 
                        - np.array([pose_old[3], pose_old[4], pose_old[5]])) / dt
            else:
                vel = np.array([0.0, 0.0, 0.0])
                ang_vel = np.array([0.0, 0.0, 0.0])
            
            pose_old = pose
            time_old = pose_time

            # Get the current timestamp
            current_time = rospy.Time.now()

            # Calculate elapsed time
            elapsed_time = (current_time - takeoff_time).to_sec()

            angular_velocity = np.sqrt(0.5)
            x_radius = 1
            y_radius = 1

            if elapsed_time < 3:
                self.desired_pose = [0, 0, 1, 
                                    0, 0, 0]
                
                psi_desired = 0
                psi_dot_ref = 0
                
                desired = [0, 0, 1, 0, 0, 0]
            
            else:
                x = x_radius * np.cos(angular_velocity * elapsed_time)
                y = y_radius * np.sin(angular_velocity * elapsed_time)
                z = 1

                vx = -x_radius * angular_velocity * np.sin(angular_velocity * elapsed_time)
                vy = y_radius * angular_velocity * np.cos(angular_velocity * elapsed_time)
                vz = 0

                desired = [x, y, z, vx, vy, vz]
                psi_desired = np.arctan2(vy, vx)*0
                psi_dot_ref = -angular_velocity*0

                # ax = -x_radius * angular_velocity**2 * np.cos(angular_velocity * elapsed_time)
                # ay = -y_radius * angular_velocity**2 * np.sin(angular_velocity * elapsed_time)
                # az = 0

            if self.goto_center:
                desired = [0, 0, 1, 0, 0, 0]
                psi_desired = 0
                psi_dot_ref = 0
                
            psi_til = psi_desired - pose[5]
            if np.abs(psi_til) > np.pi:
                if psi_til > 0:
                    psi_til = psi_til - 2*np.pi
                else:
                    psi_til = psi_til + 2*np.pi

            psi_ref = psi_dot_ref + 5*psi_til
            psi_ref = min(max(psi_ref,-1),1)
            psi_ref = psi_ref*psi_ref_max

            states_kalman = kalman_filter.update([pose[0], pose[1], pose[2], vel[0], vel[1], vel[2]])

            position = rotation_matrix_theta_phi @ np.array([pose[0], pose[1]])
            vel_LQR = rotation_matrix_theta_phi @ np.array([vel[0], vel[1]])

            position_kalman = rotation_matrix_theta_phi @ np.array(states_kalman)[0:2]
            vel_kalman = rotation_matrix_theta_phi @ np.array(states_kalman)[3:5]
            
            pose_LQR = np.array([position[0], position[1], pose[2], vel_LQR[0], vel_LQR[1], vel[2]])
            pose_kalmam = np.array([position_kalman[0], position_kalman[1], states_kalman[2], vel_kalman[0], vel_kalman[1], states_kalman[2]])
            
            # K = Twist()
            # K.linear.x = pose_kalmam[0]
            # K.linear.y = pose_kalmam[1]
            # K.linear.z = pose_kalmam[2]
            # K.angular.x = pose_kalmam[3]
            # K.angular.y  = pose_kalmam[4]
            # K.angular.z  = pose_kalmam[5]
            # self.kalman_pub.publish(K)

            # P = Twist()
            # P.linear.x = pose_LQR[0]
            # P.linear.y = pose_LQR[1]
            # P.linear.z = pose_LQR[2]
            # P.angular.x = pose_LQR[3]
            # P.angular.y  = pose_LQR[4]
            # P.angular.z  = pose_LQR[5]
            # self.pose_pub.publish(P)
            
            theta_LQR, phi_LQR, thrust_LQR = cf_LQR.compute_u(pose_LQR, desired)
            # theta_Hinf, phi_Hinf, thrust_Hinf = hinf_control.compute_u(pose_LQR, desired)
            

            # print(f'Measure : {np.round(pose_LQR, 3)}')
            # try:
            #     print(f'Kalman: {np.round(pose_kalmam, 3)}')
            # except:
            #     pass

            # print()

            ### Joystick
            if self.joy_is_true:
                if np.linalg.norm([self.joy.axes[4], self.joy.axes[3]]) > .1:
                    theta_ref = self.joy.axes[4] * self.max_angle
                    theta_LQR = theta_ref
                    phi_ref = self.joy.axes[3] * self.max_angle
                    phi_LQR = phi_ref

                if abs((self.joy.axes[2] - self.joy.axes[5])*0.5) > .1:
                    psi_ref = -((self.joy.axes[2] - self.joy.axes[5])*0.5)*100

                if abs(self.joy.axes[1]) > .1:
                    thrust = (.5 + self.joy.axes[1]*.5)*60000
                    thrust = min(max(thrust,30000),60000)
                    thrust_LQR = thrust

            # print(f'LQR : {np.round(np.array([theta_LQR, phi_LQR, int(thrust_LQR)/60000]), 2)}')
            # try:
            #     print(f'Hinf: {np.round(np.array([theta_Hinf, phi_Hinf, int(thrust_Hinf)/60000]), 2)}')
            # except:
            #     pass

            self._cf.commander.send_setpoint(-phi_LQR, theta_LQR, -psi_ref, int(thrust_LQR))
            self.rate.sleep()
        
        self.drone_is_landed = False
        return []
    
    def Rot_wb(self, theta, phi, psi):
        # Rotation matrices for each axis
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(theta), -np.sin(theta)],
                        [0, np.sin(theta), np.cos(theta)]])

        R_y = np.array([[np.cos(phi), 0, np.sin(phi)],
                        [0, 1, 0],
                        [-np.sin(phi), 0, np.cos(phi)]])

        R_z = np.array([[np.cos(psi), -np.sin(psi), 0],
                        [np.sin(psi), np.cos(psi), 0],
                        [0, 0, 1]])

        # Combined rotation matrix, R = R_z * R_y * R_x
        R = np.dot(R_z, np.dot(R_y, R_x))
        return R

    def handle_land(self, req):
        rospy.loginfo("Land requested")

        self.is_landing = True
        while not rospy.is_shutdown():
            self._cf.commander.send_setpoint(0, 0, 0, 0)

        return []

if __name__ == '__main__':
    node = MyNode()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(e)
        rospy.signal_shutdown("Parameter not set")

    node._cf.close_link()