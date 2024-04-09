#!/usr/bin/env python3
import rospy

from geometry_msgs.msg import  Twist

import cflib
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.utils import uri_helper

class CrazyflieServerNode:
    def __init__(self):
        rospy.init_node('crazyflie_server_node')

        ################### CRAZYFLIE SETUP #######################
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

        # Opening crazyflie link
        uri = uri_helper.uri_from_env(default=uri_)
        self._cf.open_link(uri)

        # Unlock startup thrust protection
        self._cf.commander.send_setpoint(0, 0, 0, 0)

        ################### CRAZYFLIE PARAMETERS #######################

        # Setting Crazyflie parameters
        self._cf.param.set_value("stabilizer.controller", 2) 
        self._cf.param.set_value("stabilizer.estimator", 3)
        self._cf.param.set_value("ctrlMel.ki_m_z", 0)
        self._cf.param.set_value("ctrlMel.i_range_m_z", 0)

        ################### CRAZYFLIE TOPICS #######################
        self.pose_subscriber = rospy.Subscriber(f"cmd_vel", 
                                                Twist, 
                                                self.publish_twist)

    def _connected(self, link_uri):
        """ This callback is called form the Crazyflie API when a Crazyflie
        has been connected and the TOCs have been downloaded."""
        rospy.loginfo(f'Crazyflie {link_uri} connected!')

        self._lg_vel = LogConfig(name='velocity in the drone frame', period_in_ms=1000/65)
        self._lg_vel.add_variable('kalman.statePX', 'float')
        self._lg_vel.add_variable('kalman.statePY', 'float')
        self._lg_vel.add_variable('kalman.statePZ', 'float')

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

    def publish_twist(self, msg):        
        roll = msg.linear.y
        pitch = msg.linear.x
        thrust = msg.linear.z
        yawrate = msg.angular.z
        self._cf.commander.send_setpoint(-roll, pitch, -yawrate, int(thrust))

if __name__ == '__main__':
    node = CrazyflieServerNode()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(e)
        rospy.signal_shutdown("Parameter not set")

    node._cf.close_link()