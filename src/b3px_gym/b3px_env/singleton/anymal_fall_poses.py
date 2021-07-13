import numpy as np
from numpy import pi

class AnymalConfig():
    def __init__(self, fallRecoverySet=0):
    #keypose initialization for fall recovery
        if fallRecoverySet == 0:
            self.key_pose = []
            # nominal
            base_pos_nom = [0, 0, 0.5]
            base_orn_nom = self.euler_to_quat(0, 0, 0)  # euler_to_quat(-1.57,0,0)#euler_to_quat(0,1.57,0)
            
            q_nom = [-0.0,     20*pi/180, -40*pi/180,
                    -0.0,     20*pi/180, -40*pi/180,
                    -0.0,    -20*pi/180,  40*pi/180,  
                    -0.0,    -20*pi/180,  40*pi/180,]
            self.key_pose.append([base_pos_nom, base_orn_nom, q_nom])

            # # stand tall
            # base_pos_nom = [0, 0, 0.65]
            # base_orn_nom = self.euler_to_quat(0, 0, 0)  # euler_to_quat(-1.57,0,0)#euler_to_quat(0,1.57,0)
            
            # q_nom = [-0.0,     10*pi/180, -20*pi/180,
            #          -0.0,     10*pi/180, -20*pi/180,
            #          -0.0,    -10*pi/180,  20*pi/180,  
            #          -0.0,    -10*pi/180,  20*pi/180,]
            # self.key_pose.append([base_pos_nom, base_orn_nom, q_nom])

            # # stand spread
            # base_pos_nom = [0, 0, 0.55]
            # base_orn_nom = self.euler_to_quat(0, 0, 0)  # euler_to_quat(-1.57,0,0)#euler_to_quat(0,1.57,0)
            
            # q_nom = [-20.0*pi/180.,     40*pi/180., -40*pi/180.,
            #          20.0*pi/180.,     40*pi/180., -40*pi/180.,
            #          -20.0*pi/180.,    -40*pi/180.,  40*pi/180.,  
            #          20.0*pi/180.,    -40*pi/180.,  40*pi/180.,]

            # self.key_pose.append([base_pos_nom, base_orn_nom, q_nom])

            # crouch
            base_pos_nom = [0, 0, 0.2]
            base_orn_nom = self.euler_to_quat(0, 0, 0)  # euler_to_quat(-1.57,0,0)#euler_to_quat(0,1.57,0)
            
            q_nom = [-0.0,     70*pi/180, -120*pi/180,
                    -0.0,     70*pi/180, -120*pi/180,
                    -0.0,    -70*pi/180,  120*pi/180,  
                    -0.0,    -70*pi/180,  120*pi/180,]
            self.key_pose.append([base_pos_nom, base_orn_nom, q_nom])

            # crouch spread
            base_pos_nom = [0, 0, 0.2]
            base_orn_nom = self.euler_to_quat(0, 0, 0)  # euler_to_quat(-1.57,0,0)#euler_to_quat(0,1.57,0)
            
            q_nom = [-0.0,     -90*pi/180, 0*pi/180,
                    -0.0,     -90*pi/180, 0*pi/180,
                    -0.0,      90*pi/180, 0*pi/180,  
                    -0.0,      90*pi/180, 0*pi/180,]
            self.key_pose.append([base_pos_nom, base_orn_nom, q_nom])

            # Robot constantly flips on its back not need for initialization


            #belly left
            base_pos_nom = [0, 0, 0.4]
            base_orn_nom = self.euler_to_quat(-0*pi/180, 0.01, 0)  # euler_to_quat(-1.57,0,0)#euler_to_quat(0,1.57,0)
            
            q_nom = [ 00.*pi/180,     70*pi/180, -120*pi/180,
                    30.*pi/180,     30*pi/180, -90*pi/180,
                    00.*pi/180,    -70*pi/180,  120*pi/180,  
                    30.*pi/180,    -30*pi/180,  90*pi/180]
            self.key_pose.append([base_pos_nom, base_orn_nom, q_nom])

            #belly right
            base_pos_nom = [0, 0, 0.4]
            base_orn_nom = self.euler_to_quat(-0*pi/180, 0.01, 0)  # euler_to_quat(-1.57,0,0)#euler_to_quat(0,1.57,0)
            
            q_nom = [-30.*pi/180,     30*pi/180, -90*pi/180,
                    00.*pi/180,     70*pi/180, -120*pi/180,
                    -30.*pi/180,    -30*pi/180,  90*pi/180,  
                    00.*pi/180,    -70*pi/180,  120*pi/180]
            self.key_pose.append([base_pos_nom, base_orn_nom, q_nom])

            #belly front
            base_pos_nom = [0, 0, 0.5]
            base_orn_nom = self.euler_to_quat(-0*pi/180, 0.01, 0)  # euler_to_quat(-1.57,0,0)#euler_to_quat(0,1.57,0)
            
            q_nom = [-00.*pi/180,     70*pi/180, -120*pi/180,
                    00.*pi/180,     70*pi/180, -120*pi/180,
                    -00.*pi/180,    -20*pi/180,  40*pi/180,  
                    00.*pi/180,    -20*pi/180,  40*pi/180]
            self.key_pose.append([base_pos_nom, base_orn_nom, q_nom])

            #belly back
            base_pos_nom = [0, 0, 0.5]
            base_orn_nom = self.euler_to_quat(-0*pi/180, 0.01, 0)  # euler_to_quat(-1.57,0,0)#euler_to_quat(0,1.57,0)
            
            q_nom = [-00.*pi/180,     20*pi/180, -40*pi/180,
                    00.*pi/180,     20*pi/180, -40*pi/180,
                    -00.*pi/180,    -70*pi/180,  120*pi/180,  
                    00.*pi/180,    -70*pi/180,  120*pi/180]
            self.key_pose.append([base_pos_nom, base_orn_nom, q_nom])


        elif fallRecoverySet==2:
            self.key_pose = []
            # nominal
            base_pos_nom = [0, 0, 0.5]
            base_orn_nom = self.euler_to_quat(0, 0, 0)  # euler_to_quat(-1.57,0,0)#euler_to_quat(0,1.57,0)
            
            q_nom = [-0.0,     20*pi/180, -40*pi/180,
                    -0.0,     20*pi/180, -40*pi/180,
                    -0.0,    -20*pi/180,  40*pi/180,  
                    -0.0,    -20*pi/180,  40*pi/180,]
            self.key_pose.append([base_pos_nom, base_orn_nom, q_nom])

            # # stand tall
            # base_pos_nom = [0, 0, 0.65]
            # base_orn_nom = self.euler_to_quat(0, 0, 0)  # euler_to_quat(-1.57,0,0)#euler_to_quat(0,1.57,0)
            
            # q_nom = [-0.0,     10*pi/180, -20*pi/180,
            #          -0.0,     10*pi/180, -20*pi/180,
            #          -0.0,    -10*pi/180,  20*pi/180,  
            #          -0.0,    -10*pi/180,  20*pi/180,]
            # self.key_pose.append([base_pos_nom, base_orn_nom, q_nom])

            # # stand spread
            # base_pos_nom = [0, 0, 0.55]
            # base_orn_nom = self.euler_to_quat(0, 0, 0)  # euler_to_quat(-1.57,0,0)#euler_to_quat(0,1.57,0)
            
            # q_nom = [-20.0*pi/180.,     40*pi/180., -40*pi/180.,
            #          20.0*pi/180.,     40*pi/180., -40*pi/180.,
            #          -20.0*pi/180.,    -40*pi/180.,  40*pi/180.,  
            #          20.0*pi/180.,    -40*pi/180.,  40*pi/180.,]

            # self.key_pose.append([base_pos_nom, base_orn_nom, q_nom])

            # crouch
            base_pos_nom = [0, 0, 0.2]
            base_orn_nom = self.euler_to_quat(0, 0, 0)  # euler_to_quat(-1.57,0,0)#euler_to_quat(0,1.57,0)
            
            q_nom = [-0.0,     70*pi/180, -120*pi/180,
                    -0.0,     70*pi/180, -120*pi/180,
                    -0.0,    -70*pi/180,  120*pi/180,  
                    -0.0,    -70*pi/180,  120*pi/180,]
            self.key_pose.append([base_pos_nom, base_orn_nom, q_nom])

            # crouch spread
            base_pos_nom = [0, 0, 0.2]
            base_orn_nom = self.euler_to_quat(0, 0, 0)  # euler_to_quat(-1.57,0,0)#euler_to_quat(0,1.57,0)
            
            q_nom = [-0.0,     -90*pi/180, 0*pi/180,
                    -0.0,     -90*pi/180, 0*pi/180,
                    -0.0,      90*pi/180, 0*pi/180,  
                    -0.0,      90*pi/180, 0*pi/180,]
            self.key_pose.append([base_pos_nom, base_orn_nom, q_nom])

            # Robot constantly flips on its back not need for initialization


            #belly left
            base_pos_nom = [0, 0, 0.4]
            base_orn_nom = self.euler_to_quat(-0*pi/180, 0.01, 0)  # euler_to_quat(-1.57,0,0)#euler_to_quat(0,1.57,0)
            
            q_nom = [ 00.*pi/180,     70*pi/180, -120*pi/180,
                    30.*pi/180,     30*pi/180, -90*pi/180,
                    00.*pi/180,    -70*pi/180,  120*pi/180,  
                    30.*pi/180,    -30*pi/180,  90*pi/180]
            self.key_pose.append([base_pos_nom, base_orn_nom, q_nom])

            #belly right
            base_pos_nom = [0, 0, 0.4]
            base_orn_nom = self.euler_to_quat(-0*pi/180, 0.01, 0)  # euler_to_quat(-1.57,0,0)#euler_to_quat(0,1.57,0)
            
            q_nom = [-30.*pi/180,     30*pi/180, -90*pi/180,
                    00.*pi/180,     70*pi/180, -120*pi/180,
                    -30.*pi/180,    -30*pi/180,  90*pi/180,  
                    00.*pi/180,    -70*pi/180,  120*pi/180]
            self.key_pose.append([base_pos_nom, base_orn_nom, q_nom])

            #belly front
            base_pos_nom = [0, 0, 0.5]
            base_orn_nom = self.euler_to_quat(-0*pi/180, 0.01, 0)  # euler_to_quat(-1.57,0,0)#euler_to_quat(0,1.57,0)
            
            q_nom = [-00.*pi/180,     70*pi/180, -120*pi/180,
                    00.*pi/180,     70*pi/180, -120*pi/180,
                    -00.*pi/180,    -20*pi/180,  40*pi/180,  
                    00.*pi/180,    -20*pi/180,  40*pi/180]
            self.key_pose.append([base_pos_nom, base_orn_nom, q_nom])

            #belly back
            base_pos_nom = [0, 0, 0.5]
            base_orn_nom = self.euler_to_quat(-0*pi/180, 0.01, 0)  # euler_to_quat(-1.57,0,0)#euler_to_quat(0,1.57,0)
            
            q_nom = [-00.*pi/180,     20*pi/180, -40*pi/180,
                    00.*pi/180,     20*pi/180, -40*pi/180,
                    -00.*pi/180,    -70*pi/180,  120*pi/180,  
                    00.*pi/180,    -70*pi/180,  120*pi/180]
            self.key_pose.append([base_pos_nom, base_orn_nom, q_nom])

            #back left
            base_pos_nom = [0, 0, 0.25]
            base_orn_nom = self.euler_to_quat(-2.2, 0.01, 0)  # euler_to_quat(-1.57,0,0)#euler_to_quat(0,1.57,0)
            
            q_nom = [0, 0.74, -1.69,  
                     0, 0.74, -1.69,
                     0, -0.74, 1.69,
                     0, -0.74, 1.69]
            self.key_pose.append([base_pos_nom, base_orn_nom, q_nom])

            #back right
            base_pos_nom = [0, 0, 0.25]
            base_orn_nom = self.euler_to_quat(2.2, 0.01, 0)  # euler_to_quat(-1.57,0,0)#euler_to_quat(0,1.57,0)
            
            q_nom = [0, 0.74, -1.69,  
                     0, 0.74, -1.69,
                     0, -0.74, 1.69,
                     0, -0.74, 1.69]
            self.key_pose.append([base_pos_nom, base_orn_nom, q_nom])

        elif fallRecoverySet==1:
            self.key_pose = []
            # nominal
            base_pos_nom = [0, 0, 0.5]
            base_orn_nom = self.euler_to_quat(0, 0, 0)  # euler_to_quat(-1.57,0,0)#euler_to_quat(0,1.57,0)
            
            q_nom = [-0.0,     20*pi/180, -40*pi/180,
                    -0.0,     20*pi/180, -40*pi/180,
                    -0.0,    -20*pi/180,  40*pi/180,  
                    -0.0,    -20*pi/180,  40*pi/180,]
            self.key_pose.append([base_pos_nom, base_orn_nom, q_nom])

            # crouch
            base_pos_nom = [0, 0, 0.2]
            base_orn_nom = self.euler_to_quat(0, 0, 0)  # euler_to_quat(-1.57,0,0)#euler_to_quat(0,1.57,0)
            
            q_nom = [-0.0,     70*pi/180, -120*pi/180,
                    -0.0,     70*pi/180, -120*pi/180,
                    -0.0,    -70*pi/180,  120*pi/180,  
                    -0.0,    -70*pi/180,  120*pi/180,]
            self.key_pose.append([base_pos_nom, base_orn_nom, q_nom])

            # crouch spread
            base_pos_nom = [0, 0, 0.2]
            base_orn_nom = self.euler_to_quat(0, 0, 0)  # euler_to_quat(-1.57,0,0)#euler_to_quat(0,1.57,0)
            
            q_nom = [-0.0,     -90*pi/180, 0*pi/180,
                    -0.0,     -90*pi/180, 0*pi/180,
                    -0.0,      90*pi/180, 0*pi/180,  
                    -0.0,      90*pi/180, 0*pi/180,]
            self.key_pose.append([base_pos_nom, base_orn_nom, q_nom])
            #back
            base_pos_nom = [0, 0, 0.3]
            base_orn_nom = self.euler_to_quat(3.14, 0, 0)  # euler_to_quat(-1.57,0,0)#euler_to_quat(0,1.57,0)
            
            q_nom = [0,  0.74, -1.69, 
                     0,  0.74, -1.69, 
                     0, -0.74, 1.69, 
                     0, -0.74, 1.69]
            self.key_pose.append([base_pos_nom, base_orn_nom, q_nom])

            #back left
            base_pos_nom = [0, 0, 0.25]
            base_orn_nom = self.euler_to_quat(-2.2, 0.01, 0)  # euler_to_quat(-1.57,0,0)#euler_to_quat(0,1.57,0)
            
            q_nom = [0, 0.74, -1.69,  
                     0, 0.74, -1.69,
                     0, -0.74, 1.69,
                     0, -0.74, 1.69]
            self.key_pose.append([base_pos_nom, base_orn_nom, q_nom])

            #back right
            base_pos_nom = [0, 0, 0.25]
            base_orn_nom = self.euler_to_quat(2.2, 0.01, 0)  # euler_to_quat(-1.57,0,0)#euler_to_quat(0,1.57,0)
            
            q_nom = [0, 0.74, -1.69,  
                     0, 0.74, -1.69,
                     0, -0.74, 1.69,
                     0, -0.74, 1.69]
            self.key_pose.append([base_pos_nom, base_orn_nom, q_nom])

            #right
            base_pos_nom = [0, 0, 0.3]
            base_orn_nom = self.euler_to_quat(1.57, 0, 0)  # euler_to_quat(-1.57,0,0)#euler_to_quat(0,1.57,0)
            q_nom = [0, 0.74, -1.69,  
                     0, 0.74, -1.69,
                     0, -0.74, 1.69,
                     0, -0.74, 1.69]
            self.key_pose.append([base_pos_nom, base_orn_nom, q_nom])

            #left
            base_pos_nom = [0, 0, 0.3]
            base_orn_nom = self.euler_to_quat(-1.57, 0, 0)  # euler_to_quat(-1.57,0,0)#euler_to_quat(0,1.57,0)
            q_nom = [-0.0, 0.74, -1.69,  
                     0.0, 0.74, -1.69,
                     -0.0, -0.74, 1.69,
                     0.0, -0.74, 1.69]
            self.key_pose.append([base_pos_nom, base_orn_nom, q_nom])
        else:
            print("Not defined: %d" % fallRecoverySet)
            assert 3 == 4
    def euler_to_quat(self, roll, pitch, yaw):  # rad
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)

        w = cy * cr * cp + sy * sr * sp
        x = cy * sr * cp - sy * cr * sp
        y = cy * cr * sp + sy * sr * cp
        z = sy * cr * cp - cy * sr * sp

        return [x, y, z, w]