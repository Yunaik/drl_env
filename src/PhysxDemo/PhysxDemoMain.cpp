/**
 * License: Bullet3 license
 * Author: Avik De <avikde@gmail.com>
 */
#include <map>
#include <string>
#include <iostream>
#include <stdio.h>
#include "../Utils/b3Clock.h"
#include "SharedMemory/PhysicsClientC_API.h"
#include "Bullet3Common/b3Vector3.h"
#include "Bullet3Common/b3Quaternion.h"
#include "SharedMemory/SharedMemoryInProcessPhysicsC_API.h"
#include "physx/PhysXServerCommandProcessor.h"
#include "physx/PhysXUrdfImporter.h"
#include "physx/URDF2PhysX.h"
#include "physx/PhysXUserData.h"
#include "PhysicsDirect.h"
#include "physx/PhysXC_API.h"
#include "../Importers/ImportURDFDemo/urdfStringSplit.h"
#include "../SharedMemory/PhysicsClientC_API.h"
#include "PhysicsServerCommandProcessor.h"

#include <cmath>
#include <vector>
#include <chrono>

extern const int CONTROL_RATE;
const int CONTROL_RATE = 500;
std::vector<int> bodies;

// Bullet globals
b3PhysicsClientHandle kPhysClient = 0;

const char * laikago = "/home/syslot/.pyenv/versions/3.6.7/envs/dev/lib/python3.6/site-packages/pybullet-2.4.3-py3.6-linux-x86_64.egg/pybullet_data/laikago/laikago.urdf";
const char * ground = "/home/syslot/.pyenv/versions/3.6.7/envs/dev/lib/python3.6/site-packages/pybullet-2.4.3-py3.6-linux-x86_64.egg/pybullet_data/plane100.urdf";

const b3Scalar FIXED_TIMESTEP = 1.0 / ((b3Scalar)CONTROL_RATE);

PhysicsDirect* init(){

    char * options = "--numCores=9 --solver=pgs --gpu=1";

    char** argv = urdfStrSplit(options, " ");
	int argc = urdfStrArrayLen(argv);

    PhysXServerCommandProcessor* sdk = new PhysXServerCommandProcessor(argc,argv);
//    PhysicsServerCommandProcessor* sdk = new PhysicsServerCommandProcessor;

	PhysicsDirect* sm = new PhysicsDirect(sdk, true);
	bool connected;
	connected = sm->connect();

	return sm;
}

int start_log(PhysicsDirect * sm, const char * log_name){


    int loggingType = b3StateLoggingType::STATE_LOGGING_PROFILE_TIMINGS;

    b3SharedMemoryCommandHandle commandHandle;
    commandHandle = b3StateLoggingCommandInit((b3PhysicsClientHandle)sm);

    b3StateLoggingStart(commandHandle, loggingType, log_name);

    b3SharedMemoryStatusHandle statusHandle = b3SubmitClientCommandAndWaitStatus((b3PhysicsClientHandle)sm, commandHandle);
    int statusType = b3GetStatusType(statusHandle);

    if (statusType == CMD_STATE_LOGGING_START_COMPLETED)
        return b3GetStatusLoggingUniqueId(statusHandle);

    return 0;
}

void stop_log(PhysicsDirect *sm, int logid){

    b3SharedMemoryCommandHandle commandHandle;
    commandHandle = b3StateLoggingCommandInit((b3PhysicsClientHandle)sm);
    b3StateLoggingStop(commandHandle, logid);
    b3SharedMemoryStatusHandle  statusHandle = b3SubmitClientCommandAndWaitStatus((b3PhysicsClientHandle)sm, commandHandle);
    int statusType = b3GetStatusType(statusHandle);
}

void render(PhysicsDirect *){
    	// Load Egl Render
//    {
//        char * pluginPath = "/home/syslot/DevSpace/bullet3/bin/libpybullet_eglRendererPlugin_gmake_x64_release.so";
//        b3SharedMemoryCommandHandle command = b3CreateCustomCommand((b3PhysicsClientHandle)sm);
//	    b3SharedMemoryStatusHandle statusHandle = 0;
//
//	    b3CustomCommandLoadPlugin(command, pluginPath);
//	    statusHandle = b3SubmitClientCommandAndWaitStatus((b3PhysicsClientHandle)sm, command);
//
//	    int statusType = -1;
//	    statusType = b3GetStatusPluginUniqueId(statusHandle);
//	    std::cout << "load library status" << statusType << std::endl;
//    }

}


int loadUrdf(PhysicsDirect * sm, const char * urdfFilename, b3Vector3 pos = b3MakeVector3(0,0,0), b3Vector4 rpy =
        b3MakeVector4(0,0,0,1)) {

        int flags = 0;

        b3SharedMemoryStatusHandle statusHandle;
        int statusType;
        b3SharedMemoryCommandHandle command = b3LoadUrdfCommandInit((b3PhysicsClientHandle) sm, urdfFilename);


        b3LoadUrdfCommandSetFlags(command, flags);
        b3LoadUrdfCommandSetStartPosition(command, pos.getX(), pos.getY(), pos.getZ());
        b3LoadUrdfCommandSetStartOrientation(command, rpy.getX(), rpy.getY(), rpy.getZ(), rpy.getW());

        statusHandle = b3SubmitClientCommandAndWaitStatus((b3PhysicsClientHandle) sm, command);
        statusType = b3GetStatusType(statusHandle);

        if (statusType != CMD_URDF_LOADING_COMPLETED) {
            std::cout << "load urdf failed!" << std::endl;
        }

        int uid = b3GetStatusBodyIndex(statusHandle);
        bodies.push_back(uid);

        return uid;
}

void stepsimulate(PhysicsDirect *sm, int step) {

    for (int i = 0; i < step; i++) {
        b3SharedMemoryStatusHandle statusHandle;
        int statusType;

        if (b3CanSubmitCommand((b3PhysicsClientHandle) sm)) {
            statusHandle = b3SubmitClientCommandAndWaitStatus(
                    (b3PhysicsClientHandle) sm,
                    b3InitStepSimulationCommand((b3PhysicsClientHandle) sm)
            );
            statusType = b3GetStatusType(statusHandle);
//            printf("%d\n", statusType);
        }
    }

}


int getNumofJoints(PhysicsDirect* sm, int uid){
    return b3GetNumJoints((b3PhysicsClientHandle)sm, uid);
}

b3JointInfo getJointInfo(PhysicsDirect *sm, int uid, int jointindex){

    b3JointInfo info;
    b3GetJointInfo((b3PhysicsClientHandle )sm, uid, jointindex, &info);
    return info;
}

b3JointSensorState getJointState(PhysicsDirect *sm, int uid, int jointindex){
    int status_type = 0;
    b3SharedMemoryCommandHandle cmd_handle;
    b3SharedMemoryStatusHandle status_handle;
    cmd_handle = b3RequestActualStateCommandInit((b3PhysicsClientHandle)sm, uid);

    status_handle = b3SubmitClientCommandAndWaitStatus((b3PhysicsClientHandle)sm, cmd_handle);

    b3JointSensorState sensorState;
    b3GetJointState((b3PhysicsClientHandle)sm, status_handle, jointindex, &sensorState);

    return sensorState;
}

void getAllJointState(PhysicsDirect *sm){
    for(auto uid : bodies){
        for(int i=0;i<12;i++)
            getJointState(sm, uid, i);
    }
}

void setJointState(PhysicsDirect *sm, int uid, int jointindex){
    int status_type=0;
    b3SharedMemoryCommandHandle cmd_handle = b3JointControlCommandInit2((b3PhysicsClientHandle)sm, uid, CONTROL_MODE_POSITION_VELOCITY_PD);
    float kp = 0.1, kd=10;
    float force = 100000.0;

    b3JointControlSetDesiredPosition(cmd_handle, jointindex,
													 0);
					b3JointControlSetKp(cmd_handle, jointindex, kp);
					b3JointControlSetDesiredVelocity(cmd_handle, jointindex,0);
					b3JointControlSetKd(cmd_handle, jointindex, kd);
					b3JointControlSetMaximumForce(cmd_handle, jointindex, force);

}

void setAllJointState(PhysicsDirect *sm){
    for(auto uid: bodies){
        for(int i=0;i<12;i++)
            setJointState(sm, uid, i);
    }
}


void stepfuntion(PhysicsDirect *sm, int step){
        for (int i = 0; i < step; i++) {
        b3SharedMemoryStatusHandle statusHandle;
        int statusType;

        setAllJointState(sm);

        if (b3CanSubmitCommand((b3PhysicsClientHandle) sm)) {
            statusHandle = b3SubmitClientCommandAndWaitStatus(
                    (b3PhysicsClientHandle) sm,
                    b3InitStepSimulationCommand((b3PhysicsClientHandle) sm)
            );
            statusType = b3GetStatusType(statusHandle);
        }

        getAllJointState(sm);
    }

}

void once(int sum){

    auto begin = std::chrono::high_resolution_clock::now();

    std::cout << "start : " <<sum << std::endl;
    auto sm = init();

    const char * logfile = "/tmp/%d.json\0";
    char buf[16];
    sprintf(buf, logfile, sum);

//    auto logid = start_log(sm, buf);
//    render(sm);

    // Load laikago
    b3Vector3 pos = b3MakeVector3(0,0, 0.47);
    b3Vector4 rpy = b3MakeVector4(0,0.707, 0.707, 0);
//    int uid = loadUrdf(sm, laikago, pos, rpy);


    int u = ceil(sqrt(sum));
    float dist = 2.5;
    float offset = u * dist/2;
    for(int i = 0 ;i < sum; i++){
        loadUrdf(sm, laikago, b3MakeVector3(i/u*dist - offset, i%u *dist - offset, 0.47), rpy);
    }

    loadUrdf(sm, ground);
    auto urdf_cost = std::chrono::high_resolution_clock::now();

//    stepsimulate(sm, 1000);

    stepfuntion(sm, 1000);

//    stop_log(sm, logid);
	sm->disconnectSharedMemory();

	auto loadurdf_cost = std::chrono::duration_cast<std::chrono::microseconds>(urdf_cost-begin).count();
	auto fn_cost = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - urdf_cost).count();
	std::cout << "urdf cost : " << loadurdf_cost << ", fn_cost : " << fn_cost << std::endl
	    << "end : " << sum << std::endl;

}

int main()
{
    once(16);


	return 0;
}