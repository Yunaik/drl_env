INCLUDE_DIRECTORIES( 
		${BULLET_PHYSICS_SOURCE_DIR}/src
		${BULLET_PHYSICS_SOURCE_DIR}/examples
		${BULLET_PHYSICS_SOURCE_DIR}/examples/ThirdPartyLibs
		${BULLET_PHYSICS_SOURCE_DIR}/examples/ThirdPartyLibs/enet/include
		${BULLET_PHYSICS_SOURCE_DIR}/examples/ThirdPartyLibs/clsocket/src
    	${BULLET_PHYSICS_SOURCE_DIR}/examples/SharedMemory
		${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/include
		${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/include/common
		${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/include/extensions
		${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/pxshared/include/foundation
		${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/source/pvd
		${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/include/cooking
		${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/include/cudamanager
		${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/pxshared/include
        ${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/source/common/include
        ${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/source/common/src
        ${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/source/fastxml/include
        ${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/source/filebuf/include
        ${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/source/foundation/include
        ${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/source/geomutils/include
        ${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/source/geomutils/src
        ${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/source/geomutils/src/ccd
        ${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/source/geomutils/src/common
        ${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/source/geomutils/src/contact
        ${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/source/geomutils/src/convex
        ${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/source/geomutils/src/distance
        ${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/source/geomutils/src/gjk
        ${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/source/geomutils/src/hf
        ${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/source/geomutils/src/intersection
        ${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/source/geomutils/src/mesh
        ${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/source/geomutils/src/pcm
        ${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/source/geomutils/src/sweep
        ${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/source/lowlevel/api/include
        ${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/source/lowlevel/common/include
        ${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/source/lowlevel/common/include/collision
        ${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/source/lowlevel/common/include/pipeline
        ${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/source/lowlevel/common/include/utils
        ${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/source/lowlevel/software/include
        ${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/source/lowlevelaabb/include
        ${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/source/lowleveldynamics/include
        ${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/source/physx/src
        ${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/source/physx/src/buffering
        ${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/source/physx/src/device
        ${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/source/physxcooking/src
        ${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/source/physxcooking/src/convex
        ${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/source/physxcooking/src/mesh
        ${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/source/physxextensions/src
        ${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/source/physxextensions/src/serialization/Binary
        ${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/source/physxextensions/src/serialization/File
        ${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/source/physxextensions/src/serialization/Xml
        ${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/source/physxmetadata/core/include
        ${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/source/physxmetadata/extensions/include
        ${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/source/physxvehicle/src
        ${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/source/physxvehicle/src/physxmetadata/include
        ${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/source/pvd/include
        ${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/source/scenequery/include
        ${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/source/simulationcontroller/include
        ${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/source/simulationcontroller/src
        ${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/include
        ${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/include/characterkinematic
        ${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/include/common
        ${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/include/cooking
        ${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/include/extensions
        ${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/include/geometry
        ${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/include/geomutils
        ${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/include/vehicle
        ${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/pxshared/include
        ${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/include/cudamanager
        ${BULLET_PHYSICS_SOURCE_DIR}/src/PhysX/physx/source/physxgpu/include
        ${BULLET_PHYSICS_SOURCE_DIR}/examples/ThirdPartyLibs/glad
		${BULLET_PHYSICS_SOURCE_DIR}/examples/ThirdPartyLibs/optionalX11

  )


SET(PhysxDemo_SRCS
        PhysxDemoMain.cpp
		../../examples/SharedMemory/plugins/collisionFilterPlugin/collisionFilterPlugin.cpp
		../../examples/SharedMemory/plugins/pdControlPlugin/pdControlPlugin.cpp
		../../examples/SharedMemory/plugins/pdControlPlugin/pdControlPlugin.h
		../../examples/SharedMemory/b3RobotSimulatorClientAPI_NoDirect.cpp
		../../examples/SharedMemory/b3RobotSimulatorClientAPI_NoDirect.h
		../../examples/SharedMemory/IKTrajectoryHelper.cpp
		../../examples/SharedMemory/IKTrajectoryHelper.h
		../../examples/ExampleBrowser/InProcessExampleBrowser.cpp
		../../examples/SharedMemory/plugins/tinyRendererPlugin/tinyRendererPlugin.cpp
		../../examples/SharedMemory/plugins/tinyRendererPlugin/TinyRendererVisualShapeConverter.cpp
		../../examples/OpenGLWindow/SimpleCamera.cpp
		../../examples/OpenGLWindow/SimpleCamera.h
		../../examples/TinyRenderer/geometry.cpp
		../../examples/TinyRenderer/model.cpp
		../../examples/TinyRenderer/tgaimage.cpp
		../../examples/TinyRenderer/our_gl.cpp
		../../examples/TinyRenderer/TinyRenderer.cpp
		../../examples/SharedMemory/InProcessMemory.cpp
		../../examples/SharedMemory/PhysicsClient.cpp
		../../examples/SharedMemory/PhysicsClient.h
		../../examples/SharedMemory/PhysicsServer.cpp
		../../examples/SharedMemory/PhysicsServer.h
		../../examples/SharedMemory/PhysicsServerExample.cpp
		../../examples/SharedMemory/PhysicsServerExampleBullet2.cpp
		../../examples/SharedMemory/SharedMemoryInProcessPhysicsC_API.cpp
		../../examples/SharedMemory/PhysicsServerSharedMemory.cpp
		../../examples/SharedMemory/PhysicsServerSharedMemory.h
		../../examples/SharedMemory/PhysicsDirect.cpp
		../../examples/SharedMemory/PhysicsDirect.h
		../../examples/SharedMemory/PhysicsDirectC_API.cpp
		../../examples/SharedMemory/PhysicsDirectC_API.h
		../../examples/SharedMemory/PhysicsServerCommandProcessor.cpp
		../../examples/SharedMemory/PhysicsServerCommandProcessor.h
		../../examples/SharedMemory/b3PluginManager.cpp
		
		../../examples/SharedMemory/PhysicsClientSharedMemory.cpp
		../../examples/SharedMemory/PhysicsClientSharedMemory.h
		../../examples/SharedMemory/PhysicsClientSharedMemory_C_API.cpp
		../../examples/SharedMemory/PhysicsClientSharedMemory_C_API.h
		../../examples/SharedMemory/PhysicsClientC_API.cpp
		../../examples/SharedMemory/PhysicsClientC_API.h
		../../examples/SharedMemory/Win32SharedMemory.cpp
		../../examples/SharedMemory/Win32SharedMemory.h
		../../examples/SharedMemory/PosixSharedMemory.cpp
		../../examples/SharedMemory/PosixSharedMemory.h
		../../examples/Utils/b3ResourcePath.cpp
		../../examples/Utils/b3ResourcePath.h
		../../examples/Utils/RobotLoggingUtil.cpp
		../../examples/Utils/RobotLoggingUtil.h
		../../examples/ThirdPartyLibs/tinyxml2/tinyxml2.cpp
		../../examples/ThirdPartyLibs/Wavefront/tiny_obj_loader.cpp
		../../examples/ThirdPartyLibs/Wavefront/tiny_obj_loader.h
		../../examples/ThirdPartyLibs/stb_image/stb_image.cpp
		../../examples/Importers/ImportColladaDemo/LoadMeshFromCollada.cpp
		../../examples/Importers/ImportObjDemo/LoadMeshFromObj.cpp
		../../examples/Importers/ImportObjDemo/Wavefront2GLInstanceGraphicsShape.cpp
		../../examples/Importers/ImportMJCFDemo/BulletMJCFImporter.cpp
		../../examples/Importers/ImportURDFDemo/BulletUrdfImporter.cpp
		../../examples/Importers/ImportURDFDemo/MyMultiBodyCreator.cpp
		../../examples/Importers/ImportURDFDemo/URDF2Bullet.cpp
		../../examples/Importers/ImportURDFDemo/UrdfParser.cpp
		../../examples/Importers/ImportURDFDemo/urdfStringSplit.cpp
		../../examples/Importers/ImportMeshUtility/b3ImportMeshUtility.cpp 
		../../examples/MultiThreading/b3PosixThreadSupport.cpp
		../../examples/MultiThreading/b3Win32ThreadSupport.cpp
		../../examples/MultiThreading/b3ThreadSupportInterface.cpp

		../../examples/SharedMemory/physx/PhysXC_API.h
		../../examples/SharedMemory/physx/PhysXC_API.cpp
		../../examples/SharedMemory/physx/PhysXServerCommandProcessor.h
		../../examples/SharedMemory/physx/PhysXServerCommandProcessor.cpp
		../../examples/SharedMemory/physx/PhysXUrdfImporter.h
		../../examples/SharedMemory/physx/PhysXUrdfImporter.cpp
		../../examples/SharedMemory/physx/PhysXUserData.h
		../../examples/SharedMemory/physx/URDF2PhysX.h
		../../examples/SharedMemory/physx/URDF2PhysX.cpp
        ../../src/PhysXLowLevelAll.cpp
        ../../src/PhysXExtensionAll.cpp
        ../../src/PhysX/physx/source/physxextensions/src/ExtDefaultCpuDispatcher.cpp
        ../../src/PhysX/physx/source/lowlevel/common/src/pipeline/PxcContactMethodImpl.cpp
        ../../src/PhysX/physx/source/lowlevel/common/src/collision/PxcContact.cpp
        ../../examples/SharedMemory/physx/PhysXC_API.cpp
        ../../examples/SharedMemory/physx/PhysXServerCommandProcessor.cpp
        ../../examples/SharedMemory/physx/PhysXUrdfImporter.cpp
        ../../examples/SharedMemory/physx/URDF2PhysX.cpp
        ../../src/btLinearMathAll.cpp
        ../../src/PhysXGeomUtilsAll.cpp
        ../../examples/SharedMemory/plugins/eglPlugin/eglRendererVisualShapeConverter.cpp
        ../../examples/SharedMemory/plugins/eglPlugin/eglRendererPlugin.cpp
        ../../src/PhysX/physx/source/geomutils/src/contact/GuFeatureCode.cpp
        ../../src/PhysX/physx/source/geomutils/src/GuSweepMTD.cpp
        ../../src/PhysX/physx/source/geomutils/src/GuSweepSharedTests.cpp
        ../../src/PhysX/physx/source/geomutils/src/GuSweepTests.cpp
        ../../src/PhysX/physx/source/geomutils/src/contact/GuContactBoxBox.cpp
        ../../src/PhysX/physx/source/geomutils/src/contact/GuContactCapsuleBox.cpp
        ../../src/PhysX/physx/source/geomutils/src/contact/GuContactCapsuleCapsule.cpp
        ../../src/PhysX/physx/source/geomutils/src/contact/GuContactCapsuleConvex.cpp
        ../../src/PhysX/physx/source/geomutils/src/contact/GuContactCapsuleMesh.cpp
        ../../src/PhysX/physx/source/geomutils/src/contact/GuContactConvexConvex.cpp
        ../../src/PhysX/physx/source/geomutils/src/contact/GuContactConvexMesh.cpp
        ../../src/PhysX/physx/source/geomutils/src/contact/GuContactPlaneBox.cpp
        ../../src/PhysX/physx/source/geomutils/src/contact/GuContactPlaneCapsule.cpp
        ../../src/PhysX/physx/source/geomutils/src/contact/GuContactPlaneConvex.cpp
        ../../src/PhysX/physx/source/geomutils/src/contact/GuContactPolygonPolygon.cpp
        ../../src/PhysX/physx/source/geomutils/src/contact/GuContactSphereBox.cpp
        ../../src/PhysX/physx/source/geomutils/src/contact/GuContactSphereCapsule.cpp
        ../../src/PhysX/physx/source/geomutils/src/contact/GuContactSphereMesh.cpp
        ../../src/PhysX/physx/source/geomutils/src/contact/GuContactSpherePlane.cpp
        ../../src/PhysX/physx/source/geomutils/src/contact/GuContactSphereSphere.cpp
        ../../src/PhysX/physx/source/geomutils/src/mesh/GuBV4_BoxOverlap.cpp
        ../../src/PhysX/physx/source/geomutils/src/mesh/GuBV4_CapsuleSweep.cpp
        ../../src/PhysX/physx/source/geomutils/src/mesh/GuBV4_CapsuleSweepAA.cpp
        ../../src/PhysX/physx/source/geomutils/src/mesh/GuBV4_OBBSweep.cpp
        ../../src/PhysX/physx/source/geomutils/src/mesh/GuBV4_Raycast.cpp
        ../../src/PhysX/physx/source/geomutils/src/mesh/GuBV4_SphereOverlap.cpp
        ../../src/PhysX/physx/source/geomutils/src/mesh/GuBV4_SphereSweep.cpp
        ../../src/PhysX/physx/source/geomutils/src/mesh/GuMeshQuery.cpp
        ../../src/PhysX/physx/source/geomutils/src/hf/GuHeightField.cpp
        ../../src/PhysX/physx/source/geomutils/src/hf/GuHeightFieldUtil.cpp
        ../../src/PhysX/physx/source/geomutils/src/mesh/GuBV4.cpp
        ../../src/PhysX/physx/source/geomutils/src/mesh/GuBV4Build.cpp
        ../../src/PhysX/physx/source/geomutils/src/mesh/GuBV4_AABBSweep.cpp
        ../../src/PhysX/physx/source/geomutils/src/mesh/GuMidphaseBV4.cpp
        ../../src/PhysX/physx/source/geomutils/src/mesh/GuMidphaseRTree.cpp
        ../../src/PhysX/physx/source/geomutils/src/sweep/GuSweepBoxBox.cpp
        ../../src/PhysX/physx/source/geomutils/src/sweep/GuSweepBoxSphere.cpp
        ../../src/PhysX/physx/source/geomutils/src/sweep/GuSweepBoxTriangle_FeatureBased.cpp
        ../../src/PhysX/physx/source/geomutils/src/sweep/GuSweepBoxTriangle_SAT.cpp
        ../../src/PhysX/physx/source/geomutils/src/sweep/GuSweepCapsuleBox.cpp
        ../../src/PhysX/physx/source/geomutils/src/sweep/GuSweepCapsuleCapsule.cpp
        ../../src/PhysX/physx/source/geomutils/src/sweep/GuSweepCapsuleTriangle.cpp
        ../../src/PhysX/physx/source/geomutils/src/sweep/GuSweepSphereCapsule.cpp
        ../../src/PhysX/physx/source/geomutils/src/sweep/GuSweepSphereSphere.cpp
        ../../src/PhysX/physx/source/geomutils/src/sweep/GuSweepSphereTriangle.cpp
        ../../src/PhysX/physx/source/geomutils/src/sweep/GuSweepTriangleUtils.cpp
        ../../src/PhysX/physx/source/geomutils/src/pcm/GuPCMTriangleContactGen.cpp
        ../../src/PhysX/physx/source/geomutils/src/pcm/GuPersistentContactManifold.cpp
        ../../src/PhysX/physx/source/geomutils/src/mesh/GuBV32.cpp
        ../../src/PhysX/physx/source/geomutils/src/mesh/GuBV32Build.cpp
        ../../src/PhysX/physx/source/geomutils/src/mesh/GuOverlapTestsMesh.cpp
        ../../src/PhysX/physx/source/geomutils/src/mesh/GuRTree.cpp
        ../../src/PhysX/physx/source/geomutils/src/mesh/GuRTreeQueries.cpp
        ../../src/PhysX/physx/source/geomutils/src/mesh/GuSweepsMesh.cpp
        ../../src/PhysX/physx/source/geomutils/src/mesh/GuTriangleMesh.cpp
        ../../src/PhysX/physx/source/geomutils/src/mesh/GuTriangleMeshBV4.cpp
        ../../src/PhysX/physx/source/geomutils/src/mesh/GuTriangleMeshRTree.cpp
        ../../src/PhysX/physx/source/geomutils/src/pcm/GuPCMContactBoxBox.cpp
        ../../src/PhysX/physx/source/geomutils/src/pcm/GuPCMContactBoxConvex.cpp
        ../../src/PhysX/physx/source/fastxml/src/PsFastXml.cpp
        ../../src/PhysX/physx/source/immediatemode/src/NpImmediateMode.cpp
        ../../src/PhysX/physx/source/physxmetadata/core/src/PxAutoGeneratedMetaDataObjects.cpp
        ../../src/PhysX/physx/source/physxmetadata/core/src/PxMetaDataObjects.cpp
        ../../src/PhysX/physx/source/physxmetadata/extensions/src/PxExtensionAutoGeneratedMetaDataObjects.cpp
        ../../src/PhysX/physx/source/task/src/TaskManager.cpp
        ../../src/PhysXVehicleAll.cpp
        ../../src/PhysX/physx/source/pvd/src/PxPvdProfileZoneClient.cpp
        ../../src/PhysX/physx/source/pvd/src/PxPvdUserRenderer.cpp
        ../../src/PhysX/physx/source/pvd/src/PxPvdImpl.cpp
        ../../src/PhysX/physx/source/pvd/src/PxPvdMemClient.cpp
        ../../src/PhysX/physx/source/pvd/src/PxPvdObjectModelMetaData.cpp
        ../../src/PhysX/physx/source/pvd/src/PxPvdObjectRegistrar.cpp
        ../../src/PhysX/physx/source/pvd/src/PxProfileEventImpl.cpp
        ../../src/PhysX/physx/source/pvd/src/PxPvd.cpp
        ../../src/PhysX/physx/source/pvd/src/PxPvdDataStream.cpp
        ../../src/PhysX/physx/source/pvd/src/PxPvdDefaultFileTransport.cpp
        ../../src/PhysX/physx/source/pvd/src/PxPvdDefaultSocketTransport.cpp
        ../../src/PhysX/physx/source/physxcooking/src/convex/ConvexHullBuilder.cpp
        ../../src/PhysX/physx/source/physxcooking/src/convex/QuickHullConvexHullLib.cpp
        ../../src/PhysX/physx/source/physxcooking/src/mesh/TriangleMeshBuilder.cpp
        ../../src/PhysXCharacterAll.cpp
        ../../src/PhysXCommonAll.cpp
        ../../src/PhysXCookingAll.cpp
        ../../src/PhysX/physx/source/physxextensions/src/ExtDefaultErrorCallback.cpp
        ../../src/PhysX/physx/source/physxextensions/src/serialization/SnSerialization.cpp
        ../../src/PhysX/physx/source/physxextensions/src/serialization/Binary/SnConvX.cpp
        ../../src/PhysX/physx/source/physxextensions/src/serialization/Binary/SnConvX_Align.cpp
        ../../src/PhysX/physx/source/physxextensions/src/serialization/Binary/SnConvX_Convert.cpp
        ../../src/PhysX/physx/source/physxextensions/src/serialization/Binary/SnConvX_Error.cpp
        ../../src/PhysX/physx/source/physxextensions/src/serialization/Binary/SnConvX_MetaData.cpp
        ../../src/PhysX/physx/source/physxextensions/src/serialization/Binary/SnConvX_Output.cpp
        ../../src/PhysX/physx/source/physxextensions/src/serialization/Binary/SnConvX_Union.cpp
        ../../src/PhysX/physx/source/physxextensions/src/serialization/Binary/SnSerializationContext.cpp
        ../../src/PhysX/physx/source/physxextensions/src/serialization/Xml/SnRepXUpgrader.cpp
        ../../src/PhysX/physx/source/physxextensions/src/serialization/Xml/SnXmlSerialization.cpp
        ../../src/PhysXFoundationAll.cpp
        ../../src/PhysXNpSrcAll.cpp
        ../../src/PhysXSceneQueryAll.cpp
        ../../src/PhysXSimulationControllerAll.cpp
        ../../src/PhysX/physx/source/simulationcontroller/src/ScShapeSim.cpp
        ../../src/PhysX/physx/source/lowleveldynamics/src/DyTGSContactPrep.cpp
        ../../src/PhysX/physx/source/lowleveldynamics/src/DyTGSContactPrepBlock.cpp
        ../../src/PhysX/physx/source/lowleveldynamics/src/DyTGSDynamics.cpp
        ../../src/PhysX/physx/source/lowlevelaabb/src/BpAABBManager.cpp
        ../../src/PhysX/physx/source/lowlevelaabb/src/BpBroadPhase.cpp
        ../../src/PhysX/physx/source/lowlevelaabb/src/BpBroadPhaseABP.cpp
        ../../src/PhysX/physx/source/lowlevelaabb/src/BpBroadPhaseMBP.cpp
        ../../src/PhysX/physx/source/lowlevelaabb/src/BpBroadPhaseSap.cpp
        ../../src/PhysX/physx/source/lowlevelaabb/src/BpBroadPhaseSapAux.cpp
        ../../src/PhysX/physx/source/lowlevelaabb/src/BpBroadPhaseShared.cpp
        ../../src/PhysX/physx/source/lowlevelaabb/src/BpMBPTasks.cpp
        ../../src/PhysX/physx/source/lowlevelaabb/src/BpSAPTasks.cpp
		../../examples/OpenGLWindow/X11OpenGLWindow.cpp
    	../../examples/ThirdPartyLibs/glad/gl.c
    	../../src/PhysXFoundationUnix.cpp
    	../../examples/ThirdPartyLibs/glad/glx.c
    	../../src/PhysX/physx/source/physx/src/device/linux/PhysXIndicatorLinux.cpp
		../../examples/OpenGLWindow/EGLOpenGLWindow.cpp
		../../examples/ThirdPartyLibs/glad/egl.c
)

IF(BUILD_CLSOCKET)
 ADD_DEFINITIONS(-DBT_ENABLE_CLSOCKET)
ENDIF(BUILD_CLSOCKET)

IF(WIN32)
	LINK_LIBRARIES(
		${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY}
	)
	IF(BUILD_ENET)
		ADD_DEFINITIONS(-DWIN32 -DBT_ENABLE_ENET)
	ENDIF(BUILD_ENET)
	IF(BUILD_CLSOCKET)
		ADD_DEFINITIONS(-DWIN32)
	ENDIF(BUILD_CLSOCKET)

ELSE(WIN32)
	IF(BUILD_ENET)
		ADD_DEFINITIONS(-DHAS_SOCKLEN_T -DBT_ENABLE_ENET)
	ENDIF(BUILD_ENET)

	IF(BUILD_CLSOCKET)	
		ADD_DEFINITIONS(${OSDEF})
	ENDIF(BUILD_CLSOCKET)
ENDIF(WIN32)


IF(BUILD_ENET)
	set(PhysxDemo_SRCS ${PhysxDemo_SRCS}
		../../examples/SharedMemory/PhysicsClientUDP.cpp
		../../examples/SharedMemory/PhysicsClientUDP_C_API.cpp
		../../examples/SharedMemory/PhysicsClientUDP.h		
		../../examples/SharedMemory/PhysicsClientUDP_C_API.h
		../../examples/ThirdPartyLibs/enet/win32.c
		../../examples/ThirdPartyLibs/enet/unix.c
		../../examples/ThirdPartyLibs/enet/callbacks.c
		../../examples/ThirdPartyLibs/enet/compress.c
		../../examples/ThirdPartyLibs/enet/host.c
		../../examples/ThirdPartyLibs/enet/list.c
		../../examples/ThirdPartyLibs/enet/packet.c
		../../examples/ThirdPartyLibs/enet/peer.c
		../../examples/ThirdPartyLibs/enet/protocol.c
	)
ENDIF(BUILD_ENET)

IF(BUILD_CLSOCKET)
	set(PhysxDemoSRCS ${PhysxDemo_SRCS}
		 ../../examples/SharedMemory/PhysicsClientTCP.cpp
                 ../../examples/SharedMemory/PhysicsClientTCP.h
                 ../../examples/SharedMemory/PhysicsClientTCP_C_API.cpp
                 ../../examples/SharedMemory/PhysicsClientTCP_C_API.h
		 ../../examples/ThirdPartyLibs/clsocket/src/SimpleSocket.cpp
		 ../../examples/ThirdPartyLibs/clsocket/src/ActiveSocket.cpp
		 ../../examples/ThirdPartyLibs/clsocket/src/PassiveSocket.cpp
	)
ENDIF()


#some code to support OpenGL and Glew cross platform
IF (WIN32)
	INCLUDE_DIRECTORIES(
			${BULLET_PHYSICS_SOURCE_DIR}/btgui/OpenGLWindow/GlewWindows
	)
	ADD_DEFINITIONS(-DGLEW_STATIC)
	LINK_LIBRARIES( ${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY} )
ELSE(WIN32)
	IF(APPLE)
		find_library(COCOA NAMES Cocoa)
		MESSAGE(${COCOA})
		link_libraries(${COCOA} ${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY})

	ELSE(APPLE)
		INCLUDE_DIRECTORIES(
				${BULLET_PHYSICS_SOURCE_DIR}/btgui/OpenGLWindow/GlewWindows
		)
		ADD_DEFINITIONS("-DGLEW_INIT_OPENGL11_FUNCTIONS=1")
		ADD_DEFINITIONS("-DGLEW_STATIC")
		ADD_DEFINITIONS("-DGLEW_DYNAMIC_LOAD_ALL_GLX_FUNCTIONS=1")
		ADD_DEFINITIONS("-DDYNAMIC_LOAD_X11_FUNCTIONS")
		ADD_DEFINITIONS("-DHAS_SOCKLEN_T")
		ADD_DEFINITIONS("-fno-inline-functions-called-once")

		ADD_DEFINITIONS("-DBT_USE_EGL")
		ADD_DEFINITIONS("-DBT_ENABLE_PHYSX")
        ADD_DEFINITIONS("-DPX_COOKING")
		ADD_DEFINITIONS("-DPX_PHYSX_STATIC_LIB")
		ADD_DEFINITIONS("-DPX_SUPPORT_PVD=0")
		ADD_DEFINITIONS("-DPX_PHYSX_STATIC_LIB")
		ADD_DEFINITIONS("-D_LINUX")
		ADD_DEFINITIONS("-DBT_THREADSAFE=1")

		LINK_LIBRARIES(  pthread ${DL} )
	ENDIF(APPLE)
ENDIF(WIN32)


ADD_EXECUTABLE(PhysxDemo  ${PhysxDemo_SRCS})
#target_compile_options(PhysxDemo PRIVATE "-std=c++11")



SET_TARGET_PROPERTIES(PhysxDemo PROPERTIES VERSION ${BULLET_VERSION})
SET_TARGET_PROPERTIES(PhysxDemo PROPERTIES DEBUG_POSTFIX "_d")


IF(WIN32)
	IF(BUILD_ENET OR BUILD_CLSOCKET)
		TARGET_LINK_LIBRARIES(PhysxDemo ws2_32 )
	ENDIF(BUILD_ENET OR BUILD_CLSOCKET)
ENDIF(WIN32)



TARGET_LINK_LIBRARIES(PhysxDemo BulletExampleBrowserLib BulletFileLoader BulletWorldImporter BulletSoftBody BulletDynamics BulletCollision BulletInverseDynamicsUtils BulletInverseDynamics LinearMath OpenGLWindow gwen BussIK Bullet3Common pthread dl)


