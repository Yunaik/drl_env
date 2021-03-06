# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import dist_pb2 as dist__pb2


class WALLEStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.Connect = channel.unary_unary(
        '/walle.dist.WALLE/Connect',
        request_serializer=dist__pb2.EnvCfg.SerializeToString,
        response_deserializer=dist__pb2.EnvInfo.FromString,
        )
    self.Reset = channel.unary_unary(
        '/walle.dist.WALLE/Reset',
        request_serializer=dist__pb2.Uids.SerializeToString,
        response_deserializer=dist__pb2.Observation.FromString,
        )
    self.Step = channel.unary_unary(
        '/walle.dist.WALLE/Step',
        request_serializer=dist__pb2.Actions.SerializeToString,
        response_deserializer=dist__pb2.Records.FromString,
        )
    self.GetObs = channel.unary_unary(
        '/walle.dist.WALLE/GetObs',
        request_serializer=dist__pb2.Uids.SerializeToString,
        response_deserializer=dist__pb2.Observation.FromString,
        )
    self.DisConnect = channel.unary_unary(
        '/walle.dist.WALLE/DisConnect',
        request_serializer=dist__pb2.Void.SerializeToString,
        response_deserializer=dist__pb2.RspState.FromString,
        )


class WALLEServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def Connect(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def Reset(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def Step(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def GetObs(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def DisConnect(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_WALLEServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'Connect': grpc.unary_unary_rpc_method_handler(
          servicer.Connect,
          request_deserializer=dist__pb2.EnvCfg.FromString,
          response_serializer=dist__pb2.EnvInfo.SerializeToString,
      ),
      'Reset': grpc.unary_unary_rpc_method_handler(
          servicer.Reset,
          request_deserializer=dist__pb2.Uids.FromString,
          response_serializer=dist__pb2.Observation.SerializeToString,
      ),
      'Step': grpc.unary_unary_rpc_method_handler(
          servicer.Step,
          request_deserializer=dist__pb2.Actions.FromString,
          response_serializer=dist__pb2.Records.SerializeToString,
      ),
      'GetObs': grpc.unary_unary_rpc_method_handler(
          servicer.GetObs,
          request_deserializer=dist__pb2.Uids.FromString,
          response_serializer=dist__pb2.Observation.SerializeToString,
      ),
      'DisConnect': grpc.unary_unary_rpc_method_handler(
          servicer.DisConnect,
          request_deserializer=dist__pb2.Void.FromString,
          response_serializer=dist__pb2.RspState.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'walle.dist.WALLE', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
