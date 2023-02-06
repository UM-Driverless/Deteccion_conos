from connection_utils.can_communication import Can_communication

can_bus = Can_communication()

# can_bus.send()
print(can_bus.receive_frame())