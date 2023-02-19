from connection_utils.CanKvaser import CanKvaser

can_bus = CanKvaser()

# can_bus.send()
print(can_bus.receive_frame())