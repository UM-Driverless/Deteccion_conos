# send_msg_pyt
from canlib import canlib, Frame

# instead of opening the two channels and closing them one by one, we will use a
# with statement. Using the with statement to open one or more channels with
# canlib.openChannel(i) as ch_x. Within this with statement we will write the
# rest of the code.
with canlib.openChannel(0) as ch_a, canlib.openChannel(3) as ch_b:

# Instead of going on bus with "copy-paste" for each channel, we will use a
# for-loop. Within this loop we will go through a list of all channels opened
# using the with statement. Currently we only have two channels, which makes
# the for-loop somewhat unnecessary. However, when we start using more
# channels the for-loop will be preferred.
    for ch in [ch_a, ch_b]:
        ch.busOn()

    frame = Frame(id_=123, data=[72, 69, 76, 76, 79, 33])
    ch_a.write(frame)

    msg = ch_b.read(timeout=500)
    print(msg)

# After we run out of code within the with statement and exit it, we don't
# need to manually close it or go off bus. The channels that were open using
# the with statement will be automatically closed, and with the channels being
# closed they also went off the bus.

