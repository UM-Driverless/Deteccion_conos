# check_ch
# Firstly import canlib so that it can be used in the script.
from canlib import canlib

# .getNumberOfChannels() is used to detect the number of channels and
# the number is saved in the variable num_channels.
num_channels = canlib.getNumberOfChannels()

# num_channels is printed out as text so that the user can see how many
# channels were found.
print(f"Found {num_channels} channels")

# Next a for loop is created. This loop will repeat the code within for each
# channel that was detected. 
for ch in range(num_channels):
# The data of each specific channel is saved in chd.
    chd = canlib.ChannelData(ch)
# Lastly the channel, channel name, product number, serial number, and local 
# channel number on the device are printed.
    print(f"{ch}. {chd.channel_name} ({chd.card_upc_no.product()}:{chd.card_serial_no}/{chd.chan_no_on_card})")
