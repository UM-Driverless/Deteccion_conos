#TODO NO FUNCIONA
#!/bin/bash

# Output file to store the temperature log
output_file="gpu_temp_log.txt"

# Duration of the test in seconds (1 hour = 3600 seconds)
duration=3

# Log GPU temperature every 10 seconds
interval=10

echo "Logging GPU temperature for $duration seconds..."

# Loop to log GPU temperature
for ((i = 0; i < duration; i += interval)); do
    # Example of what we want to filter: SOC2@48.875C
    echo "$(tegrastats | grep -E -o "SOC2@([^C]+)C")" >> gpu_temp_log.txt
    # temp=$(tegrastats | grep -E -o "SOC2@([^C]+)C")
    # echo $temp
    # echo "$(date +'%Y-%m-%d %H:%M:%S'), CPU SOC Temperature: $temp " >> "$output_file"
    #sleep "$interval"
done

echo "Logging complete. Check $output_file for the results."