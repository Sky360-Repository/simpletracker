PYTHONPATH="${PYTHONPATH}:uap_tracker/" SKY360_VISUALIZER__format=none SKY360_VIDETRACKER__CALCULATE_OPTICAL_FLOW=true SKY360_VIDEOTRACKER__DETECTION_MODE=background_subtraction SKY360_VIDEOTRACKER__MASK_PCT=0 SKY360_output_format=video python3 uap_tracker/main.py -f $1