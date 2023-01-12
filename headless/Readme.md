# Running simple tracker in headless mode

This will allow simple tracker to be run ion a headless manner
The volume mounts below will mount local folders and files into the container replacing the container files and folders

The video input folder can have as meany videos as you would like to process
The processed artifacts will then be placed into the output folder

It's adviced that you create an appropriate mask for each video so as to avoid false positives
  you will need to update the settings.toml file pointing to the correct mask as well as what the required output has to be

```
docker run --rm \
  -v ../output:/opt/simpletracker/output \
  -v ../vids:/opt/simpletracker/input \
  -v ../masks:/opt/simpletracker/masks \
  -v ./settings.toml:/opt/simpletracker/settings.toml \
  -v ./run.sh:/opt/simpletracker/run.sh \
  sky360/simpletracker:1.0.2 \
  /bin/bash ./run.sh
```

**NOTE:** Feel free to run the ```./headless_run.sh``` command to test if this works for you. It will process existing videos.