for filename in /home/sa-zhao/perception-python/Kuka-Perception/training_images/G/*.png; do
    mogrify -rotate $(awk -v min=0 -v max=360 'BEGIN{srand(); print int(min+rand()*(max-min+1))}') "$filename"
done