# !/bin/bsh

points=(3497 4000 4500 5000 5500 6000 6500 7000 7500 8000)
length=${#points[@]}

for (( i=0; i < $length-1; i++ )); do
  echo ${points[$i]}
  echo ${points[$i+1]}
  python sample_infer.py --start ${points[$i]} --end ${points[$i+1]}
  sleep 10
done
