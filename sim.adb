#$ -j y                   # Join stdin and stderr
#$ -V                     # Use current environment variables
#$ -cwd                   # Use current directory
#$ -N adb_cv              # Job name
#$ -o $JOB_NAME.o$JOB_ID  # Name of the output file

setenv WORKDIR /home/ysakamoto/work/otto
cd $WORKDIR
setenv OMP_NUM_THREADS 12

# max_depth
Python adb_cv.py $1 $2 $3

echo "DONE"
