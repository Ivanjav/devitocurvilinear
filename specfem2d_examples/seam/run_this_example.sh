#!/bin/bash
#
# This script runs the main directory SPECFEM2D example problem, which is shown
# on the cover of the User Manual (https://specfem2d.readthedocs.io/en/latest/)
#
# Under the hood it runs the mesher and the solver (in serial or parallel) for
# a multi-layered 2D domain with topography and a cavity.
#
# To create a movie of the results, have a look at the script:
# ./utils/create_main_repo_example_movie.sh
#

echo "running example: `date`"
currentdir=`pwd`

# sets up directory structure in current example directoy
echo
echo "setting up example..."
echo

# cleans output files
mkdir -p OUTPUT_FILES
rm -rf OUTPUT_FILES/*

# links executables
mkdir -p bin
cd bin/
rm -f *

export SPECFEM2D=/home/ivan/Software/specfem2d
ln -s $SPECFEM2D/bin/xmeshfem2D
ln -s $SPECFEM2D/bin/xspecfem2D

cd $currentdir

# stores setup
cp DATA/Par_file OUTPUT_FILES/
cp DATA/SOURCE OUTPUT_FILES/

# Get the number of processors
NPROC=`grep ^NPROC DATA/Par_file | cut -d = -f 2 | cut -d \# -f 1 | tr -d ' '`

# runs database generation
  echo
  echo "running mesher..."
  echo
  ./bin/xmeshfem2D
# checks exit code
if [[ $? -ne 0 ]]; then exit 1; fi

# runs simulation
if [ "$NPROC" -eq 1 ]; then
  # This is a serial simulation
  echo
  echo "running solver..."
  echo
  ./bin/xspecfem2D
else
  # This is a MPI simulation
  echo
  echo "running solver on $NPROC processors..."
  echo
  mpirun -np $NPROC ./bin/xspecfem2D
fi
# checks exit code
if [[ $? -ne 0 ]]; then exit 1; fi

# stores output
cp DATA/*SOURCE* DATA/*STATIONS* OUTPUT_FILES

echo
echo "see results in directory: OUTPUT_FILES/"
echo
echo "done"
echo `date`

