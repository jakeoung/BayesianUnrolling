# Folder structure
# The current folder: ProjectRoot/code/download_data.sh
# Run this file with argument 0, 1 or 2
# This script will download data and extract them into ProjectRoot/data/

if [ $1 == "0" ]; then
    mkdir ../data
    mkdir ../data/middlebury
    mkdir ../data/middlebury/raw
    wget https://vision.middlebury.edu/stereo/data/scenes2005/HalfSize/zip-2views/Art-2views.zip
    wget https://vision.middlebury.edu/stereo/data/scenes2005/HalfSize/zip-2views/Reindeer-2views.zip
    wget https://vision.middlebury.edu/stereo/data/scenes2005/HalfSize/zip-2views/Books-2views.zip
    wget https://vision.middlebury.edu/stereo/data/scenes2005/HalfSize/zip-2views/Dolls-2views.zip
    wget https://vision.middlebury.edu/stereo/data/scenes2005/HalfSize/zip-2views/Laundry-2views.zip
    wget https://vision.middlebury.edu/stereo/data/scenes2005/HalfSize/zip-2views/Moebius-2views.zip

    wget https://vision.middlebury.edu/stereo/data/scenes2006/HalfSize/zip-2views/Plastic-2views.zip
    wget https://vision.middlebury.edu/stereo/data/scenes2006/HalfSize/zip-2views/Midd1-2views.zip
    wget https://vision.middlebury.edu/stereo/data/scenes2006/HalfSize/zip-2views/Lampshade2-2views.zip
    wget https://vision.middlebury.edu/stereo/data/scenes2006/HalfSize/zip-2views/Aloe-2views.zip
    wget https://vision.middlebury.edu/stereo/data/scenes2006/HalfSize/zip-2views/Bowling1-2views.zip
    # wget https://vision.middlebury.edu/stereo/data/scenes2005/HalfSize/zip-2views/ALL-2views.zip
    unzip '*.zip' -d ../data/middlebury/raw
    rm *.zip
elif [ $1 == "1" ]; then
    cd ../data
    mkdir mpi
    cd mpi
    wget http://files.is.tue.mpg.de/jwulff/sintel/MPI-Sintel-stereo-training-20150305.zip
    unzip MPI-Sintel-stereo-training-20150305.zip
    rm MPI-Sintel-stereo-training-20150305.zip
fi
