
rm -rf Images/images.txt
cd Decompressed/
for i in *; 
do 
    echo 'get filtered data from '$i;
    sed 's/|/ /' $i | awk '{$1=""; print $0}' > ../Vectors/$i;
    echo 'get images links from '$i;
    sed 's/|/ /' $i | awk '{print $1}' >> ../Images/images.txt; done