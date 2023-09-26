mkdir -p Indexes/
mkdir -p Images/
mkdir -p Vectors/

rm -rf Images/images.txt

cd Decompressed/
for i in *; 
do 
    echo 'get filtered data from '$i;
    cut -f2- $i > ../Vectors/$i;
    echo 'get images links from '$i;
    sed 's/|/ /' $i | awk '{print $1}' >> ../Images/images.txt; 
    rm $i; 
done
