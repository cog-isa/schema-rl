for file in ./*.pdf
do
    convert -density 300 $file -quality 100 ${file%.*}.png
done
