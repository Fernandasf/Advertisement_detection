# Advertisement_detection
Advertisement detection using deep learning.

Responsible student: Victor Paganotto

Comand line to extract files from directory through text file names:

Dadosfold1test_namesAudios.txt = nome do arquivo txt onde estão os nomes dos arquivos
Test = nome da pasta que os arquivos serão movidos

$ cat Dadosfold1test_namesAudios.txt | parallel -m -j0 --no-notice mv {} Test
