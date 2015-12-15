#!/bin/sh
echo "running"
ipython2 main.py data/English-train.xml data/English-dev.xml KNN-English.answer SVM-English.answer Best-English.answer English
ipython2 main.py data/Catalan-train.xml data/Catalan-dev.xml KNN-Catalan.answer SVM-Catalan.answer Best-Catalan.answer Catalan
ipython2 main.py data/Spanish-train.xml data/Spanish-dev.xml KNN-Spanish.answer SVM-Spanish.answer Best-Spanish.answer Spanish

echo "scoring part A"
./scorer2 SVM-English.answer data/English-dev.key data/English.sensemap
./scorer2 KNN-English.answer data/English-dev.key data/English.sensemap
./scorer2 SVM-Spanish.answer data/Spanish-dev.key
./scorer2 KNN-Spanish.answer data/Spanish-dev.key
./scorer2 SVM-Catalan.answer data/Catalan-dev.key
./scorer2 KNN-Catalan.answer data/Catalan-dev.key
echo "scoring part B"
./scorer2 Best-English.answer data/English-dev.key data/English.sensemap
./scorer2 Best-Spanish.answer data/Spanish-dev.key
./scorer2 Best-Catalan.answer data/Catalan-dev.key
