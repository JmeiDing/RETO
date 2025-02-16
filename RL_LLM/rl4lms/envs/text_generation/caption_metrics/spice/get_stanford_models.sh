#!/usr/bin/env sh
# This script downloads the Stanford CoreNLP models.

# Adapted from:
# https://panderson.me/spice/

#CORENLP的值Stanford CoreNLP 的压缩文件名
CORENLP=stanford-corenlp-full-2015-12-09
#SPICELIB的值是目标目录的路径
SPICELIB=lib

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

echo $(pwd)
echo "$(dirname "$0")"

echo "Downloading..."

#wget http://nlp.stanford.edu/software/$CORENLP.zip

# 使用 curl 替代
curl -O http://nlp.stanford.edu/software/$CORENLP.zip


echo "Unzipping..."

#使用 unzip 命令将指定路径下的 Stanford CoreNLP 压缩文件（由 $CORENLP.zip 指定）解压缩到目标目录（由 $SPICELIB/ 指定）。
# 这一步是将整个工具包解压缩到指定目录。
unzip $CORENLP.zip -d $SPICELIB/
mv $SPICELIB/$CORENLP/stanford-corenlp-3.6.0.jar $SPICELIB/
mv $SPICELIB/$CORENLP/stanford-corenlp-3.6.0-models.jar $SPICELIB/
rm -f stanford-corenlp-full-2015-12-09.zip

echo "Done."
