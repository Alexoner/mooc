#!/bin/sh

git filter-branch --prune-empty --tree-filter '
git lfs track "*.pdf"
git lfs track "*.exe"
git lfs track "*.conll"
git lfs track "*.pptx"
git lfs track "output/*.txt"

git add .gitattributes

git ls-files -z | xargs -0 git check-attr filter | grep "filter: lfs" | sed -E "s/(.*): filter: lfs/\1/" | tr "\n" "\0" | while read -r -d $'"'\0'"' file; do
    echo "Processing ${file}"

    git rm -f --cached "${file}"
    echo "Adding $file lfs style"
    git add "${file}"
done

' --tag-name-filter cat -- --all
