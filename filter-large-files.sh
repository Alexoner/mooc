#!/bin/sh

# List all commits in the history of these branches. For each one, list all the files; descend into directories recursively; include the size of the file. Cut out everything before the size of the file (which starts at character 54). Anything that starts with space is under a million bytes, so skip it. Now, choose only the unique lines; that's approximately the unique large revisions. Sum the sizes for each filename, and output these biggest-first. Store the output in a file.
git rev-list master | while read rev; do git ls-tree -lr $rev | cut -c54- | grep -v '^ '; done | sort -u | perl -e '
  while (<>) {
    chomp;
    @stuff=split("\t");
    $sums{$stuff[1]} += $stuff[0];
  }
  print "$sums{$_} $_\n" for (keys %sums);
' | sort -rn >> large_files.txt

# For any file you want to keep in the history, delete its line from large_files.txt.
# Create an alternate universe with a history that looks like <BRANCHES>, except for each commit, take its files and remove everything in large_files.txt (which contains the filename in the second space-delimited field). Drop any commits which only affected files that don't exist anymore. Point <BRANCHES> at this new version of history<Paste>
git filter-branch --tree-filter "rm -rf $(cat /Users/Alex/Documents/mine/mooc/large_files.txt | cut -d " " -f 2 ) " --prune-empty master
