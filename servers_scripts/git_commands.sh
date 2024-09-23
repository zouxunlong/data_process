


git config credential.helper store


# Initialize the names
git config --global user.name "Zou Xunlong"
git config --global user.email "zouxunlong1988@gmail.com"

# Git initialize the project
git init

git add commands.sh
git add file2.txt file3.txt

git commit -m "wrote a command file"

git status

git diff command.sh 

git reflog

git diff HEAD -- readme.txt

git checkout -- readme.txt

git reset HEAD readme.txt

git rm test.txt