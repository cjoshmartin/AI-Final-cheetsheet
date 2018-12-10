run: 
	pandoc cheatsheet.md -o cheatsheet.pdf 

install_pandoc:
	/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
	brew install pandoc
	brew tap caskroom/cask
	brew cask install basictex
