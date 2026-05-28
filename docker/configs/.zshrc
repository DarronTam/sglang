export ZSH="/root/.oh-my-zsh"

if [[ -z "$TERM" || "$TERM" == dumb ]]; then
    export TERM=xterm-256color
fi

# Theme
ZSH_THEME="robbyrussell"

# Plugins
plugins=(
    git
    z
    zsh-autosuggestions
    zsh-syntax-highlighting
)

if [[ -f "$ZSH/oh-my-zsh.sh" ]]; then
    source "$ZSH/oh-my-zsh.sh"
else
    autoload -Uz compinit && compinit
fi

autoload -Uz colors && colors

export CLICOLOR=1
if command -v dircolors >/dev/null 2>&1; then
    eval "$(dircolors -b)"
fi

zstyle ':completion:*' list-colors "${(s.:.)LS_COLORS}"

# Aliases
alias ls='ls --color=auto'
alias ll='ls -alF --color=auto'
alias la='ls -A --color=auto'
alias l='ls -CF --color=auto'
alias grep='grep --color=auto'
alias egrep='egrep --color=auto'
alias fgrep='fgrep --color=auto'
alias vi='vim'

# Enhanced history
HISTSIZE=10000
SAVEHIST=10000
setopt HIST_IGNORE_ALL_DUPS
setopt HIST_FIND_NO_DUPS
setopt INC_APPEND_HISTORY
