how to use tmux to run sessions on server
```
sudo apt install tmux
```

create a session with a name:
```
tmux new -s session_name
```
to detach from the session:
Ctrl+b d 

reattach to session:
```
tmux attach-session -t session_name
```
to see different sessions running:
```
tmux ls
```
Ctrl+b c Create a new window (with shell)
Ctrl+b w Choose window from a list

Ctrl+b 0 Switch to window 0 (by number )
Ctrl+b , Rename the current window
Ctrl+b % Split current pane horizontally into two panes
Ctrl+b " Split current pane vertically into two panes

Ctrl+b o Go to the next pane
Ctrl+b ; Toggle between the current and previous pane
Ctrl+b x Close the current pane


